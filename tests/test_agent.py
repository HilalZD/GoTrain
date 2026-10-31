"""The agent, audit hook, tool-call guard and CLI, with a scripted model instead of Bedrock."""

import json
from itertools import count
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import make_config
from strands.models import Model
from strands.tools.executors import SequentialToolExecutor

from gotrain import cli
from gotrain.agent import (
    PHASE_TOOLS,
    ToolCallLimit,
    build_agent,
    build_model,
    kickoff_message,
)
from gotrain.audit import MAX_FIELD_CHARS, AuditLog, result_error, result_text, truncate
from gotrain.config import get_settings
from gotrain.tools import init_task_budget, register_model, run_trials

TASK = "m-eurosat"


class ScriptedModel(Model):
    """Replays a fixed list of assistant turns. Each turn is a list of
    ("text", str) or ("tool", name, input) blocks, or a function returning one; when the
    script runs out it says "done"."""

    def __init__(self, turns: list[list[tuple]], repeat_last: bool = False):
        self.turns = list(turns)
        self.repeat_last = repeat_last
        self.ids = count()

    def update_config(self, **model_config: Any) -> None:
        pass

    def get_config(self) -> dict:
        return {}

    def structured_output(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        if len(self.turns) > 1 or (self.turns and not self.repeat_last):
            turn = self.turns.pop(0)
        else:
            turn = self.turns[0] if self.turns else [("text", "done")]
        if callable(turn):  # built at call time, e.g. to use a run_id made earlier
            turn = turn()
        yield {"messageStart": {"role": "assistant"}}
        for block in turn:
            if block[0] == "text":
                yield {"contentBlockStart": {"start": {}}}
                yield {"contentBlockDelta": {"delta": {"text": block[1]}}}
            else:
                start = {"toolUse": {"toolUseId": f"t{next(self.ids)}", "name": block[1]}}
                yield {"contentBlockStart": {"start": start}}
                yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(block[2])}}}}
            yield {"contentBlockStop": {}}
        uses_tool = any(block[0] == "tool" for block in turn)
        yield {"messageStop": {"stopReason": "tool_use" if uses_tool else "end_turn"}}


def read_jsonl(home, audit: AuditLog) -> list[dict]:
    lines = (home / audit.jsonl_key).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


# ---------------------------------------------------------------- phases


def test_tune_tools_exclude_final_test():
    agent = build_agent(TASK, "tune", model=ScriptedModel([]))
    assert "final_test" not in agent.tool_names
    assert set(agent.tool_names) == {
        "inspect_dataset",
        "run_halving",
        "run_trials",
        "promote",
        "confirm_seeds",
        "leaderboard",
        "diagnose",
        "register_model",
        "read_lessons",
    }


def test_finalise_tools_include_final_test():
    agent = build_agent(TASK, "finalise", model=ScriptedModel([]))
    assert set(agent.tool_names) == {"leaderboard", "final_test", "write_lesson", "read_lessons"}


def test_final_test_only_reachable_in_finalise():
    assert [
        p for p, tools in PHASE_TOOLS.items() if any(t.__name__ == "final_test" for t in tools)
    ] == ["finalise"]


def test_unknown_phase_rejected():
    with pytest.raises(ValueError, match="phase must be one of"):
        build_agent(TASK, "all", model=ScriptedModel([]))


def test_tools_run_sequentially_and_run_trials_documents_config_fields():
    agent = build_agent(TASK, "tune", model=ScriptedModel([]))
    assert isinstance(agent.tool_executor, SequentialToolExecutor)
    spec = agent.tool_registry.registry["run_trials"].tool_spec
    assert "- lr: Learning rate" in spec["description"]
    assert "Options: ce, bce, dice_ce, dice_bce, focal" in spec["description"]
    props = spec["inputSchema"]["json"]["properties"]
    assert "GPU minutes" in props["budget_minutes_per_trial"]["description"]


def test_bedrock_model_uses_settings_and_adaptive_retries():
    model = build_model(get_settings())
    assert model.config["model_id"] == "eu.anthropic.claude-sonnet-5-5"
    assert model.client.meta.region_name == "eu-central-1"
    assert model.client.meta.config.retries["mode"] == "adaptive"


def test_kickoff_lists_budget_and_versions():
    init_task_budget(TASK, 30)
    assert "30 min remaining" in kickoff_message(TASK, "tune")
    run_id = run_trials(TASK, [make_config()], 2.0)["trials"][0]["run_id"]
    register_model(run_id)
    assert "- v1: val_score=" in kickoff_message(TASK, "finalise")


# ---------------------------------------------------------------- audit


def test_audit_hook_writes_jsonl_and_journal(home):
    init_task_budget(TASK, 30)
    script = [
        [("text", "First I look at the data."), ("tool", "inspect_dataset", {"task": TASK})],
        [
            ("text", "Baseline trial."),
            (
                "tool",
                "run_trials",
                {"task": TASK, "configs": [make_config()], "budget_minutes_per_trial": 2.0},
            ),
        ],
        [("tool", "diagnose", {"run_id": "not-a-run"})],
        [("text", "What I learned: SSL4EO works. What I'll try next: nothing.")],
    ]
    audit = AuditLog(TASK, "tune", session_id="tune-test-0001")
    agent = build_agent(TASK, "tune", audit=audit, model=ScriptedModel(script))
    agent("go")

    records = read_jsonl(home, audit)
    assert all({"ts", "session_id", "task", "phase", "event"} <= r.keys() for r in records)
    assert {r["session_id"] for r in records} == {"tune-test-0001"}
    assert audit.jsonl_key == f"logs/{TASK}/tune-test-0001.jsonl"

    calls = [r for r in records if r["event"] == "tool_call"]
    assert [c["tool"] for c in calls] == ["inspect_dataset", "run_trials", "diagnose"]
    assert all(c["duration_s"] is not None and c["duration_s"] >= 0 for c in calls)
    assert '"budget_minutes_per_trial": 2.0' in calls[1]["args"]
    assert '"val_score"' in calls[1]["result"] and calls[1]["error"] is None
    assert "no run with run_id" in calls[2]["error"]

    texts = [r["text"] for r in records if r["event"] == "assistant_text"]
    assert texts[0] == "First I look at the data." and "What I learned" in texts[-1]
    assert sum(r["event"] == "tool_start" for r in records) == 3

    journal = (home / audit.journal_key).read_text(encoding="utf-8")
    assert "**Agent:** First I look at the data." in journal
    assert "-> `diagnose`" in journal and "**ERROR:**" in journal


def test_result_summary_is_truncated_and_errors_extracted():
    big = {"content": [{"text": "y" * 5000}], "status": "success"}
    short = truncate(result_text(big), MAX_FIELD_CHARS)
    assert short.endswith("[3000 chars cut]") and len(short) < MAX_FIELD_CHARS + 50

    tool_error = {"content": [{"text": '{"error": "bad lr", "hint": "x"}'}], "status": "success"}
    assert result_error(tool_error, result_text(tool_error)) == "bad lr"
    crashed = {"content": [{"text": "Error: boom"}], "status": "error"}
    assert result_error(crashed, result_text(crashed)) == "Error: boom"


# ---------------------------------------------------------------- guard


def test_tool_call_limit_stops_a_looping_agent(home):
    looping = [[("text", "Again."), ("tool", "read_lessons", {"tags": ["x"]})]]
    audit = AuditLog(TASK, "tune", session_id="loop")
    agent = build_agent(
        TASK, "tune", audit=audit, max_tool_calls=3, model=ScriptedModel(looping, repeat_last=True)
    )
    result = agent("go")

    calls = [r for r in read_jsonl(home, audit) if r["event"] == "tool_call"]
    executed = [c for c in calls if not c["cancelled"]]
    assert len(executed) == 3
    assert len(calls) == 3 + 3  # the rest were cancelled, then the loop was ended
    assert "Tool-call limit of 3 reached" in calls[-1]["error"]
    assert result.stop_reason == "end_turn"


def test_wrap_up_tools_still_allowed_after_limit():
    limit = ToolCallLimit(max_calls=1, wrap_up_tools={"register_model"}, grace=2)

    def call(name: str) -> SimpleNamespace:
        event = SimpleNamespace(tool_use={"name": name}, cancel_tool=False)
        limit._before_tool(event)
        return event

    assert call("run_trials").cancel_tool is False  # within the limit
    assert "limit of 1 reached" in call("run_trials").cancel_tool
    assert call("register_model").cancel_tool is False  # wrap-up allowed in grace
    assert call("register_model").cancel_tool  # grace used up


# ---------------------------------------------------------------- CLI


class FakeAgent:
    def __init__(self):
        self.messages = []

    def __call__(self, message: str):
        self.messages.append(message)
        return SimpleNamespace(stop_reason="end_turn", metrics=None)


@pytest.fixture
def fake_agent(monkeypatch):
    built = []

    def fake_build(task, phase, **kwargs):
        built.append((task, phase))
        return FakeAgent()

    monkeypatch.setattr(cli, "build_agent", fake_build)
    return built


def run_cli(argv: list[str], answer: str = "") -> int:
    return cli.run(cli.parse_args(argv), ask=lambda prompt: answer)


def test_cli_tune_sets_budget_and_runs(fake_agent, home):
    assert run_cli(["run", "--task", TASK, "--budget-min", "30", "--phase", "tune"]) == 0
    assert fake_agent == [(TASK, "tune")]
    assert json.loads((home / "registry" / TASK / "budget.json").read_text())["total_min"] == 30
    assert list((home / "logs" / TASK).glob("tune-*.jsonl"))


def test_cli_tune_requires_budget(fake_agent):
    assert run_cli(["run", "--task", TASK]) == 1 and fake_agent == []


def test_cli_has_no_all_mode():
    with pytest.raises(SystemExit):
        cli.parse_args(["run", "--task", TASK, "--phase", "all"])


@pytest.fixture
def registered():
    init_task_budget(TASK, 30)
    register_model(run_trials(TASK, [make_config()], 2.0)["trials"][0]["run_id"])


def test_cli_finalise_requires_typed_confirmation(fake_agent, registered):
    argv = ["run", "--task", TASK, "--phase", "finalise"]
    assert run_cli(argv, answer="") == 1
    assert run_cli(argv, answer="m-brick-kiln") == 1
    assert fake_agent == []
    assert run_cli(argv, answer=TASK) == 0
    assert fake_agent == [(TASK, "finalise")]


def test_cli_finalise_without_input_aborts(fake_agent, registered):
    def no_terminal(prompt: str) -> str:
        raise EOFError  # e.g. under nohup

    args = cli.parse_args(["run", "--task", TASK, "--phase", "finalise"])
    assert cli.run(args, ask=no_terminal) == 1 and fake_agent == []


def test_cli_finalise_refuses_without_versions_or_after_test(fake_agent, home, registered):
    assert run_cli(["run", "--task", "m-brick-kiln", "--phase", "finalise"], TASK) == 1
    (home / "registry" / TASK / "TEST_LOCK.json").write_text("{}")
    assert run_cli(["run", "--task", TASK, "--phase", "finalise"], answer=TASK) == 1
    assert fake_agent == []


def test_cli_refuses_fake_engine_on_real_home(fake_agent, monkeypatch, tmp_path):
    monkeypatch.setenv("GOTRAIN_HOME", str(tmp_path / "gotrain_home"))
    assert run_cli(["run", "--task", TASK, "--budget-min", "30"]) == 2
    assert fake_agent == []
