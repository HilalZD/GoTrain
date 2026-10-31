"""Guards from the live m-pv4ger-seg run: the observation gate, fail-fast, and screen sizing."""

import json

import pytest
from conftest import make_config
from test_agent import ScriptedModel

from gotrain.agent import build_agent
from gotrain.audit import AuditLog
from gotrain.contracts import TrainConfig
from gotrain.engine.fake import CARDS, FakeEngine, minutes_per_epoch
from gotrain.engine.throughput import card_minutes_per_epoch, screening_advice
from gotrain.tools import init_task_budget, inspect_dataset, run_halving, run_trials

SEG = "m-pv4ger-seg"  # fake: 3.75 min per full-data epoch at 320 px
OBSERVATION = (
    "VISUAL OBSERVATION: thin panel edges are missed and small panels vanish.\n"
    "HYPOTHESIS: misses thin panel edges -> try dice loss at full resolution."
)


def seg_config(**overrides) -> dict:
    return make_config(weights="imagenet", img_size=320, loss="dice_bce", **overrides)


def events(home, audit: AuditLog) -> list[dict]:
    lines = (home / audit.jsonl_key).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


@pytest.fixture
def seg_run() -> str:
    init_task_budget(SEG, 100)
    return run_trials(SEG, [seg_config()], 10.0)["trials"][0]["run_id"]


def run_script(task: str, script: list[list[tuple]], session: str):
    audit = AuditLog(task, "tune", session_id=session)
    agent = build_agent(task, "tune", audit=audit, model=ScriptedModel(script))
    agent("go")
    return audit


# ---------------------------------------------------------------- observation gate


def test_gate_blocks_tool_calls_after_diagnose_until_observation(home, seg_run):
    script = [
        [("tool", "diagnose", {"run_id": seg_run})],
        [("text", "Let me check the board."), ("tool", "leaderboard", {"task": SEG})],
        [("text", OBSERVATION), ("tool", "leaderboard", {"task": SEG})],
        [("text", "done")],
    ]
    log = events(home, run_script(SEG, script, "gate"))
    calls = [e for e in log if e["event"] == "tool_call"]
    assert [(c["tool"], c["cancelled"]) for c in calls] == [
        ("diagnose", False),
        ("leaderboard", True),
        ("leaderboard", False),
    ]
    assert "VISUAL OBSERVATION:" in calls[1]["error"] and "HYPOTHESIS:" in calls[1]["error"]

    (blocked,) = [e for e in log if e["event"] == "observation_enforced"]
    assert blocked["blocked_tool"] == "leaderboard" and blocked["diagnosed_run"] == seg_run
    (written,) = [e for e in log if e["event"] == "observation_written"]
    assert written["diagnosed_run"] == seg_run


def test_gate_blocks_every_tool_until_observation(home, seg_run):
    script = [
        [("tool", "diagnose", {"run_id": seg_run})],
        [("tool", "read_lessons", {"tags": ["x"]})],
        [("tool", "diagnose", {"run_id": seg_run})],  # even diagnose itself
        [("text", "done")],
    ]
    log = events(home, run_script(SEG, script, "every"))
    blocked = [e["blocked_tool"] for e in log if e["event"] == "observation_enforced"]
    assert blocked == ["read_lessons", "diagnose"]


def test_gate_ignores_failed_diagnose(home):
    init_task_budget(SEG, 10)
    script = [
        [("tool", "diagnose", {"run_id": "no-such-run"})],
        [("tool", "leaderboard", {"task": SEG})],
        [("text", "done")],
    ]
    log = events(home, run_script(SEG, script, "failed"))
    assert not [e for e in log if e["event"] == "observation_enforced"]


def test_diagnose_text_ends_with_reminder(seg_run):
    audit = AuditLog(SEG, "tune", session_id="reminder")
    agent = build_agent(SEG, "tune", audit=audit, model=ScriptedModel([]))
    text = agent.tool.diagnose(run_id=seg_run)["content"][0]["text"]
    assert text.endswith(
        '"next_step": "Next: write VISUAL OBSERVATION and HYPOTHESIS before any other tool call."}'
    )


# ---------------------------------------------------------------- fail fast


def test_run_trials_stops_after_first_zero_epoch_timeout(home):
    init_task_budget(SEG, 100)
    out = run_trials(SEG, [seg_config(seed=s) for s in range(4)], 1.5)
    (trial,) = out["trials"]
    assert trial["status"] == "timeout"
    assert [n["index"] for n in out["not_run"]] == [1, 2, 3]
    assert out["budget"]["used_min"] == pytest.approx(1.5)  # only the one trial
    (warning,) = out["warnings"]
    assert f"trial {trial['run_id']} ran 1.5 min" in warning
    assert "train_subset_frac <= 0.1" in warning and "budget_minutes_per_trial >=" in warning
    assert len(list((home / "runs" / SEG).iterdir())) == 1  # nothing else started


def test_run_halving_fails_fast_and_charges_only_used_minutes(home):
    init_task_budget(SEG, 100)
    configs = [seg_config(train_subset_frac=0.1, reason="fits"), seg_config(reason="too big")]
    configs += [seg_config(seed=s, reason="never") for s in range(4)]
    out = run_halving(SEG, configs, 1.5, 4.0, 2)
    assert out["stopped_early"] is True and out["winner"] is None
    assert [t["status"] for t in out["trials"]] == ["ok", "timeout"]
    assert all(t["rung"] == 1 for t in out["trials"])  # no rung 2
    assert [n["index"] for n in out["not_run"]] == [2, 3, 4, 5]
    assert out["budget"]["used_min"] == pytest.approx(sum(t["minutes"] for t in out["trials"]))
    assert "short_minutes >=" in out["warnings"][0]


def test_errors_do_not_stop_the_round(home):
    init_task_budget("m-eurosat", 100)
    out = run_trials("m-eurosat", [make_config(bands=["B99"]), make_config()], 1.0)
    assert sorted(t["status"] for t in out["trials"]) == ["error", "ok"]
    assert out["not_run"] == [] and out["warnings"] == []


# ---------------------------------------------------------------- screen sizing


def test_screening_hint_fires_for_pv4ger():
    hint = inspect_dataset(SEG)["dataset"]["screening_hint"]
    assert "3000 samples at 320x320 px" in hint and "~3.8 min" in hint
    assert "train_subset_frac <= 0.15" in hint and "short_minutes >= 11.2" in hint


@pytest.mark.parametrize(
    "task, fires", [("m-eurosat", False), ("m-brick-kiln", True), ("m-chesapeake", True)]
)
def test_screening_hint_by_task(task, fires):
    assert (screening_advice(CARDS[task]) is not None) == fires


def test_fake_engine_timing_matches_the_estimate():
    for task, card in CARDS.items():
        cfg = TrainConfig(task=task, **make_config(img_size=card.image_size[0], weights="imagenet"))
        assert minutes_per_epoch(cfg, card) == pytest.approx(card_minutes_per_epoch(card))
    notes = FakeEngine(sleep_s=0).describe_dataset(SEG).notes
    assert "train_samples x train_subset_frac x img_size^2" in notes
