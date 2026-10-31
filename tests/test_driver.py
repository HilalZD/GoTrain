"""The session driver: continuations, the registration safety net, and the observation gate."""

import json

from conftest import make_config
from test_agent import ScriptedModel

from gotrain.agent import build_agent, make_limit
from gotrain.audit import AuditLog
from gotrain.contracts import ModelVersion
from gotrain.driver import AUTO_REGISTER_NOTE, best_for_registration, run_session
from gotrain.tools import (
    confirm_seeds,
    init_task_budget,
    leaderboard,
    register_model,
    run_trials,
)
from gotrain.tools.memory import all_lessons

TASK = "m-eurosat"  # fake: 0.1 min/epoch
TRIAL = {"task": TASK, "configs": [make_config()], "budget_minutes_per_trial": 2.0}
OBSERVATION = (
    "VISUAL OBSERVATION: rivers are confused with highways. "
    "HYPOTHESIS: more rot90 augmentation will help."
)


def best_run_id() -> str:
    return leaderboard(TASK)["top"][0]["run_id"]


def session(script, *, budget=40, max_tool_calls=None, repeat_last=False, phase="tune"):
    """Run a scripted session through the driver; returns (outcome, audit events, audit)."""
    init_task_budget(TASK, budget)
    audit = AuditLog(TASK, phase, session_id="s")
    limit = make_limit(phase, max_tool_calls)
    model = ScriptedModel(script, repeat_last=repeat_last)
    agent = build_agent(TASK, phase, audit=audit, limit=limit, model=model)
    outcome = run_session(agent, audit, TASK, phase, "go", limit)
    lines = audit._storage.read_text(audit.jsonl_key).splitlines()
    return outcome, [json.loads(line) for line in lines], audit


def of(events: list[dict], kind: str) -> list[dict]:
    return [e for e in events if e["event"] == kind]


# ---------------------------------------------------------------- continuations


def test_text_only_reply_mid_plan_triggers_continuation_and_agent_resumes():
    script = [
        [("tool", "run_trials", TRIAL)],
        [("text", OBSERVATION + " I'm about to run them.")],  # text only: the loop ends here
        lambda: [
            ("text", "Registering the best."),
            ("tool", "register_model", {"run_id": best_run_id()}),
        ],
        [("text", "Done.")],
    ]
    outcome, events, _ = session(script)
    assert outcome["continuations"] == 1
    assert outcome["registered_by"] == "agent" and outcome["version"] == "v1"
    (cont,) = of(events, "continuation")
    assert cont["message"].startswith("Continue your plan. Remaining budget: 38 min.")
    assert "register_model" in cont["message"]
    assert not of(events, "auto_registered")


def test_stops_after_three_continuations_then_safety_net(home):
    script = [[("tool", "run_trials", TRIAL)], [("text", "Thinking about it...")]]
    outcome, events, audit = session(script, repeat_last=True)
    assert outcome["continuations"] == 3
    assert [e["n"] for e in of(events, "continuation")] == [1, 2, 3]
    assert len(of(events, "continuations_exhausted")) == 1
    assert outcome["registered_by"] == "harness"  # nothing registered by the agent
    journal = (home / audit.journal_key).read_text(encoding="utf-8")
    assert journal.count("HARNESS: continuation") == 3


def test_no_continuation_when_budget_is_below_one_short_trial():
    outcome, events, _ = session([[("tool", "run_trials", TRIAL)], [("text", "Done.")]], budget=3)
    assert outcome["continuations"] == 0 and not of(events, "continuation")  # 1 min < 2 min


def test_no_continuation_once_the_tool_call_limit_ended_the_loop():
    looping = [[("tool", "read_lessons", {"tags": ["x"]})]]
    outcome, events, _ = session(looping, max_tool_calls=2, repeat_last=True)
    assert outcome["continuations"] == 0 and not of(events, "continuation")


def test_finalise_phase_has_no_continuation_or_safety_net():
    outcome, events, _ = session([[("text", "Nothing to do.")]], phase="finalise")
    assert outcome["continuations"] == 0 and outcome["registered_by"] is None
    assert not of(events, "auto_registered") and not of(events, "auto_register_skipped")


# ---------------------------------------------------------------- safety net


def test_safety_net_registers_best_run_when_agent_never_does(home):
    script = [
        [("tool", "run_trials", {**TRIAL, "configs": [make_config(seed=s) for s in range(3)]})]
    ]
    outcome, events, audit = session(script, budget=7)  # 1 min left: no continuation
    assert outcome["continuations"] == 0
    assert outcome["registered_by"] == "harness" and outcome["version"] == "v1"

    (auto,) = of(events, "auto_registered")
    assert auto["run_id"] == best_run_id() and "single-seed" in auto["basis"]
    mv = ModelVersion.model_validate_json(
        (home / "registry" / TASK / "v1" / "model_version.json").read_text()
    )
    assert mv.run_id == best_run_id()
    card = (home / "registry" / TASK / "v1" / "model_card.md").read_text(encoding="utf-8")
    assert AUTO_REGISTER_NOTE in card
    journal = (home / audit.journal_key).read_text(encoding="utf-8")
    assert "HARNESS: AUTO-REGISTERED v1" in journal and "not chosen by the agent" in journal


def test_safety_net_skips_when_no_run_succeeded():
    bad = {**TRIAL, "configs": [make_config(bands=["B99"])]}
    outcome, events, _ = session([[("tool", "run_trials", bad)]], budget=3)
    assert outcome["registered_by"] is None
    assert of(events, "auto_register_skipped")[0]["reason"] == "no successful run"


def test_best_for_registration_uses_seed_group_mean():
    init_task_budget(TASK, 100)
    ids = [
        run_trials(TASK, [make_config(lr=lr, reason=f"lr {lr}")], 2.0)["trials"][0]["run_id"]
        for lr in (5e-4, 1e-3, 2e-3)
    ]
    for run_id in ids[:2]:
        confirm_seeds(run_id, 2)
    top_group = leaderboard(TASK)["seed_groups"][0]
    single = next(r for r in leaderboard(TASK, 50)["top"] if r["run_id"] == ids[2])

    entry, basis = best_for_registration(TASK)
    assert top_group["mean"] > single["val_score"]  # lr 2e-3 is clearly worse in the fake
    assert entry["run_id"] == top_group["base_run_id"]
    assert top_group["seed_group"] in basis and top_group["mean_std"] in basis


def test_safety_net_does_not_override_an_agent_registration():
    script = [
        [("tool", "run_trials", TRIAL)],
        lambda: [("tool", "register_model", {"run_id": best_run_id()})],
        [("text", "Done.")],
    ]
    outcome, events, _ = session(script)
    assert outcome["registered_by"] == "agent" and not of(events, "auto_registered")


def test_existing_registration_from_an_earlier_session_is_not_duplicated():
    init_task_budget(TASK, 40)
    run_id = run_trials(TASK, [make_config()], 2.0)["trials"][0]["run_id"]
    register_model(run_id)  # an earlier session
    outcome, events, _ = session([[("text", "Done.")]], budget=40)
    assert outcome["registered_by"] is None
    assert "already registered" in of(events, "auto_register_failed")[0]["error"]


# ---------------------------------------------------------------- observation gate


def test_observation_in_tool_args_passes_the_gate():
    script = [
        [("tool", "run_trials", TRIAL)],
        lambda: [("tool", "diagnose", {"run_id": best_run_id()})],
        [("tool", "run_trials", {**TRIAL, "configs": [make_config(reason=OBSERVATION)]})],
        lambda: [("tool", "register_model", {"run_id": best_run_id()})],
    ]
    outcome, events, _ = session(script)
    calls = [(c["tool"], c["cancelled"]) for c in of(events, "tool_call")]
    assert calls == [
        ("run_trials", False),
        ("diagnose", False),
        ("run_trials", False),
        ("register_model", False),
    ]
    assert not of(events, "observation_enforced")
    (written,) = of(events, "observation_written")
    assert written["source"] == "tool_args"


def test_gate_message_says_same_response_and_do_not_end_turn():
    script = [
        [("tool", "run_trials", TRIAL)],
        lambda: [("tool", "diagnose", {"run_id": best_run_id()})],
        [("tool", "leaderboard", {"task": TASK})],
        [("text", OBSERVATION), ("tool", "leaderboard", {"task": TASK})],
        lambda: [("tool", "register_model", {"run_id": best_run_id()})],
    ]
    _, events, _ = session(script)
    blocked = next(c for c in of(events, "tool_call") if c["cancelled"])
    assert "AND include your next tool call in the SAME response" in blocked["error"]
    assert "do not end your turn" in blocked["error"]
    assert of(events, "observation_written")[0]["source"] == "message"


# ---------------------------------------------------------------- lessons written during tune


def tune_lessons() -> list[dict]:
    return [x for x in all_lessons() if "source:tune" in x["tags"]]


def test_agent_lessons_are_written_with_the_tune_tag():
    reply = json.dumps(
        [
            {"lesson": "rot90 helped by 0.01 on val", "tags": ["rot90"]},
            {"lesson": "lr 2e-3 was too high (val 0.80)", "tags": []},
        ]
    )
    script = [
        [("tool", "run_trials", TRIAL)],
        lambda: [("tool", "register_model", {"run_id": best_run_id()})],
        [("text", "Done.")],
        [("text", f"Here you go: {reply}")],
    ]
    outcome, events, _ = session(script)
    assert outcome["lessons_written"] == 2
    assert len(tune_lessons()) == 2 and "rot90" in tune_lessons()[0]["tags"]
    assert of(events, "session_lessons")[0]["source"] == "agent"


def test_failed_lesson_request_falls_back_to_an_automatic_lesson():
    script = [
        [("tool", "run_trials", TRIAL)],
        lambda: [("tool", "register_model", {"run_id": best_run_id()})],
        [("text", "Done.")],
        [("text", "I have no lessons in JSON form.")],
    ]
    outcome, events, _ = session(script)
    assert outcome["lessons_written"] == 1
    (lesson,) = tune_lessons()
    assert "auto-written by harness" in lesson["lesson"] and "val accuracy" in lesson["lesson"]
    assert of(events, "session_lessons")[0]["source"] == "automatic"


def test_harness_registered_session_also_writes_lessons():
    outcome, _, _ = session([[("tool", "run_trials", TRIAL)]], budget=3)
    assert outcome["registered_by"] == "harness" and outcome["lessons_written"] == 1


def test_finalise_writes_no_tune_lessons():
    session([[("text", "Nothing to do.")]], phase="finalise")
    assert tune_lessons() == []


# ---------------------------------------------------------------- GoPilot export


def test_tune_registration_exports_the_bundle_and_journals_the_tool_name(home):
    script = [
        [("tool", "run_trials", TRIAL)],
        lambda: [("tool", "register_model", {"run_id": best_run_id()})],
        [("text", "Done.")],
    ]
    outcome, events, audit = session(script)
    (export,) = of(events, "gopilot_export")
    assert export["tool_name"] == "classify_land_cover_eurosat" and export["version"] == "v1"
    bundle = json.loads((home / "registry" / TASK / "v1" / "gopilot" / "bundle.json").read_text())
    assert bundle["test"] is None
    journal = (home / audit.journal_key).read_text(encoding="utf-8")
    assert "`classify_land_cover_eurosat`" in journal


def test_finalise_exports_again_with_the_test_score(home):
    init_task_budget(TASK, 40)
    register_model(run_trials(TASK, [make_config()], 2.0)["trials"][0]["run_id"])
    script = [[("tool", "final_test", {"task": TASK, "version": "v1"})], [("text", "Done.")]]
    outcome, events, _ = session(script, phase="finalise")
    (export,) = of(events, "gopilot_export")
    assert export["when"].startswith("after finalise")
    bundle = json.loads((home / "registry" / TASK / "v1" / "gopilot" / "bundle.json").read_text())
    assert 0 < bundle["test"]["test_score"] <= 1
