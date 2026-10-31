"""Ensembles and test-time augmentation: evaluate_candidates, ensemble registration, and
final_test scoring exactly the model chosen on VALIDATION."""

import importlib
import json
from pathlib import Path

import pytest
from conftest import make_config
from test_agent import ScriptedModel, read_jsonl

import gotrain.tools.candidates as candidates_mod
from gotrain.agent import build_agent, load_prompt
from gotrain.audit import AuditLog
from gotrain.contracts import ModelVersion
from gotrain.engine.fake import ENSEMBLE_GAIN, TTA_GAIN
from gotrain.tools import (
    evaluate_candidates,
    final_test,
    init_task_budget,
    leaderboard,
    register_model,
    run_trials,
)
from gotrain.tools.registry import load_version
from gotrain.tools.trials import budget_status

TASK = "m-eurosat"
final_test_module = importlib.import_module("gotrain.tools.final_test")  # not the function


@pytest.fixture
def runs(home) -> list[str]:
    """Three finished m-eurosat runs, best first."""
    init_task_budget(TASK, 60)
    configs = [make_config(lr=lr, reason=f"lr {lr}") for lr in (5e-4, 1e-3, 2e-4)]
    run_trials(TASK, configs, 1.0)
    return [row["run_id"] for row in leaderboard(TASK)["top"]]


def cand(run_ids: list[str], tta: bool = False) -> dict:
    return {"run_ids": run_ids, "tta": tta}


# ---------------------------------------------------------------- evaluate_candidates


def test_candidates_are_ranked_against_the_single_best(runs):
    out = evaluate_candidates(TASK, [cand(runs[:1], True), cand(runs[:2]), cand(runs, True)])
    assert "error" not in out and out["single_best_run"] == runs[0]
    ranked = out["ranked"]
    assert [r["rank"] for r in ranked] == [1, 2, 3, 4]  # 3 candidates + the baseline
    scores = [r["val_score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)
    by_key = {(len(r["run_ids"]), r["tta"]): r for r in ranked}
    assert by_key[(1, False)]["run_ids"] == runs[:1]
    assert by_key[(1, False)]["delta_vs_single_best"] == 0.0
    for r in ranked:
        assert r["delta_vs_single_best"] == pytest.approx(
            r["val_score"] - by_key[(1, False)]["val_score"], abs=1e-9
        )
    # The fake engine's TTA gain shows up against the same run without TTA.
    assert by_key[(1, True)]["delta_vs_single_best"] == pytest.approx(TTA_GAIN, abs=0.002)
    assert ENSEMBLE_GAIN > 0 and "SIMPLER" in out["rules"]


def test_baseline_is_not_duplicated_when_requested(runs):
    out = evaluate_candidates(TASK, [cand(runs[:1]), cand(runs[:1], True)])
    assert len(out["ranked"]) == 2


class Fixed:
    """An engine whose validation score depends only on (members, tta)."""

    def __init__(self, scores: dict[tuple[int, bool], float]):
        self.scores = scores

    def evaluate(self, task, checkpoints, split, tta=False):
        assert split == "val"
        n = len(checkpoints) if isinstance(checkpoints, list) else 1
        return {"score": self.scores[(n, tta)], "metric_name": "accuracy"}


@pytest.mark.parametrize(
    "scores, noise, asked, expected",
    [
        # All tie exactly: the plain single run wins.
        ({(1, False): 0.9, (1, True): 0.9, (3, False): 0.9}, None, [(1, True), (3, False)],
         (1, False)),
        # Single + TTA and top-2 tie on top: fewer members beats "no TTA".
        ({(1, False): 0.89, (1, True): 0.9, (2, False): 0.9}, None, [(1, True), (2, False)],
         (1, True)),
        # Same members: no TTA beats TTA.
        ({(1, False): 0.89, (2, False): 0.9, (2, True): 0.9}, None, [(2, False), (2, True)],
         (2, False)),
        # Within the seed noise counts as a tie; outside it the better score wins.
        ({(1, False): 0.900, (3, True): 0.904}, 0.005, [(3, True)], (1, False)),
        ({(1, False): 0.900, (3, True): 0.904}, 0.002, [(3, True)], (3, True)),
    ],
)  # fmt: skip
def test_recommendation_prefers_the_simpler_candidate_on_ties(
    runs, monkeypatch, scores, noise, asked, expected
):
    monkeypatch.setattr(candidates_mod, "get_engine", lambda: Fixed(scores))
    monkeypatch.setattr(candidates_mod, "seed_noise", lambda task: noise)
    out = evaluate_candidates(TASK, [cand(runs[:n], tta) for n, tta in asked])
    pick = out["recommended"]
    assert (len(pick["run_ids"]), pick["tta"]) == expected
    assert ("note" in out) == (noise is None)


def test_seed_noise_comes_from_confirm_seeds(runs):
    from gotrain.tools import confirm_seeds

    assert candidates_mod.seed_noise(TASK) is None
    confirm_seeds(runs[0], 2)
    assert candidates_mod.seed_noise(TASK) > 0


@pytest.mark.parametrize(
    "bad, message",
    [
        ("not a list", "must be a list of 1-6"),
        ([cand(["x"])] * 7, "must be a list of 1-6"),
        ([{"run_ids": []}], "candidate 0"),
        ([{"run_ids": ["a", "b", "c", "d", "e"]}], "candidate 0"),
        ([{"run_ids": ["a"], "weights": [1]}], "Extra inputs are not permitted"),
        ([cand(["m-eurosat-20260101-000000-abcd"])], "unknown run"),
    ],
)
def test_bad_candidates_are_rejected_before_anything_runs(runs, bad, message):
    before = budget_status(TASK)
    out = evaluate_candidates(TASK, bad)
    assert message in out["error"]
    assert budget_status(TASK) == before
    assert candidates_mod.load_candidate_results(TASK) == []


def test_members_must_be_ok_runs_of_this_task(runs, home):
    init_task_budget("m-brick-kiln", 60)
    other = run_trials("m-brick-kiln", [make_config()], 1.0)["trials"][0]["run_id"]
    timeout = run_trials(TASK, [make_config()], 0.25)["trials"][0]["run_id"]
    assert "belongs to m-brick-kiln" in evaluate_candidates(TASK, [cand([runs[0], other])])["error"]
    assert "status 'timeout'" in evaluate_candidates(TASK, [cand([timeout])])["error"]
    assert "lists a run twice" in evaluate_candidates(TASK, [cand([runs[0]] * 2)])["error"]
    dup = evaluate_candidates(TASK, [cand(runs[:2]), cand(runs[1::-1])])  # same set, reordered
    assert "repeats an earlier candidate" in dup["error"]


def test_inference_time_is_charged(runs):
    before = budget_status(TASK)["used_min"]
    out = evaluate_candidates(TASK, [cand(runs[:1], True), cand(runs)])
    # Fake m-eurosat: 1,000 val images x 64x64 px per view; TTA = 8 views; 1 + 8 + 3 views.
    per_view = 1000 * 64 * 64 / (20_000 * 64 * 64)
    assert out["minutes_charged"] == pytest.approx(12 * per_view, abs=1e-3)
    assert budget_status(TASK)["used_min"] == pytest.approx(before + 12 * per_view, abs=1e-3)
    assert len(candidates_mod.load_candidate_results(TASK)) == 3


def test_refused_up_front_when_the_estimate_does_not_fit(runs, monkeypatch):
    monkeypatch.setattr(candidates_mod, "estimate_minutes", lambda task, card, c: 1000.0)
    before = budget_status(TASK)
    out = evaluate_candidates(TASK, [cand(runs)])
    assert "estimated at" in out["error"] and "TTA" in out["hint"]
    assert budget_status(TASK) == before


# ---------------------------------------------------------------- registering an ensemble


def test_register_ensemble_needs_a_measured_score(runs):
    out = register_model(run_ids=runs[:2], tta=True)
    assert "no validation score yet" in out["error"]
    assert "no validation score yet" in register_model(run_id=runs[0], tta=True)["error"]
    assert "exactly one of" in register_model(run_id=runs[0], run_ids=runs[:2])["error"]
    assert "exactly one of" in register_model()["error"]


def test_register_ensemble_copies_members_and_records_tta(runs, home):
    scored = evaluate_candidates(TASK, [cand(runs[1::-1], True)])  # members listed worst first
    measured = next(r for r in scored["ranked"] if len(r["run_ids"]) == 2)["val_score"]

    out = register_model(run_ids=[runs[1], runs[0]], tta=True, notes="top-2 + TTA")
    assert "error" not in out and out["budget"] is not None
    assert out["members"] == runs[:2] and out["tta"] is True  # best member first
    mv = load_version(TASK, out["version"])
    assert (mv.run_id, mv.members, mv.tta, mv.val_score) == (runs[0], runs[:2], True, measured)
    version_dir = home / "registry" / TASK / out["version"]
    for rid in runs[:2]:
        assert (version_dir / "members" / rid / "checkpoints" / "best.ckpt").exists()
    card = (version_dir / "model_card.md").read_text(encoding="utf-8")
    assert "ensemble of 2 runs with TTA" in card and "## Members" in card
    assert all(f"`{rid}`" in card for rid in runs[:2])

    again = register_model(run_ids=runs[:2], tta=True)
    assert "already registered" in again["error"]


def test_single_run_registration_is_unchanged(runs, home):
    out = register_model(runs[0], "plain")
    mv = load_version(TASK, out["version"])
    assert (mv.members, mv.tta) == ([runs[0]], False)
    assert (home / "registry" / TASK / out["version"] / "checkpoints" / "best.ckpt").exists()


def test_old_model_versions_still_load():
    old = {
        "task": TASK, "version": "v1", "run_id": "r1", "config": make_config(task=TASK),
        "val_score": 0.9, "created_at": "2026-10-31T00:00:00Z", "git_commit": "abc",
        "artifact_path": "/x",
    }  # fmt: skip
    mv = ModelVersion.model_validate(old)
    assert (mv.members, mv.tta) == (["r1"], False)
    with pytest.raises(ValueError, match="run_id must be one of the members"):
        ModelVersion.model_validate(old | {"members": ["r2"]})


# ---------------------------------------------------------------- final_test


class Recorder:
    def __init__(self):
        self.calls = []

    def evaluate(self, task, checkpoint_path, split, **kwargs):
        self.calls.append((checkpoint_path, split, kwargs))
        return {"score": 0.97, "metric_name": "accuracy"}


def test_final_test_scores_the_registered_ensemble_with_tta(runs, home, monkeypatch):
    evaluate_candidates(TASK, [cand(runs, True)])
    version = register_model(run_ids=runs, tta=True)["version"]
    recorder = Recorder()
    monkeypatch.setattr(final_test_module, "get_engine", lambda: recorder)

    out = final_test(TASK, version)
    ((checkpoints, split, kwargs),) = recorder.calls
    assert split == "test" and kwargs == {"tta": True}
    members_dir = home / "registry" / TASK / version / "members"
    assert checkpoints == [members_dir / rid / "checkpoints" / "best.ckpt" for rid in runs]
    assert all(Path(p).exists() for p in checkpoints)
    assert out["members"] == runs and out["tta"] is True


def test_final_test_single_run_is_called_as_before(runs, home, monkeypatch):
    version = register_model(runs[0])["version"]
    recorder = Recorder()
    monkeypatch.setattr(final_test_module, "get_engine", lambda: recorder)
    final_test(TASK, version)
    ((checkpoint, split, kwargs),) = recorder.calls
    assert checkpoint == home / "registry" / TASK / version / "checkpoints" / "best.ckpt"
    assert (split, kwargs) == ("test", {})


def test_final_test_with_the_fake_engine_end_to_end(runs):
    evaluate_candidates(TASK, [cand(runs[:2])])
    version = register_model(run_ids=runs[:2])["version"]
    out = final_test(TASK, version)
    assert "error" not in out and 0 < out["test_score"] <= 1 and out["locked"]


# ---------------------------------------------------------------- session quota and prompt


def test_third_evaluate_candidates_call_is_blocked(runs, home):
    call = ("tool", "evaluate_candidates", {"task": TASK, "candidates": [cand(runs[:2])]})
    bad = ("tool", "evaluate_candidates", {"task": TASK, "candidates": "oops"})
    script = [[("text", "Try."), call], [("text", "Typo."), bad], [("text", "Again."), call],
              [("text", "Once more."), call]]  # fmt: skip
    audit = AuditLog(TASK, "tune", session_id="quota")
    agent = build_agent(TASK, "tune", audit=audit, model=ScriptedModel(script))
    agent("go")

    calls = [r for r in read_jsonl(home, audit) if r["event"] == "tool_call"]
    assert [c["cancelled"] for c in calls] == [False, False, False, True]  # errors don't count
    assert "already used 2 times" in calls[-1]["error"]
    assert any(r["event"] == "selection_quota_enforced" for r in read_jsonl(home, audit))
    assert len({json.dumps(r) for r in candidates_mod.load_candidate_results(TASK)}) == 2 * 2


def test_prompt_documents_the_selection_rules():
    prompt = " ".join(load_prompt("tune").split())  # ignore line wrapping
    assert "`evaluate_candidates`" in prompt
    assert "choose the SIMPLER one: fewer members first, then no TTA" in prompt
    assert "At most 2 `evaluate_candidates` calls per session" in prompt
    assert "target zone or exploration is done" in prompt
    assert "VALIDATION" in prompt
    assert "same members and TTA flag" in load_prompt("finalise")
