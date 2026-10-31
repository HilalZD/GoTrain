"""run_halving, confirm_seeds, seed noise in the fake engine, and mean ± std in leaderboard."""

import json
import statistics

import pytest
from conftest import make_config
from test_agent import ScriptedModel

from gotrain.agent import build_agent
from gotrain.audit import AuditLog
from gotrain.tools import confirm_seeds, init_task_budget, leaderboard, run_halving, run_trials

TASK = "m-eurosat"  # fake engine: 0.1 min/epoch at img 64 + 0.2 min setup


def diverse_configs() -> list[dict]:
    return [
        make_config(reason="baseline"),
        make_config(weights="imagenet", reason="imagenet"),
        make_config(weights="none", reason="scratch"),
        make_config(lr=3e-3, reason="high lr"),
        make_config(backbone="resnet18", reason="small"),
    ]


@pytest.fixture
def budget():
    init_task_budget(TASK, 100)


# ---------------------------------------------------------------- halving


def test_halving_keeps_exactly_keep_top_and_links_parents(budget, home):
    out = run_halving(TASK, diverse_configs(), 1.0, 3.0, 2)
    rung1 = [t for t in out["trials"] if t["rung"] == 1]
    rung2 = [t for t in out["trials"] if t["rung"] == 2]
    assert len(rung1) == 5 and len(rung2) == 2
    assert all(t["parent"] is None for t in rung1)

    best_two = sorted(rung1, key=lambda t: -t["val_score"])[:2]
    assert {t["parent"] for t in rung2} == {t["run_id"] for t in best_two}
    assert out["winner"] == max(rung2, key=lambda t: t["val_score"])
    assert set(out["winner"]) >= {"run_id", "rung", "val_score", "minutes", "reason", "parent"}

    board = {r["run_id"]: r for r in leaderboard(TASK, top_k=20)["top"]}
    assert all(board[t["run_id"]]["parent"] == t["parent"] for t in rung2)
    assert out["budget"]["used_min"] == pytest.approx(sum(t["minutes"] for t in out["trials"]))


def test_halving_refuses_when_over_budget(home):
    init_task_budget(TASK, 10)
    out = run_halving(TASK, diverse_configs(), 1.0, 4.0, 2)  # 5*1 + 2*4 = 13 > 10
    assert "could cost up to 13 min" in out["error"] and "only 10 min remain" in out["error"]
    assert "lower long_minutes to 2.5" in out["hint"] or "run only" in out["hint"]
    assert out["budget"]["used_min"] == 0
    assert not (home / "runs").exists()


def test_halving_rejects_bad_keep_top_and_reports_invalid(budget):
    configs = diverse_configs()[:2] + [make_config(backbone="convnext_tiny")]
    out = run_halving(TASK, configs, 1.0, 3.0, 2)  # only 2 valid: keep_top must be < 2
    assert "fewer than the 2 valid configs" in out["error"]
    assert out["invalid"][0]["index"] == 2


def test_halving_does_not_retrain_finished_runs(budget):
    configs = [make_config(max_epochs=4, seed=i) for i in range(3)]  # 4 epochs fit in 1 min
    out = run_halving(TASK, configs, 1.0, 3.0, 1)
    assert [t["rung"] for t in out["trials"]] == [1, 1, 1]
    assert "finished all epochs" in out["skipped"][0]["reason"]
    assert out["winner"]["rung"] == 1


def test_every_halving_trial_is_in_the_audit_log(budget, home):
    audit = AuditLog(TASK, "tune", session_id="halving")
    agent = build_agent(TASK, "tune", audit=audit, model=ScriptedModel([]))
    agent.tool.run_halving(
        task=TASK, configs=diverse_configs(), short_minutes=1.0, long_minutes=3.0, keep_top=2
    )
    lines = (home / audit.jsonl_key).read_text(encoding="utf-8").splitlines()
    trials = [r for r in map(json.loads, lines) if r["event"] == "trial"]
    assert sorted(t["rung"] for t in trials) == [1, 1, 1, 1, 1, 2, 2]
    assert all(t["tool"] == "run_halving" and t["val_score"] is not None for t in trials)
    assert "rung 2" in (home / audit.journal_key).read_text(encoding="utf-8")


# ---------------------------------------------------------------- seeds


def test_fake_seed_noise_is_realistic_and_deterministic(budget):
    configs = [make_config(seed=s) for s in range(8)]
    scores = [t["val_score"] for t in run_trials(TASK, configs, 2.0)["trials"]]
    assert 0.001 < statistics.stdev(scores) < 0.008  # roughly +-0.3 points
    again = run_trials(TASK, [make_config(seed=3)], 2.0)["trials"][0]["val_score"]
    assert again in scores


def test_confirm_seeds_builds_group_with_mean_and_std(budget):
    original = run_trials(TASK, [make_config()], 2.0)["trials"][0]
    out = confirm_seeds(original["run_id"], n_seeds=2)
    assert out["seed_group"] == f"sg-{original['run_id']}"
    assert len(out["trials"]) == 2 and out["n"] == 3
    assert out["scores"][0] == original["val_score"]
    assert out["mean"] == pytest.approx(statistics.mean(out["scores"]), abs=1e-4)
    assert out["std"] == pytest.approx(statistics.stdev(out["scores"]), abs=1e-4)
    assert all(t["minutes"] == original["minutes"] for t in out["trials"])  # same budget

    more = confirm_seeds(out["trials"][0]["run_id"], n_seeds=1)  # via a member: same group
    assert more["seed_group"] == out["seed_group"] and more["n"] == 4


def test_confirm_seeds_uses_new_seeds(budget, home):
    run_id = run_trials(TASK, [make_config(seed=0)], 2.0)["trials"][0]["run_id"]
    out = confirm_seeds(run_id, n_seeds=2)
    seeds = [
        json.loads((home / "runs" / TASK / t["run_id"] / "config.json").read_text())["seed"]
        for t in out["trials"]
    ]
    assert seeds == [1, 2]


def test_confirm_seeds_budget_checked(home):
    init_task_budget(TASK, 7)
    run_id = run_trials(TASK, [make_config()], 2.0)["trials"][0]["run_id"]  # 5 min left
    out = confirm_seeds(run_id, n_seeds=3)  # needs 6
    assert "needs 6 min but only 5 min" in out["error"] and "n_seeds=1" in out["hint"]
    assert "differs from the original" in confirm_seeds(run_id, 1, budget_minutes=1.0)["error"]


def test_confirm_seeds_refuses_failed_and_unknown_runs(budget):
    bad = run_trials(TASK, [make_config(bands=["B99"])], 2.0)["trials"][0]["run_id"]
    assert "only successful runs" in confirm_seeds(bad)["error"]
    assert "no run with run_id" in confirm_seeds("nope")["error"]


def test_leaderboard_shows_mean_and_std(budget):
    a = run_trials(TASK, [make_config(reason="a")], 2.0)["trials"][0]["run_id"]
    b = run_trials(TASK, [make_config(lr=1e-3, reason="b")], 2.0)["trials"][0]["run_id"]
    sa, sb = confirm_seeds(a, 2), confirm_seeds(b, 2)

    board = leaderboard(TASK, top_k=20)
    row = next(r for r in board["top"] if r["run_id"] == a)
    assert row["seed_group"] == sa["seed_group"]
    assert row["mean_std"] == f"{sa['mean']:.4f} ± {sa['std']:.4f} (n=3)"
    means = [g["mean"] for g in board["seed_groups"]]
    assert means == sorted(means, reverse=True) and set(means) == {sa["mean"], sb["mean"]}


def test_only_tune_agent_has_search_tools():
    tune = build_agent(TASK, "tune", model=ScriptedModel([]))
    finalise = build_agent(TASK, "finalise", model=ScriptedModel([]))
    assert {"run_halving", "confirm_seeds"} <= set(tune.tool_names)
    assert not {"run_halving", "confirm_seeds"} & set(finalise.tool_names)
    spec = tune.tool_registry.registry["run_halving"].tool_spec
    assert "- lr: Learning rate" in spec["description"]
