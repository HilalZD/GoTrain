import json
import re

import pytest
from conftest import make_config

from gotrain.tools import init_task_budget, leaderboard, promote, run_trials

TASK = "m-eurosat"  # fake engine: 0.1 min/epoch at img 64, plus 0.2 min setup


@pytest.fixture
def budget(home):
    init_task_budget(TASK, 60)
    return home


def test_run_trials_saves_run_folder_and_returns_table(budget):
    out = run_trials(TASK, [make_config()], 2.0)
    assert "error" not in out
    (row,) = out["trials"]
    assert set(row) == {"run_id", "status", "val_score", "minutes", "reason"}
    assert re.fullmatch(r"m-eurosat-\d{8}-\d{6}-[0-9a-f]{4}", row["run_id"])

    run_dir = budget / "runs" / TASK / row["run_id"]
    assert json.loads((run_dir / "config.json").read_text())["task"] == TASK
    assert json.loads((run_dir / "result.json").read_text())["status"] == "ok"
    assert (run_dir / "git_commit").read_text().strip()
    assert (run_dir / "checkpoints" / "best.ckpt").exists()
    assert out["budget"] == {"used_min": row["minutes"], "remaining_min": 60 - row["minutes"]}


def test_invalid_configs_reported_valid_ones_still_run(budget):
    configs = [make_config(), make_config(lr=5.0), make_config(learning_rate=1e-3), "oops"]
    out = run_trials(TASK, configs, 1.0)
    assert len(out["trials"]) == 1
    errors = {item["index"]: item["error"] for item in out["invalid"]}
    assert "lr: Input should be less than or equal to 0.1" in errors[1]
    assert "Hint: Learning rate" in errors[1]
    assert "Allowed fields" in errors[2]
    assert "must be an object" in errors[3]
    assert "hint" in out


def test_config_for_another_task_is_invalid(budget):
    out = run_trials(TASK, [make_config(task="m-brick-kiln")], 1.0)
    assert out["trials"] == [] and "but task is 'm-eurosat'" in out["invalid"][0]["error"]


def test_budget_must_be_initialised(home):
    out = run_trials(TASK, [make_config()], 1.0)
    assert "no GPU budget" in out["error"] and "init_task_budget" in out["hint"]


def test_budget_is_enforced(home):
    init_task_budget(TASK, 5)
    out = run_trials(TASK, [make_config(seed=i) for i in range(3)], 2.0)
    assert len(out["trials"]) == 2  # each fits 18 epochs = 2.0 min, leaving 1.0 < 2.0
    (skip,) = out["skipped"]
    assert skip["index"] == 2 and "only 1 min of budget remain" in skip["reason"]
    assert out["budget"]["remaining_min"] == pytest.approx(1.0)

    refused = run_trials(TASK, [make_config()], 2.0)
    assert "needs 2 min but only 1 min" in refused["error"]


def test_budget_cannot_be_changed_mid_task(home):
    assert init_task_budget(TASK, 30)["created"] is True
    assert init_task_budget(TASK, 30)["created"] is False
    assert "can't be changed" in init_task_budget(TASK, 999)["error"]


def test_leaderboard_ranks_by_val_score_errors_last(budget):
    configs = [
        make_config(weights="none", reason="none"),
        make_config(bands=["B99"], reason="broken"),
        make_config(weights="ssl4eo_s2", reason="ssl"),
        make_config(weights="imagenet", reason="imagenet"),
    ]
    run_trials(TASK, configs, 2.0)
    board = leaderboard(TASK, top_k=10)
    assert board["n_runs"] == 4 and board["metric_name"] == "accuracy"
    statuses = [r["status"] for r in board["top"]]
    assert statuses == ["ok", "ok", "ok", "error"]
    scores = [r["val_score"] for r in board["top"][:3]]
    assert scores == sorted(scores, reverse=True)
    assert "ssl4eo_s2" in board["top"][0]["config"] and "/none" in board["top"][2]["config"]
    assert [r["rank"] for r in leaderboard(TASK, top_k=2)["top"]] == [1, 2]


def test_leaderboard_bad_args_and_empty(budget):
    assert "top_k" in leaderboard(TASK, top_k=0)["error"]
    assert "No runs yet" in leaderboard(TASK)["hint"]


def test_promote_links_parent_to_child(budget):
    parent = run_trials(TASK, [make_config()], 1.0)["trials"][0]
    out = promote(parent["run_id"], 2.0)
    assert out["parent"]["run_id"] == parent["run_id"]
    assert out["child"]["epochs"] > out["parent"]["epochs"]
    assert out["delta_val_score"] == pytest.approx(
        out["child"]["val_score"] - out["parent"]["val_score"]
    )
    child_row = next(r for r in leaderboard(TASK)["top"] if r["run_id"] == out["child"]["run_id"])
    assert child_row["parent"] == parent["run_id"]


def test_promote_refusals(home):
    init_task_budget(TASK, 3)
    short = run_trials(TASK, [make_config()], 1.0)["trials"][0]["run_id"]
    assert "remain" in promote(short, 2.5)["error"]  # needs 3.5, has 2.1
    assert "no run with run_id" in promote("m-eurosat-nope", 1.0)["error"]

    complete = run_trials(TASK, [make_config(max_epochs=5)], 1.0)["trials"][0]["run_id"]
    assert "already finished all 5 epochs" in promote(complete, 1.0)["error"]
