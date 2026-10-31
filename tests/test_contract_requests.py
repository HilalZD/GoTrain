"""Serena's contract requests #2 (budget-stopped runs), #9 (extra metric), #10 (task name),
and her measured facts: segmentation timing and the human baselines."""

import json

import pytest
from conftest import make_config

import gotrain.tools.trials as trials_mod
from gotrain.config import REPO_ROOT
from gotrain.contracts import TrainResult
from gotrain.engine.fake import CARDS, FAKE_TIMING, FakeEngine
from gotrain.engine.throughput import REAL_T4, card_minutes_per_epoch, estimate_minutes_per_epoch
from gotrain.knowledge.targets import load_targets
from gotrain.storage import get_storage
from gotrain.tools import init_task_budget, leaderboard, promote, register_model, run_trials

TASK = "m-eurosat"  # fake engine: 0.1 min/epoch at img 64, plus 0.2 min setup


@pytest.fixture
def budget(home):
    init_task_budget(TASK, 60)
    init_task_budget("m-chesapeake", 60)
    return home


# ------------------------------------------------- #2: budget-stopped runs are "ok"


def test_budget_stopped_run_is_ok_and_flagged(budget):
    (row,) = run_trials(TASK, [make_config()], 1.0)["trials"]  # ~8 of 30 epochs fit
    assert row["status"] == "ok" and row["stopped_by_budget"] is True
    metrics = json.loads((budget / "runs" / TASK / row["run_id"] / "metrics.json").read_text())
    assert metrics["stopped_by_budget"] is True and metrics["stop_reason"] == "budget"
    board = leaderboard(TASK)
    assert board["top"][0]["stopped_by_budget"] is True
    assert "may still be rising" in board["stopped_by_budget_note"]


def test_run_that_reaches_max_epochs_is_not_flagged(budget):
    (row,) = run_trials(TASK, [make_config(max_epochs=5)], 1.0)["trials"]
    assert row["status"] == "ok" and row["stopped_by_budget"] is False
    assert "stopped_by_budget_note" not in leaderboard(TASK)


def test_zero_epoch_run_stays_a_timeout(budget):
    (row,) = run_trials(TASK, [make_config()], 0.25)["trials"]  # setup alone takes 0.2 min
    assert row["status"] == "timeout" and row["stopped_by_budget"] is False


def result(status="ok", epochs=3):
    from gotrain.contracts import EpochRecord

    history = [
        EpochRecord(epoch=e, train_loss=1.0, val_score=0.5, elapsed_s=e)
        for e in range(1, epochs + 1)
    ]
    ok = status == "ok"
    return TrainResult(
        run_id="r",
        task=TASK,
        status=status,
        metric_name="accuracy",
        history=history,
        val_score=0.5 if ok else None,
        minutes_used=1.0,
        checkpoint_path="best.ckpt" if ok else None,
        error=None if ok else "budget",
    )


@pytest.mark.parametrize(
    "metrics, expected",
    [
        ({"stopped_by_budget": True, "stop_reason": "max_epochs"}, True),  # the flag wins
        ({"stopped_by_budget": False}, False),
        ({"stop_reason": "budget"}, True),  # the real engine's older metrics.json
        ({"stop_reason": "max_epochs"}, False),
        ({}, True),  # no metrics.json: fewer epochs than max_epochs
    ],
)
def test_stopped_by_budget_reads_metrics_json(metrics, expected):
    config = trials_mod.TrainConfig.model_validate(make_config(task=TASK, max_epochs=5))
    assert trials_mod.stopped_by_budget(result(), config, metrics) is expected
    assert trials_mod.stopped_by_budget(result("timeout", 0), config, metrics) is False


def test_promote_accepts_a_budget_stopped_run_even_at_max_epochs(budget, monkeypatch):
    """The real engine validates a cut-short last epoch, so a budget-stopped run can show
    max_epochs epochs; it is still promotable."""

    class BudgetStopped(FakeEngine):
        def train(self, config, budget_minutes, run_dir):
            out = super().train(config, budget_minutes, run_dir)
            (run_dir / "metrics.json").write_text(json.dumps({"stopped_by_budget": True}))
            return out

    monkeypatch.setattr(trials_mod, "get_engine", lambda: BudgetStopped(sleep_s=0))
    (row,) = run_trials(TASK, [make_config(max_epochs=5)], 1.0)["trials"]
    assert row["stopped_by_budget"] is True
    out = promote(row["run_id"], 1.0)
    assert "error" not in out and out["child"]["status"] == "ok"


# ------------------------------------------------- #9: extra metric in model cards only


def test_miou_excl_nodata_is_shown_but_not_ranked(budget):
    (row,) = run_trials("m-chesapeake", [chesapeake_config()], 4.0)["trials"]
    entry = trials_mod.find_entry(row["run_id"])
    assert entry["miou_excl_nodata"] > entry["val_score"]
    assert leaderboard("m-chesapeake")["top"][0]["val_score"] == entry["val_score"]
    card = get_storage().read_text(register_model(row["run_id"])["model_card"])
    assert f"no-data class {entry['miou_excl_nodata']:.4f}" in card
    assert "not used for selection" in card


def chesapeake_config(**overrides):
    return make_config(weights="imagenet", img_size=256, batch_size=16, **overrides)


def test_no_extra_metric_for_tasks_without_a_nodata_class(budget):
    (row,) = run_trials(TASK, [make_config()], 1.0)["trials"]
    assert "miou_excl_nodata" not in trials_mod.find_entry(row["run_id"])


# ------------------------------------------------- #10: the task is "m-chesapeake"


def test_chesapeake_uses_the_real_geobench_name():
    assert "m-chesapeake" in CARDS and CARDS["m-chesapeake"].task == "m-chesapeake"
    assert "m-chesapeake" in load_targets().tasks
    assert CARDS["m-chesapeake"].class_names[0] == "no data"  # Serena's real card is used


def test_old_chesapeake_name_is_gone_from_our_code():
    old = "m-chesapeake-" + "landcover"
    files = [*(REPO_ROOT / "gotrain").rglob("*.py"), *(REPO_ROOT / "gotrain").rglob("*.md")]
    files += [*(REPO_ROOT / "gotrain").rglob("*.json"), REPO_ROOT / "README.md"]
    offenders = [p.name for p in files if old in p.read_text(encoding="utf-8")]
    assert not offenders


# ------------------------------------------------- measured facts: timing and baselines


def test_segmentation_timing_matches_the_pv4ger_measurement():
    """m-pv4ger-seg, U-Net resnet34, 320 px, batch 16: ~34 s per epoch (3,000 train + 403 val)."""
    card = CARDS["m-pv4ger-seg"]
    assert card.split_sizes == {"train": 3000, "val": 403}
    assert card_minutes_per_epoch(card, timing=REAL_T4) * 60 == pytest.approx(34.0, rel=0.02)


def test_segmentation_rate_applies_to_segmentation_only():
    cls = estimate_minutes_per_epoch(3000, 320, 320, 1.0, 403, REAL_T4)
    seg = estimate_minutes_per_epoch(3000, 320, 320, 1.0, 403, REAL_T4, segmentation=True)
    assert seg > cls * 1.5
    fake = [
        estimate_minutes_per_epoch(3000, 320, 320, timing=FAKE_TIMING, segmentation=s)
        for s in (False, True)
    ]
    assert fake[0] == fake[1]  # the fake engine keeps its own single constant


def test_classification_calibration_is_unchanged():
    for side, measured_s in [(64, 1.9), (224, 10.0)]:
        minutes = estimate_minutes_per_epoch(2000, side, side, 1.0, 1000, REAL_T4)
        assert minutes * 60 == pytest.approx(measured_s, rel=0.02)


def test_human_baselines_and_citations():
    tasks = load_targets().tasks
    assert (tasks["m-eurosat"].human_baseline.value, tasks["m-eurosat"].human_baseline.split) == (
        0.986,
        "val",
    )
    pv = tasks["m-pv4ger-seg"].human_baseline
    assert (pv.value, pv.split, tasks["m-pv4ger-seg"].metric) == (0.959, "val", "miou")
    sources = [r.source for t in tasks.values() for r in t.references]
    assert not [s for s in sources if "to be added" in s]
    assert all(
        "arXiv 2412.02732" in t.references[0].source for k, t in tasks.items() if k != "m-eurosat"
    )
    assert any("arXiv 2506.06281" in r.source for r in tasks["m-eurosat"].references)
