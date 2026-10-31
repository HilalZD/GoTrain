"""Budget discipline: screening quality, the seed-check reserve, and fake subsampling."""

import pytest
from conftest import make_config

from gotrain.contracts import TrainConfig
from gotrain.engine.fake import FakeEngine
from gotrain.tools import init_task_budget, promote, run_halving, run_trials
from gotrain.tools.search import INCONCLUSIVE_WARNING, screening_summary

TASK = "m-eurosat"  # fake engine: 0.1 min/epoch at img 64 + 0.2 min setup


def ok(score: float, epochs: int = 10, run_id: str = "r") -> dict:
    return {"run_id": run_id, "status": "ok", "val_score": score, "epochs": epochs}


FAILED = {"run_id": "bad", "status": "error", "val_score": None, "epochs": 0}


# ---------------------------------------------------------------- screening


def test_inconclusive_screening_warns():
    out = screening_summary([ok(0.680), ok(0.683), ok(0.6815)])
    assert out == {"screening_spread": 0.003, "warnings": [INCONCLUSIVE_WARNING]}


def test_conclusive_screening_has_no_warning():
    out = screening_summary([ok(0.667), ok(0.683), FAILED])
    assert out == {"screening_spread": 0.016, "warnings": []}
    assert screening_summary([ok(0.7)]) == {"screening_spread": None, "warnings": []}


def test_few_epochs_warning_names_the_short_runs():
    rung1 = [ok(0.667, 1, "a"), ok(0.683, 2, "b"), ok(0.70, 3, "c"), FAILED]
    (warning,) = screening_summary(rung1)["warnings"]  # spread 0.033: only the epoch warning
    assert warning.startswith(
        "rung 1 trials finished only 1 epochs; screening is unreliable. "
        "Use train_subset_frac 0.2–0.3 or a longer short_minutes"
    )
    assert "a (1), b (2)" in warning and "c (" not in warning and "bad" not in warning


def test_both_screening_warnings_can_fire():
    out = screening_summary([ok(0.680, 1), ok(0.682, 1)])
    assert len(out["warnings"]) == 2 and out["warnings"][0] == INCONCLUSIVE_WARNING


def test_halving_warns_on_one_epoch_screen_and_not_with_subsample(home):
    init_task_budget("m-brick-kiln", 100)  # 15k train samples: 0.75 min per full epoch
    full = run_halving("m-brick-kiln", [make_config(seed=s) for s in range(3)], 1.5, 4.0, 1)
    rung1 = [t["run_id"] for t in full["trials"] if t["rung"] == 1]
    epoch_warning = next(w for w in full["warnings"] if "finished only 1 epochs" in w)
    assert all(run_id in epoch_warning for run_id in rung1)

    subsampled = [make_config(seed=s, train_subset_frac=0.25) for s in range(3)]
    out = run_halving("m-brick-kiln", subsampled, 1.5, 4.0, 1)  # 6 epochs each
    assert not any("epochs" in w for w in out["warnings"])


def test_halving_reports_spread_and_warning(home):
    init_task_budget(TASK, 100)
    # identical configs except the seed: rung-1 differences are pure seed noise
    configs = [make_config(seed=s, reason=f"seed {s}") for s in range(3)]
    out = run_halving(TASK, configs, 1.0, 3.0, 1)
    rung1 = [t["val_score"] for t in out["trials"] if t["rung"] == 1]
    assert out["screening_spread"] == pytest.approx(max(rung1) - min(rung1), abs=1e-4)
    assert (INCONCLUSIVE_WARNING in out["warnings"]) == (out["screening_spread"] < 0.005)
    assert {"seed_check_cost_min", "can_afford_seed_check"} <= out.keys()


# ---------------------------------------------------------------- seed-check reserve


def test_can_afford_seed_check_is_computed_from_best_run(home):
    init_task_budget(TASK, 10)
    out = run_trials(TASK, [make_config()], 2.0)  # uses 2.0, best budget 2 -> check costs 4
    assert out["seed_check_cost_min"] == 4.0
    assert out["can_afford_seed_check"] is True  # 8 left

    out = run_trials(TASK, [make_config(seed=1), make_config(seed=2)], 2.0)  # 4 left
    assert out["seed_check_cost_min"] == 4.0 and out["can_afford_seed_check"] is True
    out = run_trials(TASK, [make_config(seed=3)], 2.0)  # 2 left < 4
    assert out["can_afford_seed_check"] is False


def test_seed_check_fields_are_none_without_a_successful_run(home):
    init_task_budget(TASK, 10)
    out = run_trials(TASK, [make_config(bands=["B99"])], 1.0)
    assert out["seed_check_cost_min"] is None and out["can_afford_seed_check"] is None


def test_promote_refuses_without_force_and_runs_with_force(home):
    init_task_budget(TASK, 8)
    parent = run_trials(TASK, [make_config()], 1.0)["trials"][0]["run_id"]  # 7 min left
    # promote to 3 min: leaves 4, but seed-checking a 3-min winner needs 6
    refused = promote(parent, 2.0)
    assert "would leave 4 min" in refused["error"] and "needs 6 min" in refused["error"]
    assert "force=true" in refused["hint"]
    assert refused["budget"]["remaining_min"] == pytest.approx(7.0)  # nothing was trained

    forced = promote(parent, 2.0, force=True)
    assert forced["forced"] is True and forced["child"]["status"] == "ok"
    assert forced["can_afford_seed_check"] is False


def test_promote_allowed_when_reserve_fits(home):
    init_task_budget(TASK, 20)
    parent = run_trials(TASK, [make_config()], 1.0)["trials"][0]["run_id"]
    out = promote(parent, 2.0)  # leaves 16 >= 6
    assert out["forced"] is False and out["can_afford_seed_check"] is True


# ---------------------------------------------------------------- fake subsampling


def train(tmp_path, name: str, frac: float, max_epochs: int, budget: float):
    cfg = TrainConfig(
        task="m-brick-kiln",  # 15k train samples: 0.75 min per full epoch
        **make_config(train_subset_frac=frac, max_epochs=max_epochs, reason="r"),
    )
    return FakeEngine(sleep_s=0).train(cfg, budget, tmp_path / name)


def test_subsample_fits_proportionally_more_epochs(tmp_path):
    full = train(tmp_path, "full", 1.0, 100, 1.5)
    quarter = train(tmp_path, "quarter", 0.25, 100, 1.5)
    assert len(full.history) == 1
    assert len(quarter.history) == 6  # 1.3 min / (0.75 * 0.25) min per epoch
    assert quarter.val_score > full.val_score  # several epochs beat one: screening works


def test_subsample_final_score_is_slightly_capped(tmp_path):
    full = train(tmp_path, "full", 1.0, 30, 60.0)  # both finish all 30 epochs
    quarter = train(tmp_path, "quarter", 0.25, 30, 60.0)
    assert len(full.history) == len(quarter.history) == 30
    assert 0.01 < full.val_score - quarter.val_score < 0.04
    assert train(tmp_path, "again", 0.25, 30, 60.0).val_score == quarter.val_score
