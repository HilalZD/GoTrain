"""Calibrated T4 timing, and the published reference scores (targets) the agent can see."""

import json
import re

import pytest
from pydantic import ValidationError

from gotrain.agent import load_prompt
from gotrain.engine.fake import CARDS, FAKE_TIMING
from gotrain.engine.throughput import (
    REAL_T4,
    active_timing,
    card_minutes_per_epoch,
    estimate_minutes_per_epoch,
    screening_advice,
    subset_for_epochs,
)
from gotrain.knowledge.targets import TARGETS_PATH, Targets, load_targets, targets_for
from gotrain.tools import inspect_dataset

# ---------------------------------------------------------------- timing


@pytest.mark.parametrize("side, measured_s", [(64, 1.9), (224, 10.0)])
def test_real_t4_matches_the_eurosat_measurements(side, measured_s):
    """m-eurosat, resnet50, batch 64, fp16: 2,000 train + 1,000 val images per epoch."""
    minutes = estimate_minutes_per_epoch(2000, side, side, 1.0, 1000, REAL_T4)
    assert minutes * 60 == pytest.approx(measured_s, rel=0.02)


def test_val_images_count_in_the_real_model_only():
    with_val = estimate_minutes_per_epoch(2000, 64, 64, val_samples=1000, timing=REAL_T4)
    without = estimate_minutes_per_epoch(2000, 64, 64, timing=REAL_T4)
    assert with_val == pytest.approx(without * 1.5)
    fake = estimate_minutes_per_epoch(2000, 64, 64, val_samples=1000, timing=FAKE_TIMING)
    assert fake == estimate_minutes_per_epoch(2000, 64, 64, timing=FAKE_TIMING)


def test_hints_follow_the_active_engine(monkeypatch, tmp_path):
    card = CARDS["m-pv4ger-seg"]
    assert active_timing() is FAKE_TIMING  # the test default
    assert screening_advice(card) is not None  # fake: ~3.75 min per epoch
    monkeypatch.setenv("GOTRAIN_ENGINE", "real")
    monkeypatch.setenv("GOTRAIN_HOME", str(tmp_path / "gotrain_home"))
    assert active_timing() is REAL_T4
    assert card_minutes_per_epoch(card) * 60 == pytest.approx(21.8, abs=0.5)
    assert screening_advice(card) is None  # 3 real epochs fit in a 2-min screen


def test_subset_accounts_for_the_fixed_validation_pass():
    card = CARDS["m-pv4ger-seg"]
    # A screen so short that even the validation passes don't fit: no subsample helps.
    assert subset_for_epochs(card, 0.05, timing=REAL_T4) is None
    frac = subset_for_epochs(card, 1.0, timing=REAL_T4)
    per_epoch = card_minutes_per_epoch(card, frac=frac, timing=REAL_T4)
    assert 3 * per_epoch <= 1.0


# ---------------------------------------------------------------- targets


def test_targets_load_and_validate():
    targets = load_targets()
    assert set(targets.tasks) == set(CARDS)
    eurosat = targets.tasks["m-eurosat"]
    assert eurosat.metric == "accuracy"
    assert {r.value for r in eurosat.references} == {0.9816, 0.986}
    assert all(r.split == "test" for t in targets.tasks.values() for r in t.references)
    assert eurosat.human_baseline.value == 0.986 and eurosat.human_baseline.split == "val"
    assert targets.tasks["m-pv4ger-seg"].references[0].value == 0.9523
    assert targets.tasks["m-chesapeake-landcover"].metric == "miou"


def test_target_metrics_match_the_tasks():
    metric = {"classification": "accuracy", "segmentation": "miou", "multilabel": "f1"}
    for task, t in load_targets().tasks.items():
        assert t.metric == metric[CARDS[task].task_type], task


def test_schema_rejects_recipes():
    data = json.loads(TARGETS_PATH.read_text(encoding="utf-8"))
    data["tasks"]["m-eurosat"]["references"][0]["lr"] = 0.001
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Targets.model_validate(data)


HYPERPARAMETER_KEY = re.compile(
    r"^(lr|learning_?rate|backbone|weights|batch_?size|epochs?|max_epochs|optimi[sz]er|"
    r"weight_?decay|momentum|loss|augment(ation)?s?|img_?size|image_?size|resolution|"
    r"scheduler|warmup|dropout|seed|train_subset_frac|bands|config|recipe|hyper\w*)$",
    re.IGNORECASE,
)


def all_keys(node) -> list[str]:
    if isinstance(node, dict):
        return [k for k, v in node.items() for k in [k, *all_keys(v)]]
    if isinstance(node, list):
        return [k for item in node for k in all_keys(item)]
    return []


def test_targets_json_has_no_hyperparameter_keys():
    keys = all_keys(json.loads(TARGETS_PATH.read_text(encoding="utf-8")))
    offenders = sorted({k for k in keys if HYPERPARAMETER_KEY.match(k)})
    assert not offenders, f"targets.json must hold scores only, found: {offenders}"


def test_hyperparameter_guard_catches_nested_keys():
    assert {"batch_size", "LR"} <= {
        k
        for k in all_keys({"a": [{"b": {"batch_size": 64}}], "LR": 1})
        if HYPERPARAMETER_KEY.match(k)
    }


def test_inspect_dataset_includes_targets():
    targets = inspect_dataset("m-eurosat")["targets"]
    assert targets["metric"] == "accuracy" and targets["best_reference"] == 0.986
    assert targets["human_baseline"]["value"] == 0.986
    assert "TEST scores" in targets["note"] and "VALIDATION" in targets["note"]
    assert targets_for("m-unknown") is None


def test_tune_prompt_uses_targets():
    prompt = load_prompt("tune")
    assert "`targets`" in prompt and "TEST numbers from papers" in prompt
    assert "VALIDATION" in prompt and "Far below the best reference" in prompt
