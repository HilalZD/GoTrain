"""95% bootstrap CI on the test score."""

import json

import numpy as np
from test_tools_models import TASK, budget, one_run  # noqa: F401, F811

from gotrain.bootstrap import bootstrap_ci
from gotrain.contracts import TestRecord
from gotrain.tools import final_test, register_model


def test_classification_ci_brackets_accuracy_and_is_deterministic():
    correct = [1] * 90 + [0] * 10
    ci = bootstrap_ci("classification", {"correct": correct})
    assert ci == bootstrap_ci("classification", {"correct": correct})
    assert ci["ci_low"] < 0.9 < ci["ci_high"] and ci["ci_level"] == 0.95
    assert 0.8 < ci["ci_low"] and ci["ci_high"] < 0.97


def test_segmentation_ci_recomputes_miou_from_images():
    rng = np.random.default_rng(1)
    union = rng.integers(50, 100, size=(200, 2))
    inter = (union * 0.8).astype(int)
    ci = bootstrap_ci("segmentation", {"intersection": inter.tolist(), "union": union.tolist()})
    miou = np.mean(inter.sum(0) / union.sum(0))
    assert ci["ci_low"] <= miou <= ci["ci_high"] and ci["ci_high"] - ci["ci_low"] < 0.05


def test_unusable_per_sample_gives_no_ci():
    assert bootstrap_ci("classification", {}) is None
    assert bootstrap_ci("classification", {"correct": []}) is None


def test_final_test_stores_ci_in_record_lock_and_model_card(budget):  # noqa: F811
    home = budget
    register_model(one_run())
    out = final_test(TASK, "v1")
    assert out["ci_low"] <= out["test_score"] <= out["ci_high"]
    lock = json.loads((home / "registry" / TASK / "TEST_LOCK.json").read_text())
    assert lock["ci_low"] == out["ci_low"]
    card = (home / "registry" / TASK / "v1" / "model_card.md").read_text(encoding="utf-8")
    assert "## Test result" in card and "95% CI: [" in card


def test_old_records_without_ci_still_load():
    old = {
        "task": TASK,
        "version": "v1",
        "test_score": 0.9,
        "metric_name": "accuracy",
        "git_commit": "abc",
        "timestamp": "2026-10-01T00:00:00Z",
    }
    record = TestRecord.model_validate(old)
    assert record.ci_low is None and record.ci_level is None
