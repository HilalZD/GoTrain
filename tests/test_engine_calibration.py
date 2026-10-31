"""Confidence flags (gotrain/engine/calibration.py): temperature, threshold, predict() extras.

Pure tests use fake data; the end-to-end tests train a tiny run (CPU) and skip without data.
"""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from test_engine_contract_v2 import make_ckpt, raw_eurosat  # noqa: E402

from gotrain.contracts import TrainConfig  # noqa: E402
from gotrain.engine import calibration as cal  # noqa: E402
from gotrain.engine import data  # noqa: E402
from gotrain.engine import train as engine  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def repo_stats(monkeypatch):
    monkeypatch.setenv("GOTRAIN_STATS_DIR", str(REPO / "stats"))


def have(task: str) -> bool:
    try:
        import geobench  # noqa: F401

        data.find_task_dir(task)
        return True
    except (ImportError, FileNotFoundError):
        return False


def overconfident(n: int = 20000, k: int = 5, t_true: float = 2.0, seed: int = 0):
    """Logits z whose labels follow softmax(z / t_true): the raw model is over-confident."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 3, (n, k))
    p = cal.softmax_t(z, t_true)
    y = np.array([rng.choice(k, p=row) for row in p])
    return z, y


# ------------------------------------------------------------------ pure (fake data)
def test_fit_temperature_recovers_the_true_temperature():
    z, y = overconfident()
    t = cal.fit_temperature(z, y)
    assert t == pytest.approx(2.0, abs=0.15)
    assert cal.nll(z, y, t) < cal.nll(z, y, 1.0)


def test_ece_drops_after_scaling():
    z, y = overconfident()
    t = cal.fit_temperature(z, y)
    correct = z.argmax(1) == y
    before = cal.ece(cal.softmax_t(z, 1.0).max(1), correct)
    after = cal.ece(cal.softmax_t(z, t).max(1), correct)
    assert after < before / 2


def test_binary_unet_logits_match_sigmoid_temperature():
    """softmax(log[1 - p, p] / T)[:, 1] == sigmoid(z / T) for a one-channel U-Net."""
    z = np.linspace(-6, 6, 25)
    p = 1 / (1 + np.exp(-z))
    two = cal.softmax_t(cal.log_probs(np.stack([1 - p, p], axis=1)), 2.5)[:, 1]
    assert np.allclose(two, 1 / (1 + np.exp(-z / 2.5)), atol=1e-5)


def test_choose_threshold_keeps_error_at_most_one_percent():
    rng = np.random.default_rng(0)
    conf = np.sort(rng.uniform(0.5, 1, 1000))[::-1]
    correct = np.ones(1000, bool)
    correct[600:] = rng.uniform(size=400) < 0.7  # the least confident 40% are noisy
    out = cal.choose_threshold(conf, correct)
    keep = conf >= out["confidence"]
    assert 1 - correct[keep].mean() <= 0.01
    assert out["coverage"] >= 0.6
    assert out["flagged_pct"] == pytest.approx(100 * (1 - keep.mean()), abs=0.01)


def test_choose_threshold_when_nothing_is_good_enough():
    out = cal.choose_threshold(np.array([0.9, 0.8]), np.array([False, True]))
    assert out["coverage"] == 0.0 and out["confident_accuracy"] is None


def test_tile_needs_review():
    pred = np.array([0, 1, 1, 1, 1])
    conf = np.array([0.1, 0.99, 0.99, 0.5, 0.99])  # 1 of 4 target pixels uncertain = 25%
    assert cal.tile_needs_review(conf, pred, threshold=0.9, background=0) is True
    assert cal.tile_needs_review(conf, pred, threshold=0.4, background=0) is False
    assert cal.tile_needs_review(conf, np.zeros(5, int), 0.9, background=0) is False


def test_coverage_curve_ends_at_overall_accuracy():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([True, True, False, True])
    curve = cal.coverage_curve(conf, correct, points=4)
    assert curve["coverage"][-1] == 1.0 and curve["accuracy"][-1] == 0.75
    assert curve["accuracy"][0] == 1.0


# ------------------------------------------------------------------ fake calibrated run
@pytest.fixture
def fake_run(tmp_path):
    """A run dir with an untrained eurosat checkpoint and fake over-confident val predictions."""
    run = tmp_path / "m-eurosat-cal-0001"
    make_ckpt(run / "checkpoints" / "best.ckpt", "m-eurosat")
    z, y = overconfident(n=1000, k=10)
    np.savez_compressed(
        run / "val_predictions.npz", probs=cal.softmax_t(z, 1.0), y_true=y, y_pred=z.argmax(1)
    )
    return run


def test_calibrate_writes_json_and_reliability_png(fake_run):
    out = cal.calibrate("m-eurosat", fake_run)
    saved = json.loads((fake_run / "calibration.json").read_text())
    assert saved == out and (fake_run / "reliability.png").exists()
    assert out["fitted_on"] == "val" and "VALIDATION" in out["note"]
    assert out["temperature"] == pytest.approx(2.0, abs=0.4)
    assert out["ece_after"] < out["ece_before"] and out["nll_after"] < out["nll_before"]
    assert {"confidence", "coverage", "confident_accuracy", "flagged_pct"} <= set(out["threshold"])


def test_predict_adds_confidence_keys_only_when_calibrated(fake_run):
    ckpt = fake_run / "checkpoints" / "best.ckpt"
    images = raw_eurosat(3)
    before = engine.predict("m-eurosat", ckpt, images)
    assert "confidence" not in before and "needs_review" not in before
    cal.calibrate("m-eurosat", fake_run)
    after = engine.predict("m-eurosat", ckpt, images)
    assert after["confidence"].shape == (3,) and after["needs_review"].dtype == bool
    assert np.array_equal(after["labels"], before["labels"])  # existing outputs unchanged
    assert np.allclose(after["probabilities"], before["probabilities"])
    assert "confidence" not in engine.predict("m-eurosat", ckpt, images, tta=True)
    assert "confidence" not in engine.predict("m-eurosat", [ckpt, ckpt], images)


def test_calibrate_refuses_other_task(fake_run):
    with pytest.raises(ValueError, match="not 'm-pv4ger-seg'"):
        cal.calibrate("m-pv4ger-seg", fake_run)


# ------------------------------------------------------------------ end to end (CPU, real data)
@pytest.mark.skipif(not have("m-eurosat"), reason="needs geobench + m-eurosat on disk")
def test_end_to_end_eurosat_tiny_run(tmp_path):
    run = tmp_path / "m-eurosat-e2e-0001"
    config = TrainConfig(
        task="m-eurosat",
        backbone="resnet18",
        weights="none",
        img_size=64,
        batch_size=16,
        lr=1e-3,
        max_epochs=2,
        loss="ce",
        train_subset_frac=0.1,
        reason="calibration e2e",
    )
    assert engine.train(config, 600.0, run).status == "ok"
    out = cal.calibrate("m-eurosat", run)
    assert out["n_val"] == 1000 and 0.05 < out["temperature"] < 20
    assert out["nll_after"] <= out["nll_before"] + 1e-9
    images = np.stack(
        [data.GeoBenchDataset("m-eurosat", "val", norm="none")[i][0].numpy() for i in range(4)]
    )
    pred = engine.predict("m-eurosat", run / "checkpoints" / "best.ckpt", images)
    assert pred["confidence"].shape == (4,)


@pytest.mark.skipif(not have("m-pv4ger-seg"), reason="needs geobench + m-pv4ger-seg on disk")
def test_end_to_end_segmentation_tile_flags(tmp_path):
    run = tmp_path / "m-pv4ger-seg-e2e-0001"
    config = TrainConfig(
        task="m-pv4ger-seg",
        backbone="resnet18",
        weights="none",
        img_size=64,
        batch_size=16,
        lr=1e-3,
        max_epochs=1,
        loss="dice_bce",
        train_subset_frac=0.05,
        reason="calibration seg e2e",
    )
    assert engine.train(config, 600.0, run).status == "ok"
    out = cal.calibrate("m-pv4ger-seg", run)
    assert out["unit"].startswith("pixels") and out["n_val"] == 403
    assert out["tiles"]["background_class"] == "no solar pv"
    assert 0 <= out["tiles"]["flagged_pct"] <= 100
    images = np.random.default_rng(0).uniform(0, 255, (2, 3, 64, 64)).astype(np.float32)
    pred = engine.predict("m-pv4ger-seg", run / "checkpoints" / "best.ckpt", images)
    assert pred["confidence"].shape == (2, 64, 64) and pred["needs_review"].shape == (2, 64, 64)
