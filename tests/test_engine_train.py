"""Tests for gotrain/engine/train.py and models.py (Serena's engine).

Pure tests (config checks, losses, metrics, transforms, model shapes) run anywhere with torch.
End-to-end trials need the GEO-Bench files and are skipped without them.
"""

import json
import time
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("segmentation_models_pytorch")

from gotrain.contracts import TrainConfig  # noqa: E402
from gotrain.engine import data  # noqa: E402
from gotrain.engine import train as engine  # noqa: E402
from gotrain.engine.models import ModelSpec, build_model, ssl4eo_inputs  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def repo_stats(monkeypatch):
    monkeypatch.setenv("GOTRAIN_STATS_DIR", str(REPO / "stats"))


def cfg(task="m-eurosat", **kw) -> TrainConfig:
    base = {
        "task": task,
        "backbone": "resnet18",
        "weights": "none",
        "img_size": 64,
        "batch_size": 16,
        "lr": 1e-3,
        "max_epochs": 2,
        "loss": "ce",
        "reason": "test",
    }
    return TrainConfig(**(base | kw))


def have(task: str) -> bool:
    try:
        data.find_task_dir(task)
        return True
    except FileNotFoundError:
        return False


# ------------------------------------------------------------------ config checks
@pytest.mark.parametrize(
    ("task", "overrides", "expected"),
    [
        ("m-eurosat", {"loss": "dice_ce"}, "does not fit a classification task"),
        ("m-eurosat", {"bands": ["B04", "B99"]}, "not in m-eurosat"),
        ("m-eurosat", {"weights": "ssl4eo_s2", "backbone": "convnext_tiny"}, "not available"),
        ("m-eurosat", {"backbone": "not_a_model"}, "not available"),
        ("m-pv4ger-seg", {"loss": "dice_bce", "img_size": 100}, "multiple of 32"),
        ("m-pv4ger-seg", {"loss": "ce", "backbone": "nope"}, "not available"),
        ("m-pv4ger-seg", {"loss": "dice_bce", "weights": "ssl4eo_s2"}, "only works on Sentinel-2"),
        ("m-eurosat", {"weights": "ssl4eo_s2", "bands": ["B10"], "backbone": "resnet18"}, None),
    ],
)
def test_config_problems_have_actionable_messages(task, overrides, expected):
    card = data.describe_dataset(task)
    problem = engine.config_problem(cfg(task, **overrides), card)
    if expected is None:
        assert problem is None
    else:
        assert problem and expected in problem


@pytest.mark.parametrize(
    ("task", "overrides"),
    [
        ("m-eurosat", {"weights": "ssl4eo_s2", "backbone": "resnet50"}),
        ("m-eurosat", {"loss": "focal", "bands": ["B04", "B03", "B02"]}),
        ("m-pv4ger-seg", {"loss": "dice_bce", "backbone": "resnet34", "weights": "imagenet"}),
        ("m-pv4ger-seg", {"loss": "dice_ce", "img_size": 320}),
    ],
)
def test_valid_configs_pass(task, overrides):
    assert engine.config_problem(cfg(task, **overrides), data.describe_dataset(task)) is None


def test_binary_seg_loss_rejected_for_multiclass():
    card = data.describe_dataset("m-pv4ger-seg").model_copy(
        update={"n_classes": 3, "class_names": ["a", "b", "c"]}
    )
    assert "binary masks" in engine.config_problem(cfg("m-pv4ger-seg", loss="bce"), card)


# ------------------------------------------------------------------ schedule, losses, metrics
def test_lr_factor_warms_up_then_decays_to_zero():
    assert engine.lr_factor(0.0) == pytest.approx(0.01)
    assert engine.lr_factor(engine.WARMUP_FRACTION) == pytest.approx(1.0)
    assert engine.lr_factor(0.5) < 1.0
    assert engine.lr_factor(1.0) == pytest.approx(0.0, abs=1e-9)


def test_reserve_is_capped_at_half_the_training_window():
    # SageMaker bug: warm-up made one sample look like 0.25 s -> 0.5 * 0.25 * 1000 val samples
    # = 125 s reserve, more than a 3-minute trial's whole window after setup.
    assert engine.reserve_seconds(None, 0.25, 1000, window_s=100.0) == pytest.approx(50.0)
    assert engine.reserve_seconds(None, None, 1000, window_s=100.0) == engine.SAVE_MARGIN_S
    assert engine.reserve_seconds(4.0, 99.0, 1000, window_s=100.0) == pytest.approx(
        1.3 * 4.0 + engine.SAVE_MARGIN_S
    )
    assert engine.reserve_seconds(None, 0.001, 1000, window_s=-5.0) == 0.0


@pytest.mark.parametrize(
    ("task_type", "loss", "shape_logits", "target"),
    [
        ("classification", "ce", (4, 10), lambda: torch.randint(10, (4,))),
        ("classification", "focal", (4, 10), lambda: torch.randint(10, (4,))),
        ("classification", "bce", (4, 10), lambda: torch.randint(10, (4,))),
        ("multilabel", "bce", (4, 5), lambda: torch.randint(2, (4, 5)).float()),
        ("multilabel", "focal", (4, 5), lambda: torch.randint(2, (4, 5)).float()),
        ("segmentation", "bce", (2, 1, 8, 8), lambda: torch.randint(2, (2, 8, 8))),
        ("segmentation", "dice_bce", (2, 1, 8, 8), lambda: torch.randint(2, (2, 8, 8))),
        ("segmentation", "ce", (2, 3, 8, 8), lambda: torch.randint(3, (2, 8, 8))),
        ("segmentation", "dice_ce", (2, 3, 8, 8), lambda: torch.randint(3, (2, 8, 8))),
        ("segmentation", "focal", (2, 3, 8, 8), lambda: torch.randint(3, (2, 8, 8))),
    ],
)
def test_every_contract_loss_is_finite_and_differentiable(task_type, loss, shape_logits, target):
    logits = torch.randn(*shape_logits, requires_grad=True)
    y = target()
    if task_type == "segmentation":
        y[0, 0, 0] = data.IGNORE_INDEX  # ignored pixels must not break the loss
    n_classes = shape_logits[1] if shape_logits[1] > 1 else 2
    value = engine.compute_loss(loss, task_type, logits, y, n_classes)
    value.backward()
    assert torch.isfinite(value) and logits.grad is not None


def test_miou_and_micro_f1():
    cm = np.array([[8, 2], [2, 8]])
    assert engine.iou_per_class(cm) == pytest.approx([8 / 12, 8 / 12])
    absent = np.array([[5, 0, 0], [0, 5, 0], [0, 0, 0]])
    assert engine.miou_from_confusion(absent) == pytest.approx(1.0)  # absent class ignored
    y = np.array([[1, 0], [1, 1]])
    p = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert engine.micro_f1(y, p) == pytest.approx(2 * 2 / (2 * 2 + 0 + 1))


def test_augment_keeps_masks_aligned():
    torch.manual_seed(0)
    mask = torch.randint(2, (6, 16, 16))
    x = mask[:, None].float().repeat(1, 3, 1, 1)
    xa, ya = engine.augment(x, mask, ["flip", "rot90"], segmentation=True)
    assert torch.equal(xa[:, 0].long(), ya)


def test_tta_views_are_invertible():
    x = torch.randn(2, 3, 8, 8)
    views = engine._tta_views(x)
    assert len(views) == 8
    for fwd, back in views:
        assert torch.equal(back(fwd(x)), x)


# ------------------------------------------------------------------ model menu
def test_classifier_adapts_to_any_band_count():
    spec = ModelSpec(
        "classification",
        "resnet18",
        "none",
        [f"b{i}" for i in range(13)],
        10,
        [0.0] * 13,
        [1.0] * 13,
    )
    assert build_model(spec)(torch.randn(2, 13, 64, 64)).shape == (2, 10)


def test_unet_shapes():
    spec = ModelSpec("segmentation", "resnet18", "none", ["r", "g", "b", "n"], 1, [0] * 4, [1] * 4)
    assert build_model(spec)(torch.randn(1, 4, 64, 64)).shape == (1, 1, 64, 64)


def test_unet_convnext_goes_through_timm_universal_encoder():
    spec = ModelSpec("segmentation", "convnext_tiny", "none", ["r", "g", "b"], 1, [0] * 3, [1] * 3)
    assert build_model(spec)(torch.randn(1, 3, 64, 64)).shape == (1, 1, 64, 64)


def test_combos_ssl4eo_list_matches_torchgeo():
    """combos.py (Hilal) must only offer ssl4eo_s2 for backbones TorchGeo really provides."""
    import torchgeo.models as tgm

    from gotrain.engine.combos import VALID_COMBOS

    for backbone in VALID_COMBOS["ssl4eo_s2"]:
        enum = getattr(tgm, f"ResNet{backbone.removeprefix('resnet')}_Weights")
        meta = enum.SENTINEL2_ALL_MOCO.meta
        assert meta["in_chans"] == 13 and meta["model"] == backbone
        order = [f"B{b[1:].upper().zfill(2)}" for b in meta["bands"]]
        assert order == data.SSL4EO_S2_BANDS


def test_ssl4eo_inputs_zero_fill_brick_kiln_style():
    kiln = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B07",
        "B8A",
        "B08",
        "B11",
        "B12",
        "TCI_R",
        "TCI_G",
        "TCI_B",
    ]
    inputs, zero, dropped = ssl4eo_inputs(kiln)
    assert inputs == data.SSL4EO_S2_BANDS
    assert zero == ["B06", "B09", "B10"]
    assert dropped == ["TCI_R", "TCI_G", "TCI_B"]


def _ssl4eo_cached() -> bool:
    hub = Path(torch.hub.get_dir()) / "checkpoints"
    return any(hub.glob("resnet50_sentinel2_all_moco-*.pth"))


@pytest.mark.skipif(not _ssl4eo_cached(), reason="SSL4EO resnet50 weights not downloaded")
def test_ssl4eo_model_loads_all_13_pretrained_filters():
    from torchgeo.models import ResNet50_Weights

    from gotrain.engine.models import input_normalisation

    state = ResNet50_Weights.SENTINEL2_ALL_MOCO.get_state_dict(progress=False)
    inputs, zero, _ = ssl4eo_inputs(["B02", "B03", "B04"])
    spec = ModelSpec(
        "classification", "resnet50", "ssl4eo_s2", inputs, 2, [0] * 13, [1] * 13, zero_fill=zero
    )
    model = build_model(spec)
    assert torch.equal(model.state_dict()["conv1.weight"], state["conv1.weight"])
    mean, std = input_normalisation(spec)
    assert set(std.tolist()) == {10000.0} and set(mean.tolist()) == {0.0}


@pytest.mark.skipif(not have("m-eurosat"), reason="m-eurosat not downloaded")
def test_zero_filled_channels_are_zero():
    ds = data.GeoBenchDataset(
        "m-eurosat",
        "val",
        bands=["B04", "B06", "B03"],
        norm=([0, 0, 0], [1, 1, 1]),
        zero_fill=["B06"],
        indices=range(2),
    )
    x, _ = ds[0]
    assert float(x[1].abs().max()) == 0.0 and float(x[0].abs().max()) > 0


# ------------------------------------------------------------------ train() never raises
def test_train_returns_error_for_bad_config(tmp_path):
    result = engine.train(cfg(loss="dice_bce"), 1.0, tmp_path / "run-bad")
    assert result.status == "error" and "does not fit" in result.error


def test_train_returns_error_for_unknown_task(tmp_path, monkeypatch):
    monkeypatch.setenv("GEO_BENCH_DIR", str(tmp_path))
    result = engine.train(cfg(task="m-nope"), 1.0, tmp_path / "run-x")
    assert result.status == "error" and "FileNotFoundError" in result.error


def test_evaluate_rejects_unknown_split(tmp_path):
    with pytest.raises(ValueError, match="split"):
        engine.evaluate("m-eurosat", tmp_path / "x.ckpt", "valid")


# ------------------------------------------------------------------ end to end (needs data)
@pytest.mark.skipif(not have("m-eurosat"), reason="m-eurosat not downloaded")
def test_eurosat_trial_end_to_end(tmp_path):
    run = tmp_path / "m-eurosat-test-0001"
    config = cfg(train_subset_frac=0.05, augment=["flip", "rot90"], max_epochs=2)
    result = engine.train(config, budget_minutes=3.0, run_dir=run)
    assert result.status == "ok", result.error
    assert 0.0 <= result.val_score <= 1.0 and result.metric_name == "accuracy"
    assert result.checkpoint_path == run / "checkpoints" / "best.ckpt"
    for name in ("config.json", "history.json", "metrics.json", "val_predictions.npz"):
        assert (run / name).exists(), name
    assert json.loads((run / "metrics.json").read_text())["n_train"] == 100
    assert len(result.history) == 2 and result.minutes_used < 3.0
    rows = json.loads((run / "history.json").read_text())
    assert all(r["train_s"] > 0 and r["val_s"] > 0 for r in rows)
    assert all(r["train_s"] + r["val_s"] <= r["elapsed_s"] for r in rows)
    out = engine.evaluate("m-eurosat", result.checkpoint_path, "val")
    assert out["metric_name"] == "accuracy"
    assert out["score"] == pytest.approx(result.val_score, abs=1e-4)
    tta = engine.evaluate("m-eurosat", result.checkpoint_path, "val", tta=True)
    assert tta["tta"] is True and 0.0 <= tta["score"] <= 1.0


@pytest.mark.skipif(not have("m-eurosat"), reason="m-eurosat not downloaded")
def test_tiny_budget_times_out_cleanly(tmp_path):
    result = engine.train(cfg(), budget_minutes=0.001, run_dir=tmp_path / "run-tiny")
    assert result.status == "timeout" and result.checkpoint_path is None
    assert result.history == []  # tools/trials.py fails fast on 0-epoch timeouts


@pytest.mark.skipif(not have("m-eurosat"), reason="m-eurosat not downloaded")
def test_slow_first_batch_does_not_end_the_trial(tmp_path, monkeypatch):
    """Regression (SageMaker): warm-up in the first batch inflated the val reserve past the
    budget, so a 3-minute trial stopped after ~10 s with 0 epochs."""
    real_augment, calls = engine.augment, []

    def slow_first_batch(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            time.sleep(4.0)  # 4 s / 16 samples = 0.25 s per "sample" if warm-up were counted
        return real_augment(*args, **kwargs)

    monkeypatch.setattr(engine, "augment", slow_first_batch)
    config = cfg(train_subset_frac=0.05, max_epochs=2)  # 100 train tiles, full 1000-tile val
    result = engine.train(config, budget_minutes=1.0, run_dir=tmp_path / "run-slow")
    assert result.status == "ok", result.error
    assert len(result.history) >= 1
    metrics = json.loads((tmp_path / "run-slow" / "metrics.json").read_text())
    assert metrics["full_epochs"] >= 1 and metrics["n_val"] == 1000


@pytest.mark.skipif(not have("m-pv4ger-seg"), reason="m-pv4ger-seg not downloaded")
def test_pv4ger_segmentation_trial_end_to_end(tmp_path):
    run = tmp_path / "m-pv4ger-seg-test-0001"
    config = cfg("m-pv4ger-seg", loss="dice_bce", train_subset_frac=0.05, max_epochs=1)
    result = engine.train(config, budget_minutes=3.0, run_dir=run)
    assert result.status == "ok", result.error
    assert result.metric_name == "miou" and 0.0 <= result.val_score <= 1.0
    preds = np.load(run / "val_predictions.npz")
    assert preds["pred_masks"].shape == (403, 320, 320)  # scored at native resolution
    tta = engine.evaluate("m-pv4ger-seg", result.checkpoint_path, "val", tta=True)
    assert tta["metric_name"] == "miou" and 0.0 <= tta["score"] <= 1.0
