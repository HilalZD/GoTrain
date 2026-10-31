"""Tests for gotrain/engine/diagnostics.py (Serena's diagnose()).

Pure tests run anywhere; end-to-end tests train a tiny run first and need the GEO-Bench files.
"""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("matplotlib")

from gotrain.contracts import TrainConfig  # noqa: E402
from gotrain.engine import data  # noqa: E402
from gotrain.engine import diagnostics as diag  # noqa: E402
from gotrain.engine import train as engine  # noqa: E402
from gotrain.engine.fake import FakeEngine  # noqa: E402
from gotrain.tools.diagnose import summarize_diagnosis  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def repo_stats(monkeypatch):
    monkeypatch.setenv("GOTRAIN_STATS_DIR", str(REPO / "stats"))


def have(task: str) -> bool:
    """True only if geobench is importable AND the task is on disk (else data tests skip)."""
    try:
        import geobench  # noqa: F401

        data.find_task_dir(task)
        return True
    except (ImportError, FileNotFoundError):
        return False


def test_diagnose_is_reexported_from_train():
    assert engine.diagnose is diag.diagnose


@pytest.mark.parametrize(
    ("scores", "verdict"),
    [
        ([0.5, 0.7, 0.8, 0.857, 0.86, 0.86], "fine"),
        ([0.5, 0.8, 0.9, 0.85, 0.82, 0.80], "overfitting"),
        ([0.3, 0.4, 0.5, 0.6, 0.7, 0.8], "underfitting"),
        ([0.5, 0.6], "underfitting"),
        ([], "underfitting"),
    ],
)
def test_curve_verdict(scores, verdict):
    assert diag.curve_verdict(scores) == verdict


def test_classification_report_rows_sum_to_one():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    per_class, matrix = diag.classification_report(y_true, y_pred, ["a", "b", "c"])
    assert per_class["a"]["recall"] == 0.5 and per_class["b"]["recall"] == 1.0
    assert list(per_class["a"])[0] == "recall"  # tools/diagnose.py reads the first key
    assert np.allclose(np.sum(matrix, axis=1), 1.0)
    assert matrix[0] == [0.5, 0.5, 0.0]


def test_segmentation_report_headline_is_iou():
    confusion = np.array([[80, 0], [10, 10]])
    per_class, matrix = diag.segmentation_report(confusion, ["bg", "panel"])
    assert list(per_class["panel"])[0] == "iou"
    assert per_class["panel"]["iou"] == pytest.approx(10 / 20)
    assert matrix[1] == [0.5, 0.5]


def test_multilabel_report_rows():
    y = np.array([[1, 0], [1, 1], [0, 1]])
    p = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]])
    per_class, matrix = diag.multilabel_report(y, p, ["x", "y"])
    assert per_class["y"]["recall"] == 0.5
    assert matrix[0] == [1.0, 0.0]


def test_rgb_indices_by_name():
    assert diag.rgb_indices(["B01", "B02", "B03", "B04"]) == [3, 2, 1]
    assert diag.rgb_indices(["Red", "Green", "Blue"]) == [0, 1, 2]
    assert diag.rgb_indices(["x", "y"]) == [0, 1, 1]


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        diag.diagnose("m-eurosat", tmp_path)


def _same_keys_as_fake(out: dict, tmp_path: Path, task: str) -> None:
    cfg = TrainConfig(
        task=task,
        backbone="resnet50",
        weights="imagenet",
        img_size=64,
        batch_size=8,
        lr=1e-3,
        max_epochs=3,
        loss="ce",
        reason="fake",
    )
    fake = FakeEngine(sleep_s=0)
    run = tmp_path / "fake-run"
    fake.train(cfg, 60, run)
    assert set(out) == set(fake.diagnose(task, run))


@pytest.mark.skipif(not have("m-eurosat"), reason="needs geobench + m-eurosat on disk")
def test_diagnose_eurosat_run(tmp_path):
    run = tmp_path / "m-eurosat-diag-0001"
    config = TrainConfig(
        task="m-eurosat",
        backbone="resnet18",
        weights="none",
        img_size=64,
        batch_size=16,
        lr=1e-3,
        max_epochs=2,
        loss="ce",
        train_subset_frac=0.05,
        reason="diag test",
    )
    assert engine.train(config, 3.0, run).status == "ok"
    out = engine.diagnose("m-eurosat", run)
    _same_keys_as_fake(out, tmp_path, "m-eurosat")
    assert out["metric_name"] == "accuracy" and out["class_names"][9] == "Sea and Lake"
    assert len(out["confusion_matrix"]) == 10
    assert all(abs(sum(r) - 1) < 1e-3 or sum(r) == 0 for r in out["confusion_matrix"])
    assert out["curve_verdict"] in ("overfitting", "underfitting", "fine")
    assert Path(out["error_grid_path"]).exists()
    summary = summarize_diagnosis(out)  # Hilal's tool can digest it
    assert summary["per_class_metric"] == "recall"


@pytest.mark.skipif(not have("m-pv4ger-seg"), reason="needs geobench + m-pv4ger-seg on disk")
def test_diagnose_pv4ger_run(tmp_path):
    run = tmp_path / "m-pv4ger-seg-diag-0001"
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
        reason="diag test",
    )
    assert engine.train(config, 3.0, run).status == "ok"
    out = engine.diagnose("m-pv4ger-seg", run)
    assert set(out["per_class"]) == {"no solar pv", "solar pv"}
    assert summarize_diagnosis(out)["per_class_metric"] == "iou"
    assert Path(out["error_grid_path"]).name == "errors.png"


# ------------------------------------------------------------------ miou_excl_nodata (request #9)
def test_miou_excluding_nodata_by_hand():
    # rows = truth (0 = no data), cols = prediction
    cm = np.array([[50, 5, 5], [2, 8, 0], [0, 2, 8]])
    got = engine.miou_excluding(cm, nodata=0)
    iou1 = 8 / (10 + (8 + 2) - 8)  # row sum 10, predicted-1 among real truth = 8 + 2
    iou2 = 8 / (10 + (0 + 8) - 8)
    assert got == pytest.approx((iou1 + iou2) / 2)
    official = engine.miou_from_confusion(cm)  # all 3 classes, as score
    assert official != pytest.approx(got)


def test_nodata_extras_only_for_tasks_with_a_nodata_class():
    cm = np.eye(7, dtype=np.int64) * 10
    assert engine.nodata_extras("m-pv4ger-seg", cm[:2, :2]) == {}
    extras = engine.nodata_extras("m-chesapeake", cm)
    assert extras == {"miou_excl_nodata": 1.0, "nodata_class": 0}


@pytest.mark.skipif(not have("m-pv4ger-seg"), reason="needs geobench + m-pv4ger-seg on disk")
def test_extra_metric_reaches_metrics_json_and_diagnose(tmp_path, monkeypatch):
    """Wiring check on local data: pretend pv4ger's class 0 were a no-data class."""
    import json

    monkeypatch.setitem(engine.NODATA_CLASSES, "m-pv4ger-seg", 0)
    run = tmp_path / "m-pv4ger-seg-nodata-0001"
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
        reason="extra metric",
    )
    result = engine.train(config, 3.0, run)
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["nodata_class"] == 0 and 0.0 <= metrics["miou_excl_nodata"] <= 1.0
    assert metrics["val_score"] == result.val_score  # the official score is unchanged
    out = engine.diagnose("m-pv4ger-seg", run)
    assert out["miou_excl_nodata"] == metrics["miou_excl_nodata"]
    assert summarize_diagnosis(out)["per_class_metric"] == "iou"  # Hilal's tool still works
