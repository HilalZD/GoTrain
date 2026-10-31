"""Contract v2 for the REAL engine (docs/contract-v2.md): evaluate ensembles/TTA/per_sample,
predict. Hilal's tests/test_engine_v2.py covers the fake engine; this file covers ours.

Pure tests build tiny untrained checkpoints from the committed DatasetCards, so they need
neither geobench nor the data. Tests marked with the data markers train a tiny run first.
"""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("segmentation_models_pytorch")

from gotrain.contracts import TrainConfig  # noqa: E402
from gotrain.engine import data  # noqa: E402
from gotrain.engine import train as engine  # noqa: E402
from gotrain.engine.models import ModelSpec, build_model  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ATOL = 2e-3 if torch.cuda.is_available() else 1e-5  # fp16 autocast on GPU


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


def cfg(task: str, **kw) -> TrainConfig:
    base = {
        "task": task,
        "backbone": "resnet18",
        "weights": "none",
        "img_size": 64,
        "batch_size": 16,
        "lr": 1e-3,
        "max_epochs": 1,
        "loss": "ce",
        "reason": "v2 test",
    }
    return TrainConfig(**(base | kw))


def make_ckpt(path: Path, task: str, card=None, seed: int = 0, **kw) -> Path:
    """An untrained but valid checkpoint for `task`, as train() would write it."""
    torch.manual_seed(seed)
    card = card or data.describe_dataset(task)
    config = cfg(task, **kw)
    stats = engine.load_band_stats(task)
    spec, _ = engine.model_spec(config, card, stats)
    model = build_model(ModelSpec(**{**spec.__dict__, "weights": "none"}))
    norm = engine.model_normalisation(spec)
    engine.save_checkpoint(path, model, config, card, spec, norm, 0.5, 1)
    return path


def raw_eurosat(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 4000, size=(n, 13, 64, 64)).astype(np.float32)


# ------------------------------------------------------------------ per_sample and scores
def test_segmentation_score_is_dataset_level_not_mean_of_images():
    # image 0: panel I=1/U=1 (IoU 1.0); image 1: panel I=0/U=99 (IoU 0.0)
    per_sample = {"intersection": [[50, 1], [40, 0]], "union": [[50, 1], [139, 99]]}
    got = engine.score_from_per_sample("segmentation", per_sample)
    dataset_level = np.mean([90 / 189, 1 / 100])
    mean_of_images = np.mean([np.mean([1, 1]), np.mean([40 / 139, 0])])
    assert got == pytest.approx(dataset_level)
    assert abs(got - mean_of_images) > 0.1


def test_segmentation_score_skips_classes_with_no_union():
    per_sample = {"intersection": [[10, 0, 0]], "union": [[20, 0, 0]]}
    assert engine.score_from_per_sample("segmentation", per_sample) == pytest.approx(0.5)


def test_classification_and_multilabel_scores():
    assert engine.score_from_per_sample("classification", {"correct": [1, 0, 1, 1]}) == 0.75
    ml = {"tp": [2, 1], "fp": [0, 1], "fn": [1, 0]}
    assert engine.score_from_per_sample("multilabel", ml) == pytest.approx(6 / (6 + 1 + 1))


def test_per_sample_batch_segmentation_excludes_ignore_pixels():
    probs = np.zeros((1, 2, 2, 2), np.float32)
    probs[0, 1] = 1.0  # predict class 1 everywhere
    y = np.array([[[1, 0], [data.IGNORE_INDEX, 1]]])
    out = engine.per_sample_batch("segmentation", probs, y, 2)
    assert out == {"intersection": [[0, 2]], "union": [[1, 3]]}


# ------------------------------------------------------------------ split names
def test_val_maps_to_geobench_valid_and_test_needs_evaluate():
    assert data.GEOBENCH_SPLITS["val"] == "valid"
    with pytest.raises(ValueError, match="'train' or 'val'"):
        data.open_split(object(), "valid")  # geobench's name is not an engine split
    with pytest.raises(PermissionError, match="sacred"):
        data.open_split(object(), data.TEST_SPLIT)


def test_evaluate_rejects_geobench_split_names(tmp_path):
    with pytest.raises(ValueError, match="split"):
        engine.evaluate("m-eurosat", tmp_path / "x.ckpt", "valid")


# ------------------------------------------------------------------ inputs per checkpoint
def test_checkpoint_inputs_reorder_zero_fill_and_normalise():
    raw = torch.arange(3 * 4, dtype=torch.float32).view(1, 3, 2, 2) + 1  # bands A, B, C
    ckpt = {
        "spec": {"bands": ["C", "X", "A"], "zero_fill": ["X"]},
        "norm_mean": [1.0, 0.0, 0.0],
        "norm_std": [2.0, 1.0, 10.0],
    }
    x = engine.checkpoint_inputs(raw, ckpt, ["A", "B", "C"])
    assert torch.allclose(x[0, 0], (raw[0, 2] - 1) / 2)
    assert float(x[0, 1].abs().max()) == 0.0
    assert torch.allclose(x[0, 2], raw[0, 0] / 10)


# ------------------------------------------------------------------ predict
def test_predict_classification_shapes_and_probabilities(tmp_path):
    ckpt = make_ckpt(tmp_path / "a.ckpt", "m-eurosat")
    out = engine.predict("m-eurosat", ckpt, raw_eurosat(3))
    assert out["probabilities"].shape == (3, 10) and out["probabilities"].dtype == np.float32
    assert np.allclose(out["probabilities"].sum(1), 1.0, atol=1e-5)
    assert out["labels"].dtype == np.int64
    assert np.array_equal(out["labels"], out["probabilities"].argmax(1))
    assert out["n_checkpoints"] == 1 and out["tta"] is False


def test_predict_segmentation_shapes(tmp_path):
    ckpt = make_ckpt(tmp_path / "s.ckpt", "m-pv4ger-seg", loss="dice_bce")  # 1-logit U-Net
    images = np.random.default_rng(0).uniform(0, 255, (2, 3, 64, 64)).astype(np.float32)
    out = engine.predict("m-pv4ger-seg", [ckpt], images, tta=True)
    assert out["probabilities"].shape == (2, 2, 64, 64)
    assert np.allclose(out["probabilities"].sum(1), 1.0, atol=1e-5)
    assert out["labels"].shape == (2, 64, 64) and out["tta"] is True


@pytest.mark.parametrize("tta", [False, True])
def test_predict_is_independent_of_the_batch(tmp_path, tta):
    ckpt = make_ckpt(tmp_path / "a.ckpt", "m-eurosat")
    images = raw_eurosat(5)
    batch = engine.predict("m-eurosat", ckpt, images, tta=tta)["probabilities"]
    alone = engine.predict("m-eurosat", ckpt, images[2:3], tta=tta)["probabilities"]
    assert np.allclose(batch[2], alone[0], atol=ATOL)


def test_predict_rejects_wrong_band_count(tmp_path):
    ckpt = make_ckpt(tmp_path / "a.ckpt", "m-eurosat")
    with pytest.raises(ValueError, match="N x 13 x H x W"):
        engine.predict("m-eurosat", ckpt, np.zeros((2, 3, 64, 64), np.float32))
    with pytest.raises(ValueError, match="N x 13 x H x W"):
        engine.predict("m-eurosat", ckpt, np.zeros((13, 64, 64), np.float32))


def test_tta_averages_probabilities_over_eight_views(tmp_path):
    ckpt_path = make_ckpt(tmp_path / "a.ckpt", "m-eurosat")
    images = raw_eurosat(2)
    ckpt, model = engine.load_checkpoint(ckpt_path, torch.device("cpu"))
    model.eval()
    x = engine.checkpoint_inputs(torch.from_numpy(images), ckpt, data.SSL4EO_S2_BANDS)
    with torch.no_grad():
        manual = torch.stack([model(fwd(x)).softmax(1) for fwd, _ in engine._tta_views(x)])
    got = engine.predict("m-eurosat", ckpt_path, images, tta=True)["probabilities"]
    bands = list(data.describe_dataset("m-eurosat").bands)
    if bands != data.SSL4EO_S2_BANDS:  # inputs must follow the card's order
        x = engine.checkpoint_inputs(torch.from_numpy(images), ckpt, bands)
        with torch.no_grad():
            manual = torch.stack([model(fwd(x)).softmax(1) for fwd, _ in engine._tta_views(x)])
    assert np.allclose(got, manual.mean(0).numpy(), atol=ATOL)


# ------------------------------------------------------------------ ensembles
def test_ensemble_averages_member_probabilities_with_mixed_backbones_and_sizes(tmp_path):
    a = make_ckpt(tmp_path / "a.ckpt", "m-eurosat", seed=1)
    b = make_ckpt(tmp_path / "b.ckpt", "m-eurosat", seed=2, backbone="resnet34", img_size=96)
    images = raw_eurosat(4)
    pa = engine.predict("m-eurosat", a, images)["probabilities"]
    pb = engine.predict("m-eurosat", b, images)["probabilities"]
    both = engine.predict("m-eurosat", [a, b], images)
    assert both["n_checkpoints"] == 2
    assert np.allclose(both["probabilities"], (pa + pb) / 2, atol=ATOL)
    reversed_order = engine.predict("m-eurosat", [b, a], images)["probabilities"]
    assert np.allclose(both["probabilities"], reversed_order, atol=ATOL)


def test_ensemble_refuses_checkpoints_of_another_task(tmp_path):
    a = make_ckpt(tmp_path / "a.ckpt", "m-eurosat")
    seg = make_ckpt(tmp_path / "s.ckpt", "m-pv4ger-seg", loss="dice_bce")
    with pytest.raises(ValueError, match="cannot score task 'm-eurosat'"):
        engine.predict("m-eurosat", [a, seg], raw_eurosat(1))


def test_ensemble_refuses_different_class_counts(tmp_path):
    card = data.describe_dataset("m-eurosat")
    five = card.model_copy(
        update={
            "n_classes": 5,
            "class_names": card.class_names[:5],
            "class_distribution": dict(list(card.class_distribution.items())[:5]),
        }
    )
    a = make_ckpt(tmp_path / "a.ckpt", "m-eurosat")
    b = make_ckpt(tmp_path / "b.ckpt", "m-eurosat", card=five)
    with pytest.raises(ValueError, match="disagree on the classes"):
        engine.predict("m-eurosat", [a, b], raw_eurosat(1))


def test_empty_checkpoint_list_is_an_error():
    with pytest.raises(ValueError, match="no checkpoints"):
        engine.checkpoint_list([])


# ------------------------------------------------------------------ evaluate (needs data)
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """One tiny trained run per available task: {task: (TrainResult, run_dir)}."""
    mp = pytest.MonkeyPatch()
    base = tmp_path_factory.mktemp("v2runs")
    mp.setenv("GOTRAIN_STATS_DIR", str(REPO / "stats"))
    mp.setenv("GOTRAIN_HOME", str(base / "gotrain_dev"))  # conftest's home is per-test only
    mp.setenv("GOTRAIN_ENGINE", "fake")
    runs = {}
    for task, loss in (("m-eurosat", "ce"), ("m-pv4ger-seg", "dice_bce")):
        if have(task):
            run = base / f"{task}-v2-0001"
            config = cfg(task, loss=loss, train_subset_frac=0.05, augment=["flip"])
            result = engine.train(config, 3.0, run)
            assert result.status == "ok", result.error
            runs[task] = (result, run)
    yield runs
    mp.undo()


@pytest.mark.skipif(not have("m-eurosat"), reason="needs geobench + m-eurosat on disk")
def test_evaluate_classification_v2_format(trained):
    result, _ = trained["m-eurosat"]
    out = engine.evaluate("m-eurosat", result.checkpoint_path, "val")
    expected = {"task", "split", "metric_name", "score", "n_samples", "per_sample"}
    assert expected | {"n_checkpoints", "tta"} == set(out)
    assert out["split"] == "val" and out["n_samples"] == 1000  # full, fixed val set
    correct = out["per_sample"]["correct"]
    assert len(correct) == 1000 and set(correct) <= {0, 1}
    assert out["score"] == pytest.approx(np.mean(correct), abs=1e-5)
    assert out["score"] == pytest.approx(result.val_score, abs=1e-4)  # same as train()


@pytest.mark.skipif(not have("m-pv4ger-seg"), reason="needs geobench + m-pv4ger-seg on disk")
def test_evaluate_segmentation_matches_train_and_is_dataset_level(trained):
    result, _ = trained["m-pv4ger-seg"]
    out = engine.evaluate("m-pv4ger-seg", [result.checkpoint_path], "val")
    inter = np.asarray(out["per_sample"]["intersection"])
    union = np.asarray(out["per_sample"]["union"])
    assert inter.shape == union.shape == (403, 2)
    assert (inter <= union).all()
    assert out["metric_name"] == "miou" and out["n_checkpoints"] == 1
    assert out["score"] == pytest.approx(np.mean(inter.sum(0) / union.sum(0)), abs=1e-5)
    assert out["score"] == pytest.approx(result.val_score, abs=1e-4)  # same as train()


@pytest.mark.skipif(not have("m-eurosat"), reason="needs geobench + m-eurosat on disk")
def test_evaluate_ensemble_and_tta_on_val(trained, tmp_path):
    result, _ = trained["m-eurosat"]
    ckpt = result.checkpoint_path
    single = engine.evaluate("m-eurosat", ckpt, "val")
    twice = engine.evaluate("m-eurosat", [ckpt, ckpt], "val")
    assert twice["n_checkpoints"] == 2 and twice["score"] == pytest.approx(single["score"])
    tta = engine.evaluate("m-eurosat", [ckpt, ckpt], "val", tta=True)
    assert tta["tta"] is True and 0.0 <= tta["score"] <= 1.0
    seg = make_ckpt(tmp_path / "s.ckpt", "m-pv4ger-seg", loss="dice_bce")
    with pytest.raises(ValueError, match="cannot score"):
        engine.evaluate("m-eurosat", [ckpt, seg], "val")
