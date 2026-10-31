"""Engine contract v2: evaluate (ensembles, tta, per_sample) and predict, plus v1 compatibility."""

import sys
from types import ModuleType

import numpy as np
import pytest
from conftest import make_config

from gotrain.contracts import TrainConfig
from gotrain.engine.fake import CARDS, FakeEngine, miou
from gotrain.engine.interface import RealEngine


@pytest.fixture(scope="module")
def engine() -> FakeEngine:
    return FakeEngine(sleep_s=0)


def checkpoint(engine, tmp_path, task: str, seed: int = 0, **overrides):
    if CARDS[task].task_type == "segmentation":
        overrides = {"weights": "imagenet", "img_size": 320, "loss": "dice_bce"} | overrides
    cfg = TrainConfig(task=task, **make_config(seed=seed, **overrides))
    result = engine.train(cfg, 30.0, tmp_path / f"{task}-{seed}")
    return result.checkpoint_path


# ---------------------------------------------------------------- evaluate


def test_classification_per_sample_is_one_flag_per_image(engine, tmp_path):
    out = engine.evaluate("m-eurosat", checkpoint(engine, tmp_path, "m-eurosat"), "val")
    flags = out["per_sample"]["correct"]
    assert out["n_samples"] == len(flags) == CARDS["m-eurosat"].split_sizes["val"]
    assert set(flags) <= {0, 1}
    assert out["score"] == pytest.approx(sum(flags) / len(flags), abs=1e-4)
    assert out["n_checkpoints"] == 1 and out["tta"] is False


def test_segmentation_per_sample_has_per_image_per_class_counts(engine, tmp_path):
    task = "m-pv4ger-seg"
    out = engine.evaluate(task, checkpoint(engine, tmp_path, task), "val")
    inter = np.array(out["per_sample"]["intersection"])
    union = np.array(out["per_sample"]["union"])
    n, k = out["n_samples"], CARDS[task].n_classes
    assert inter.shape == union.shape == (n, k)
    assert (inter >= 0).all() and (inter <= union).all()
    assert out["metric_name"] == "miou"
    assert out["score"] == pytest.approx(miou(out["per_sample"]), abs=1e-4)


def test_ensemble_beats_its_average_member_and_ignores_order(engine, tmp_path):
    paths = [checkpoint(engine, tmp_path, "m-eurosat", seed=s) for s in range(3)]
    singles = [engine.evaluate("m-eurosat", p, "val")["score"] for p in paths]
    ensemble = engine.evaluate("m-eurosat", paths, "val")
    assert ensemble["n_checkpoints"] == 3
    assert ensemble["score"] > sum(singles) / 3
    assert engine.evaluate("m-eurosat", paths[::-1], "val") == ensemble
    assert engine.evaluate("m-eurosat", [paths[0]], "val") == engine.evaluate(
        "m-eurosat", paths[0], "val"
    )  # a one-element list is a single model


@pytest.mark.parametrize("task", ["m-eurosat", "m-pv4ger-seg"])
def test_tta_raises_the_score_deterministically(engine, tmp_path, task):
    path = checkpoint(engine, tmp_path, task)
    plain = engine.evaluate(task, path, "val")
    with_tta = engine.evaluate(task, path, "val", tta=True)
    assert with_tta["tta"] is True and with_tta["score"] > plain["score"]
    assert engine.evaluate(task, path, "val", tta=True) == with_tta


def test_test_split_scores_lower_than_val(engine, tmp_path):
    path = checkpoint(engine, tmp_path, "m-eurosat")
    assert (
        engine.evaluate("m-eurosat", path, "test")["score"]
        <= engine.evaluate("m-eurosat", path, "val")["score"]
    )


def test_evaluate_rejects_bad_input(engine, tmp_path):
    path = checkpoint(engine, tmp_path, "m-eurosat")
    with pytest.raises(ValueError, match="split"):
        engine.evaluate("m-eurosat", path, "train")
    with pytest.raises(ValueError, match="empty list"):
        engine.evaluate("m-eurosat", [], "val")
    with pytest.raises(ValueError, match="not 'm-brick-kiln'"):
        engine.evaluate("m-brick-kiln", path, "val")


# ---------------------------------------------------------------- predict


def test_predict_classification_shapes(engine, tmp_path):
    path = checkpoint(engine, tmp_path, "m-eurosat")
    images = np.random.default_rng(0).uniform(0, 3000, (5, 13, 64, 64))
    out = engine.predict("m-eurosat", path, images)
    assert out["probabilities"].shape == (5, 10) and out["labels"].shape == (5,)
    assert np.allclose(out["probabilities"].sum(axis=1), 1, atol=1e-5)
    assert (out["labels"] == out["probabilities"].argmax(axis=1)).all()


def test_predict_segmentation_shapes(engine, tmp_path):
    task = "m-pv4ger-seg"
    path = checkpoint(engine, tmp_path, task)
    images = np.random.default_rng(0).uniform(0, 255, (2, 3, 32, 48))
    out = engine.predict(task, path, images, tta=True)
    assert out["probabilities"].shape == (2, 2, 32, 48) and out["labels"].shape == (2, 32, 48)
    assert np.allclose(out["probabilities"].sum(axis=1), 1, atol=1e-5)
    assert out["tta"] is True


def test_predict_ensemble_averages_probabilities(engine, tmp_path):
    paths = [checkpoint(engine, tmp_path, "m-eurosat", seed=s) for s in range(2)]
    images = np.random.default_rng(1).uniform(0, 3000, (4, 13, 16, 16))
    singles = [engine.predict("m-eurosat", p, images)["probabilities"] for p in paths]
    both = engine.predict("m-eurosat", paths, images)
    assert np.allclose(both["probabilities"], (singles[0] + singles[1]) / 2, atol=1e-6)
    assert both["n_checkpoints"] == 2


def test_predict_is_per_image(engine, tmp_path):
    path = checkpoint(engine, tmp_path, "m-eurosat")
    images = np.random.default_rng(2).uniform(0, 3000, (4, 13, 16, 16))
    whole = engine.predict("m-eurosat", path, images)["probabilities"]
    first = engine.predict("m-eurosat", path, images[:1])["probabilities"]
    assert np.allclose(whole[:1], first)  # no dependence on the rest of the batch


def test_predict_rejects_wrong_band_count(engine, tmp_path):
    path = checkpoint(engine, tmp_path, "m-eurosat")
    with pytest.raises(ValueError, match="N x 13 x H x W"):
        engine.predict("m-eurosat", path, np.zeros((2, 3, 64, 64)))


# ---------------------------------------------------------------- real-engine adapter


def test_real_engine_still_supports_a_v1_evaluate(monkeypatch, tmp_path):
    """Serena's evaluate(task, path, split) keeps working until she adds tta."""
    module = ModuleType("gotrain.engine.train")
    calls = []

    def evaluate(task, checkpoint_path, split):  # v1 signature: no tta parameter
        calls.append((task, checkpoint_path, split))
        return {"task": task, "split": split, "metric_name": "accuracy", "score": 0.9}

    module.evaluate = evaluate
    monkeypatch.setitem(sys.modules, "gotrain.engine.train", module)
    out = RealEngine().evaluate("m-eurosat", tmp_path / "ckpt", "val")
    assert out["score"] == 0.9 and calls == [("m-eurosat", tmp_path / "ckpt", "val")]
    with pytest.raises(TypeError):
        RealEngine().evaluate("m-eurosat", tmp_path / "ckpt", "val", tta=True)  # needs v2


def test_real_engine_forwards_v2_arguments(monkeypatch, tmp_path):
    module = ModuleType("gotrain.engine.train")
    module.evaluate = lambda task, ckpt, split, tta=False: {"tta": tta, "ckpt": ckpt}
    module.predict = lambda task, ckpt, images, tta=False: {"tta": tta, "n": len(images)}
    monkeypatch.setitem(sys.modules, "gotrain.engine.train", module)
    paths = [tmp_path / "a", tmp_path / "b"]
    assert RealEngine().evaluate("t", paths, "val", tta=True) == {"tta": True, "ckpt": paths}
    assert RealEngine().predict("t", paths, np.zeros((3, 1, 2, 2)), tta=True)["n"] == 3
