import pytest

from gotrain.contracts import TrainConfig, TrainResult
from gotrain.engine.fake import CARDS, FakeEngine


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine(sleep_s=0)


def config(**overrides) -> TrainConfig:
    cfg = {
        "task": "m-eurosat",
        "backbone": "resnet50",
        "weights": "ssl4eo_s2",
        "img_size": 64,
        "batch_size": 64,
        "lr": 5e-4,
        "max_epochs": 30,
        "loss": "ce",
        "augment": ["flip"],
        "reason": "test",
    }
    return TrainConfig(**(cfg | overrides))


def train(engine, tmp_path, cfg, budget=30.0, name="run") -> TrainResult:
    return engine.train(cfg, budget, tmp_path / name)


def test_returns_valid_ok_result(engine, tmp_path):
    result = train(engine, tmp_path, config())
    assert TrainResult.model_validate(result.model_dump()) == result
    assert result.status == "ok" and result.metric_name == "accuracy"
    assert 0.5 < result.val_score < 1.0
    assert result.checkpoint_path.exists()
    assert len(result.history) == 30
    assert result.history[-1].train_loss < result.history[0].train_loss


def test_deterministic_and_reason_does_not_matter(engine, tmp_path):
    a = train(engine, tmp_path, config(reason="first"), name="a")
    b = train(engine, tmp_path, config(reason="second"), name="b")
    keep = {"val_score", "history", "minutes_used", "metric_name"}
    assert a.model_dump(include=keep) == b.model_dump(include=keep)


def test_seed_changes_result(engine, tmp_path):
    a = train(engine, tmp_path, config(seed=0), name="a")
    b = train(engine, tmp_path, config(seed=1), name="b")
    assert a.val_score != b.val_score


def test_pretrained_and_rot90_score_higher(engine, tmp_path):
    scores = {
        w: train(engine, tmp_path, config(weights=w), name=w).val_score
        for w in ("none", "imagenet", "ssl4eo_s2")
    }
    assert scores["none"] < scores["imagenet"] < scores["ssl4eo_s2"]
    with_rot = train(engine, tmp_path, config(augment=["flip", "rot90"]), name="rot")
    assert with_rot.val_score > scores["ssl4eo_s2"]


def test_small_budget_runs_fewer_epochs_or_times_out(engine, tmp_path):
    short = train(engine, tmp_path, config(), budget=1.0, name="short")
    assert short.status == "ok" and len(short.history) < 30
    seg = config(task="m-pv4ger-seg", img_size=320, loss="dice_bce", weights="imagenet")
    assert train(engine, tmp_path, seg, budget=0.5, name="seg").status == "timeout"


def test_unknown_task_or_band_is_error_result(engine, tmp_path):
    assert train(engine, tmp_path, config(task="m-nope")).status == "error"
    bad_band = train(engine, tmp_path, config(bands=["B99"]), name="band")
    assert bad_band.status == "error" and "B99" in bad_band.error


def test_evaluate_val_matches_training(engine, tmp_path):
    result = train(engine, tmp_path, config())
    out = engine.evaluate("m-eurosat", result.checkpoint_path, "val")
    assert out == {
        "task": "m-eurosat",
        "split": "val",
        "metric_name": "accuracy",
        "score": result.val_score,
    }
    with pytest.raises(ValueError, match="split"):
        engine.evaluate("m-eurosat", result.checkpoint_path, "train")


@pytest.mark.parametrize("task", sorted(CARDS))
def test_describe_dataset_cards(engine, task):
    card = engine.describe_dataset(task)
    assert card.task == task
    assert set(card.split_sizes) <= {"train", "val"}


def test_describe_unknown_task(engine):
    with pytest.raises(ValueError, match="known"):
        engine.describe_dataset("m-nope")
