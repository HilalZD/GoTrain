from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from gotrain.contracts import (
    BandStats,
    DatasetCard,
    ModelVersion,
    TrainConfig,
    TrainResult,
    format_errors,
)


def good_config(**overrides) -> dict:
    cfg = {
        "task": "m-eurosat",
        "backbone": "resnet50",
        "weights": "ssl4eo_s2",
        "img_size": 64,
        "batch_size": 64,
        "lr": 3e-4,
        "max_epochs": 20,
        "loss": "ce",
        "augment": ["flip", "rot90"],
        "reason": "Sentinel-2 task, so start from SSL4EO weights.",
    }
    return cfg | overrides


def rejection(**overrides) -> str:
    with pytest.raises(ValidationError) as exc:
        TrainConfig(**good_config(**overrides))
    return format_errors(exc.value, TrainConfig)


def test_good_config_validates():
    cfg = TrainConfig(**good_config())
    assert cfg.bands == "all"
    assert cfg.train_subset_frac == 1.0


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"lr": 0.5}, "lr: Input should be less than or equal to 0.1"),
        ({"lr": 1e-9}, "lr: Input should be greater than or equal to 0.000001"),
        ({"max_epochs": 500}, "max_epochs"),
        ({"loss": "mse"}, "'ce', 'bce', 'dice_ce', 'dice_bce' or 'focal'"),
        ({"weights": "sentinel"}, "'imagenet', 'ssl4eo_s2' or 'none'"),
        ({"augment": ["mixup"]}, "augment.0"),
        ({"augment": ["flip", "flip"]}, "augment has duplicates ['flip']"),
        ({"bands": []}, "bands list is empty"),
        ({"train_subset_frac": 0.01}, "train_subset_frac"),
        ({"reason": "   "}, "reason"),
    ],
)
def test_bad_config_rejected_with_helpful_message(overrides, expected):
    assert expected in rejection(**overrides)


def test_message_includes_got_value_and_hint():
    msg = rejection(lr=0.5)
    assert "(got 0.5)" in msg
    assert "Hint: Learning rate, 1e-6 to 1e-1." in msg


def test_typo_field_lists_allowed_fields():
    msg = rejection(learning_rate=1e-3)
    assert "learning_rate: Extra inputs are not permitted" in msg
    assert "Allowed fields:" in msg and "lr" in msg


def test_missing_reason_is_reported():
    cfg = good_config()
    del cfg["reason"]
    with pytest.raises(ValidationError) as exc:
        TrainConfig(**cfg)
    assert "reason: Field required" in format_errors(exc.value, TrainConfig)


def inspect_task_output() -> dict:
    """Shaped like the data side's inspect_task output, including optional fields."""
    return {
        "task": "m-pv4ger-seg",
        "task_type": "segmentation",
        "sensor": "aerial-rgb",
        "bands": ["red", "green", "blue"],
        "n_classes": 2,
        "class_names": ["background", "solar_panel"],
        "image_size": [320, 320],
        "split_sizes": {"train": 3000, "valid": 403},
        "class_distribution": {"background": 0.97, "solar_panel": 0.03},
        "band_mean": [120.1, 118.4, 110.2],
        "band_std": [52.3, 48.0, 46.7],
        "notes": ["masks are 0/1", "panels ~3% of pixels"],
        "dtype": "uint8",
        "value_range": [0, 255],
        "label_format": "mask",
        "mask_unique_values": [0, 1],
        "nodata_fraction": 0.0,
        "constant_bands": [],
        "stats_n_samples": 500,
        "some_future_field": "ignored",
    }


def test_inspect_task_output_validates_as_dataset_card():
    card = DatasetCard.model_validate(inspect_task_output())
    assert card.image_size == (320, 320)
    assert card.split_sizes == {"train": 3000, "val": 403}
    assert card.notes == "masks are 0/1\npanels ~3% of pixels"


def test_minimal_card_leaves_optionals_none():
    data = inspect_task_output()
    for key in (
        "dtype",
        "value_range",
        "label_format",
        "mask_unique_values",
        "nodata_fraction",
        "constant_bands",
        "stats_n_samples",
    ):
        del data[key]
    card = DatasetCard.model_validate(data | {"image_size": 320})
    assert card.dtype is None and card.image_size == (320, 320)


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"split_sizes": {"train": 1, "test": 1}}, "test set is sacred"),
        ({"class_names": ["only_one"]}, "n_classes=2 but 1 class_names"),
        ({"band_std": [1.0]}, "must be parallel"),
    ],
)
def test_bad_dataset_card_rejected(overrides, expected):
    with pytest.raises(ValidationError, match=expected):
        DatasetCard.model_validate(inspect_task_output() | overrides)


def test_band_stats_must_be_parallel():
    stats = {
        "task": "m-eurosat",
        "split": "train",
        "n_samples": 100,
        "seed": 0,
        "bands": ["B02", "B03"],
        "mean": [1.0, 2.0],
        "std": [0.5, 0.5],
    }
    assert BandStats.model_validate(stats).excluded_pixels is None
    with pytest.raises(ValidationError, match="must be parallel"):
        BandStats.model_validate(stats | {"std": [0.5]})
    with pytest.raises(ValidationError, match="split"):
        BandStats.model_validate(stats | {"split": "val"})


def test_train_result_status_rules():
    base = {"run_id": "r1", "task": "m-eurosat", "metric_name": "accuracy", "minutes_used": 1}
    with pytest.raises(ValidationError, match="requires both val_score and checkpoint_path"):
        TrainResult(status="ok", **base)
    with pytest.raises(ValidationError, match="requires an error message"):
        TrainResult(status="error", **base)
    assert TrainResult(status="timeout", **base).val_score is None


def test_model_version_format():
    kwargs = {
        "task": "m-eurosat",
        "run_id": "r1",
        "config": good_config(),
        "val_score": 0.9,
        "created_at": datetime.now(timezone.utc),
        "git_commit": "abc123",
        "artifact_path": "s3://bucket/m-eurosat/v1",
    }
    assert ModelVersion(version="v2", **kwargs).config.task == "m-eurosat"
    with pytest.raises(ValidationError, match="version"):
        ModelVersion(version="2", **kwargs)
