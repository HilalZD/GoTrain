"""Data contracts shared by the agent, the tools and the engine.

Every object that crosses a module boundary is one of these Pydantic models, so a
bad value fails at the boundary with a message the agent (or a human) can act on.
Use `format_errors` to turn a ValidationError into text for the agent.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

Weights = Literal["imagenet", "ssl4eo_s2", "none"]
Loss = Literal["ce", "bce", "dice_ce", "dice_bce", "focal"]
Augment = Literal["flip", "rot90", "color_jitter"]
TaskType = Literal["classification", "segmentation", "multilabel"]
VERSION_PATTERN = r"^v[1-9][0-9]*$"


def _no_duplicates(values: list[str], field: str) -> list[str]:
    dupes = sorted({v for v in values if values.count(v) > 1})
    if dupes:
        raise ValueError(f"{field} has duplicates {dupes}; list each item once")
    return values


def _same_length(**lists: list[Any]) -> None:
    lengths = {name: len(values) for name, values in lists.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"these lists must be parallel (same length): {lengths}")


class TrainConfig(BaseModel):
    """One training trial, as proposed by the agent."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    task: str = Field(min_length=1, description="GEO-Bench task name, e.g. 'm-eurosat'.")
    backbone: str = Field(min_length=1, description="Backbone name, e.g. 'resnet50'.")
    weights: Weights = Field(
        description="Pretrained weights. Sentinel-2 tasks should default to 'ssl4eo_s2'."
    )
    bands: Literal["all"] | list[str] = Field(
        default="all", description="'all', or a list of band names from the DatasetCard."
    )
    img_size: int = Field(ge=32, le=1024, description="Input side length in pixels (32-1024).")
    batch_size: int = Field(ge=1, le=1024, description="Batch size (1-1024).")
    lr: float = Field(ge=1e-6, le=1e-1, description="Learning rate, 1e-6 to 1e-1.")
    weight_decay: float = Field(default=0.0, ge=0.0, le=1.0, description="0 to 1.")
    max_epochs: int = Field(
        ge=1, le=200, description="Epoch cap (1-200); the time budget may stop earlier."
    )
    loss: Loss = Field(
        description="Use ce/focal for classification, bce for multilabel, dice_* for segmentation."
    )
    augment: list[Augment] = Field(
        default_factory=list, description="Any of 'flip', 'rot90', 'color_jitter'."
    )
    train_subset_frac: float = Field(
        default=1.0, ge=0.05, le=1.0, description="Fraction of TRAIN used (0.05-1.0)."
    )
    seed: int = Field(default=0, ge=0, description="Random seed (>= 0).")
    reason: str = Field(min_length=1, description="Why the agent chose this config.")

    @field_validator("bands")
    @classmethod
    def _check_bands(cls, v: Literal["all"] | list[str]) -> Literal["all"] | list[str]:
        if isinstance(v, list):
            if not v:
                raise ValueError("bands list is empty; use 'all' or name at least one band")
            _no_duplicates(v, "bands")
        return v

    @field_validator("augment")
    @classmethod
    def _check_augment(cls, v: list[str]) -> list[str]:
        return _no_duplicates(v, "augment")


class EpochRecord(BaseModel):
    epoch: int = Field(ge=1)
    train_loss: float
    val_score: float | None = Field(default=None, ge=0.0, le=1.0)
    elapsed_s: float = Field(ge=0.0)


class TrainResult(BaseModel):
    """What the engine returns after one trial. Scores are on VALIDATION."""

    run_id: str = Field(min_length=1)
    task: str
    status: Literal["ok", "timeout", "error"]
    metric_name: str
    val_score: float | None = Field(default=None, ge=0.0, le=1.0)
    history: list[EpochRecord] = Field(default_factory=list)
    minutes_used: float = Field(ge=0.0)
    checkpoint_path: Path | None = None
    error: str | None = None

    @model_validator(mode="after")
    def _status_consistent(self) -> "TrainResult":
        if self.status == "ok" and (self.val_score is None or self.checkpoint_path is None):
            raise ValueError("status 'ok' requires both val_score and checkpoint_path")
        if self.status == "error" and not self.error:
            raise ValueError("status 'error' requires an error message")
        return self


class DatasetCard(BaseModel):
    """Everything the agent needs to know about a task before proposing configs.

    Built from TRAIN and VAL only. Unknown extra keys are ignored so the data side
    can add fields without breaking the agent.
    """

    task: str
    task_type: TaskType
    sensor: str
    bands: list[str] = Field(min_length=1)
    n_classes: int = Field(ge=1)
    class_names: list[str]
    image_size: tuple[int, int] = Field(description="(height, width); a single int means square.")
    split_sizes: dict[str, int] = Field(description="Sample counts for 'train' and 'val' only.")
    class_distribution: dict[str, float]
    band_mean: list[float]
    band_std: list[float]
    notes: str = ""

    dtype: str | None = None
    value_range: tuple[float, float] | None = None
    label_format: str | None = None
    mask_unique_values: list[int] | None = None
    nodata_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    constant_bands: list[str] | None = None
    stats_n_samples: int | None = Field(default=None, ge=1)

    @field_validator("image_size", mode="before")
    @classmethod
    def _square_if_int(cls, v: Any) -> Any:
        return (v, v) if isinstance(v, int) else v

    @field_validator("notes", mode="before")
    @classmethod
    def _join_note_list(cls, v: Any) -> Any:
        return "\n".join(str(n) for n in v) if isinstance(v, list) else v

    @field_validator("split_sizes", mode="before")
    @classmethod
    def _train_val_only(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            return v
        aliases = {"valid": "val", "validation": "val"}
        sizes = {aliases.get(k, k): n for k, n in v.items()}
        if "test" in sizes:
            raise ValueError("split_sizes must not describe the test split; the test set is sacred")
        unknown = set(sizes) - {"train", "val"}
        if unknown:
            raise ValueError(f"split_sizes keys must be 'train'/'val', got {sorted(unknown)}")
        if "train" not in sizes:
            raise ValueError("split_sizes must include 'train'")
        return sizes

    @model_validator(mode="after")
    def _consistent(self) -> "DatasetCard":
        if len(self.class_names) != self.n_classes:
            raise ValueError(
                f"n_classes={self.n_classes} but {len(self.class_names)} class_names given"
            )
        _same_length(bands=self.bands, band_mean=self.band_mean, band_std=self.band_std)
        if any(n < 0 for n in self.split_sizes.values()):
            raise ValueError(f"split_sizes cannot be negative: {self.split_sizes}")
        return self


class BandStats(BaseModel):
    """Per-band normalisation stats, computed on TRAIN only (stats/<task>.json)."""

    task: str
    split: Literal["train"]
    n_samples: int = Field(ge=1)
    seed: int
    bands: list[str] = Field(min_length=1)
    mean: list[float]
    std: list[float]
    excluded_pixels: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _parallel(self) -> "BandStats":
        _same_length(bands=self.bands, mean=self.mean, std=self.std)
        return self


class ModelVersion(BaseModel):
    """A registered model: one run of one task, or an ensemble of runs, with or without TTA.

    `run_id` and `config` are the best member's; `val_score` is the whole model's (the
    ensemble's, with TTA if set), as measured on VALIDATION.
    """

    task: str
    version: str = Field(pattern=VERSION_PATTERN, description="'v1', 'v2', ...")
    run_id: str
    config: TrainConfig
    val_score: float = Field(ge=0.0, le=1.0)
    created_at: datetime
    git_commit: str = Field(min_length=1)
    artifact_path: str = Field(min_length=1, description="Local path or s3:// URI.")
    members: list[str] = Field(
        default_factory=list, description="Run ids averaged at inference; [run_id] if single."
    )
    tta: bool = Field(default=False, description="Test-time augmentation (8 flip/rot90 views).")

    @model_validator(mode="after")
    def _members_include_run(self) -> "ModelVersion":
        if not self.members:  # versions registered before ensembles existed
            self.members = [self.run_id]
        if self.run_id not in self.members:
            raise ValueError("run_id must be one of the members")
        if len(set(self.members)) != len(self.members):
            raise ValueError("members must be distinct runs")
        return self


class TestRecord(BaseModel):
    """The one-time test-set score of a registered model version."""

    __test__ = False  # not a pytest test class

    task: str
    version: str = Field(pattern=VERSION_PATTERN)
    test_score: float = Field(ge=0.0, le=1.0)
    metric_name: str
    git_commit: str = Field(min_length=1)
    timestamp: datetime


def format_errors(err: ValidationError, model: type[BaseModel] | None = None) -> str:
    """Render a ValidationError as short, actionable lines for the agent."""
    lines = [f"{err.error_count()} problem(s) with {err.title}:"]
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"]) or "(whole object)"
        line = f"- {loc}: {e['msg']}"
        if e["type"] not in ("missing", "model_type"):
            line += f" (got {e['input']!r})"
        line += _hint(model, e)
        lines.append(line)
    return "\n".join(lines)


def _hint(model: type[BaseModel] | None, error: Any) -> str:
    if model is None or not error["loc"]:
        return ""
    if error["type"] == "extra_forbidden":
        return f". Allowed fields: {', '.join(model.model_fields)}"
    field = model.model_fields.get(str(error["loc"][0]))
    return f". Hint: {field.description}" if field and field.description else ""
