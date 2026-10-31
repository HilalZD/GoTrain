"""The contract between the agent side and the ML side (v2; full spec: docs/contract-v2.md).

Any engine (fake or real) provides:

    train(config, budget_minutes, run_dir) -> TrainResult   scored on VALIDATION
    evaluate(task, checkpoint_path | [paths], split, tta=False) -> dict
        keys: task, split, metric_name, score, n_samples, per_sample, n_checkpoints, tta
        per_sample: {"correct": [0/1] * N}                      (classification)
                    {"intersection": N x K, "union": N x K}     (segmentation, pixel counts)
        A list of checkpoints is an ensemble: their probabilities are averaged.
    predict(task, checkpoint_path | [paths], images N x C x H x W, tta=False) -> dict
        keys: probabilities (N x K, or N x K x H x W), labels (N, or N x H x W)
    describe_dataset(task) -> DatasetCard   built from train/val only
    diagnose(task, run_dir) -> dict   keys: per_class, confusion_matrix,
                                      curve_verdict ("overfitting"|"underfitting"|"fine"),
                                      error_grid_path (optional)

v1 calls still work: evaluate(task, path, split) is evaluate(..., tta=False) on one model.

The real engine is assembled from Serena's modules, imported lazily so this repo
works before they exist:
    gotrain.engine.train  must define  train, evaluate, diagnose and predict
    gotrain.engine.data   must define  describe_dataset

Only gotrain/tools/final_test.py may ask evaluate() for the test split.
"""

import importlib
import importlib.util
from pathlib import Path
from typing import Protocol

import numpy as np

from gotrain.config import get_settings
from gotrain.contracts import BandStats, DatasetCard, TrainConfig, TrainResult

REAL_MODULES = ("gotrain.engine.train", "gotrain.engine.data")
Checkpoints = Path | list[Path]  # one model, or an ensemble whose probabilities are averaged


class Engine(Protocol):
    def train(self, config: TrainConfig, budget_minutes: float, run_dir: Path) -> TrainResult: ...

    def evaluate(
        self, task: str, checkpoint_path: Checkpoints, split: str, tta: bool = False
    ) -> dict: ...

    def predict(
        self, task: str, checkpoint_path: Checkpoints, images: np.ndarray, tta: bool = False
    ) -> dict: ...

    def describe_dataset(self, task: str) -> DatasetCard: ...

    def diagnose(self, task: str, run_dir: Path) -> dict: ...


class RealEngine:
    """Thin adapter over Serena's modules; nothing heavy is imported until a call."""

    def train(self, config: TrainConfig, budget_minutes: float, run_dir: Path) -> TrainResult:
        return importlib.import_module("gotrain.engine.train").train(
            config, budget_minutes, run_dir
        )

    def evaluate(
        self, task: str, checkpoint_path: Checkpoints, split: str, tta: bool = False
    ) -> dict:
        # `tta` is only passed when set, so a v1 evaluate(task, path, split) keeps working.
        extra = {"tta": True} if tta else {}
        return importlib.import_module("gotrain.engine.train").evaluate(
            task, checkpoint_path, split, **extra
        )

    def predict(
        self, task: str, checkpoint_path: Checkpoints, images: np.ndarray, tta: bool = False
    ) -> dict:
        extra = {"tta": True} if tta else {}
        return importlib.import_module("gotrain.engine.train").predict(
            task, checkpoint_path, images, **extra
        )

    def describe_dataset(self, task: str) -> DatasetCard:
        return importlib.import_module("gotrain.engine.data").describe_dataset(task)

    def diagnose(self, task: str, run_dir: Path) -> dict:
        return importlib.import_module("gotrain.engine.train").diagnose(task, run_dir)


def get_engine() -> Engine:
    """Return the engine selected by GOTRAIN_ENGINE ("fake" by default)."""
    settings = get_settings()
    if settings.engine == "fake":
        from gotrain.engine.fake import FakeEngine

        return FakeEngine(sleep_s=settings.fake_sleep_s)
    missing = [m for m in REAL_MODULES if importlib.util.find_spec(m) is None]
    if missing:
        raise RuntimeError(
            f"GOTRAIN_ENGINE=real but {missing} not found yet. "
            "Use GOTRAIN_ENGINE=fake until the real engine is merged."
        )
    return RealEngine()


def load_band_stats(task: str) -> BandStats:
    """Read stats/<task>.json (location set by GOTRAIN_STATS_DIR)."""
    path = get_settings().stats_dir / f"{task}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No band stats for {task!r} at {path}. They are produced by the stats/ scripts."
        )
    return BandStats.model_validate_json(path.read_text(encoding="utf-8"))
