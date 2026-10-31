"""The contract between the agent side and the ML side.

Any engine (fake or real) provides three methods:

    train(config, budget_minutes, run_dir) -> TrainResult   scored on VALIDATION
    evaluate(task, checkpoint_path, split) -> dict   keys: task, split, metric_name, score
    describe_dataset(task) -> DatasetCard   built from train/val only

The real engine is assembled from Serena's modules, imported lazily so this repo
works before they exist:
    gotrain.engine.train  must define  train(...) and evaluate(...)
    gotrain.engine.data   must define  describe_dataset(...)

Only gotrain/tools/final_test.py may ask evaluate() for the test split.
"""

import importlib
import importlib.util
from pathlib import Path
from typing import Protocol

from gotrain.config import get_settings
from gotrain.contracts import BandStats, DatasetCard, TrainConfig, TrainResult

REAL_MODULES = ("gotrain.engine.train", "gotrain.engine.data")


class Engine(Protocol):
    def train(self, config: TrainConfig, budget_minutes: float, run_dir: Path) -> TrainResult: ...

    def evaluate(self, task: str, checkpoint_path: Path, split: str) -> dict: ...

    def describe_dataset(self, task: str) -> DatasetCard: ...


class RealEngine:
    """Thin adapter over Serena's modules; nothing heavy is imported until a call."""

    def train(self, config: TrainConfig, budget_minutes: float, run_dir: Path) -> TrainResult:
        return importlib.import_module("gotrain.engine.train").train(
            config, budget_minutes, run_dir
        )

    def evaluate(self, task: str, checkpoint_path: Path, split: str) -> dict:
        return importlib.import_module("gotrain.engine.train").evaluate(
            task, checkpoint_path, split
        )

    def describe_dataset(self, task: str) -> DatasetCard:
        return importlib.import_module("gotrain.engine.data").describe_dataset(task)


def get_engine() -> Engine:
    """Return the engine selected by GOTRAIN_ENGINE ("fake" by default)."""
    if get_settings().engine == "fake":
        from gotrain.engine.fake import FakeEngine

        return FakeEngine()
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
