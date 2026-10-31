"""A fake engine: no GPU, no data, no torch. Plausible, deterministic numbers.

Lets the agent, tools, registry and logging be built and tested before the real
engine exists. Same config (ignoring `reason`) -> same result, every time.
"""

import hashlib
import json
import math
import random
import time
from pathlib import Path

from gotrain.contracts import DatasetCard, EpochRecord, TrainConfig, TrainResult

S2_BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
]
S2_MEAN = [
    1354.0,
    1118.0,
    1042.0,
    947.0,
    1199.0,
    2003.0,
    2374.0,
    2301.0,
    2599.0,
    732.0,
    12.0,
    1820.0,
    1118.0,
]
S2_STD = [
    245.0,
    333.0,
    395.0,
    594.0,
    566.0,
    861.0,
    1086.0,
    1118.0,
    1231.0,
    404.0,
    4.0,
    1003.0,
    761.0,
]
EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]
CHESAPEAKE_CLASSES = [
    "water",
    "tree_canopy",
    "low_vegetation",
    "barren",
    "impervious_other",
    "impervious_road",
    "no_data",
]

CARDS: dict[str, DatasetCard] = {
    "m-eurosat": DatasetCard(
        task="m-eurosat",
        task_type="classification",
        sensor="sentinel-2",
        bands=S2_BANDS,
        n_classes=10,
        class_names=EUROSAT_CLASSES,
        image_size=64,
        split_sizes={"train": 2000, "val": 1000},
        class_distribution={c: 0.1 for c in EUROSAT_CLASSES},
        band_mean=S2_MEAN,
        band_std=S2_STD,
        dtype="float32",
        label_format="int",
        notes="FAKE card. Balanced 10-class land cover.",
    ),
    "m-brick-kiln": DatasetCard(
        task="m-brick-kiln",
        task_type="classification",
        sensor="sentinel-2",
        bands=S2_BANDS,
        n_classes=2,
        class_names=["no_kiln", "kiln"],
        image_size=64,
        split_sizes={"train": 15063, "val": 999},
        class_distribution={"no_kiln": 0.5, "kiln": 0.5},
        band_mean=S2_MEAN,
        band_std=S2_STD,
        dtype="float32",
        label_format="int",
        notes="FAKE card. Binary brick kiln detection.",
    ),
    "m-pv4ger-seg": DatasetCard(
        task="m-pv4ger-seg",
        task_type="segmentation",
        sensor="aerial-rgb",
        bands=["red", "green", "blue"],
        n_classes=2,
        class_names=["background", "solar_panel"],
        image_size=320,
        split_sizes={"train": 3000, "val": 403},
        class_distribution={"background": 0.97, "solar_panel": 0.03},
        band_mean=[120.0, 118.0, 110.0],
        band_std=[52.0, 48.0, 46.0],
        dtype="uint8",
        value_range=(0, 255),
        label_format="mask",
        mask_unique_values=[0, 1],
        notes="FAKE card. Heavy class imbalance: panels are ~3% of pixels.",
    ),
    "m-chesapeake-landcover": DatasetCard(
        task="m-chesapeake-landcover",
        task_type="segmentation",
        sensor="naip",
        bands=["red", "green", "blue", "nir"],
        n_classes=7,
        class_names=CHESAPEAKE_CLASSES,
        image_size=256,
        split_sizes={"train": 3000, "val": 1000},
        class_distribution=dict(
            zip(CHESAPEAKE_CLASSES, [0.12, 0.45, 0.25, 0.01, 0.08, 0.05, 0.04], strict=True)
        ),
        band_mean=[90.0, 105.0, 95.0, 130.0],
        band_std=[40.0, 35.0, 30.0, 45.0],
        dtype="uint8",
        value_range=(0, 255),
        label_format="mask",
        mask_unique_values=list(range(7)),
        notes="FAKE card. 7-class land cover, rare 'barren' class.",
    ),
}

BASE_SCORE = {  # val score of a well-tuned imagenet run
    "m-eurosat": 0.90,
    "m-brick-kiln": 0.94,
    "m-pv4ger-seg": 0.82,
    "m-chesapeake-landcover": 0.60,
}
WEIGHTS_BONUS = {"none": -0.08, "imagenet": 0.0}
AUGMENT_BONUS = {"rot90": 0.012, "flip": 0.008, "color_jitter": -0.004}
SETUP_MINUTES = 0.2
BEST_LOG10_LR = -3.3


def config_seed(config: TrainConfig) -> int:
    """Stable seed from everything that affects training (not the free-text reason)."""
    payload = config.model_dump_json(exclude={"reason"})
    return int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16)


def metric_for(card: DatasetCard) -> str:
    return {"classification": "accuracy", "segmentation": "miou", "multilabel": "f1"}[
        card.task_type
    ]


def target_score(config: TrainConfig, card: DatasetCard, rng: random.Random) -> float:
    """The score this config converges to with enough epochs."""
    s2 = card.sensor == "sentinel-2"
    score = BASE_SCORE[card.task]
    score += (
        0.03 if config.weights == "ssl4eo_s2" and s2 else WEIGHTS_BONUS.get(config.weights, -0.02)
    )
    score += sum(AUGMENT_BONUS[a] for a in config.augment)
    score -= min(0.3, 0.04 * (math.log10(config.lr) - BEST_LOG10_LR) ** 2)
    score -= 0.06 * (1.0 - config.train_subset_frac)
    if config.bands != "all" and len(config.bands) < 3:
        score -= 0.03
    return score + rng.uniform(-0.01, 0.01)


def minutes_per_epoch(config: TrainConfig, card: DatasetCard) -> float:
    pixels = (config.img_size / 64) ** 2
    return card.split_sizes["train"] * config.train_subset_frac * pixels / 20000


def _clip(x: float) -> float:
    return min(0.995, max(0.0, x))


def make_history(
    target: float, epochs: int, per_epoch_min: float, rng: random.Random
) -> list[EpochRecord]:
    history = []
    for e in range(1, epochs + 1):
        val = target * (1 - 0.35 * math.exp(-e / 8)) + rng.gauss(0, 0.004)
        loss = 0.15 + 1.6 * math.exp(-e / 5) + abs(rng.gauss(0, 0.02))
        elapsed = (SETUP_MINUTES + e * per_epoch_min) * 60
        history.append(
            EpochRecord(
                epoch=e,
                train_loss=round(loss, 4),
                val_score=round(_clip(val), 4),
                elapsed_s=round(elapsed, 1),
            )
        )
    return history


class FakeEngine:
    def __init__(self, sleep_s: float = 1.0) -> None:
        self.sleep_s = sleep_s

    def train(self, config: TrainConfig, budget_minutes: float, run_dir: Path) -> TrainResult:
        time.sleep(self.sleep_s)
        run_id = run_dir.name
        card = CARDS.get(config.task)
        problem = self._config_problem(config, card)
        if problem:
            return TrainResult(
                run_id=run_id,
                task=config.task,
                status="error",
                metric_name="unknown",
                minutes_used=0.0,
                error=problem,
            )

        rng = random.Random(config_seed(config))
        per_epoch = minutes_per_epoch(config, card)
        epochs = min(config.max_epochs, int((budget_minutes - SETUP_MINUTES) / per_epoch))
        if epochs < 1:
            return TrainResult(
                run_id=run_id,
                task=config.task,
                status="timeout",
                metric_name=metric_for(card),
                minutes_used=budget_minutes,
                error="budget too small for one epoch",
            )

        history = make_history(target_score(config, card, rng), epochs, per_epoch, rng)
        best = max(h.val_score for h in history)
        checkpoint = self._write_checkpoint(run_dir, config, best)
        return TrainResult(
            run_id=run_id,
            task=config.task,
            status="ok",
            metric_name=metric_for(card),
            val_score=best,
            history=history,
            minutes_used=round(SETUP_MINUTES + epochs * per_epoch, 3),
            checkpoint_path=checkpoint,
        )

    def evaluate(self, task: str, checkpoint_path: Path, split: str) -> dict:
        if split not in ("val", "test"):
            raise ValueError(f"split must be 'val' or 'test', got {split!r}")
        ckpt = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
        if ckpt["task"] != task:
            raise ValueError(f"checkpoint is for {ckpt['task']!r}, not {task!r}")
        score = ckpt["val_score"]
        if split != "val":
            score = round(score - random.Random(ckpt["seed"]).uniform(0.0, 0.02), 4)
        return {"task": task, "split": split, "metric_name": ckpt["metric_name"], "score": score}

    def describe_dataset(self, task: str) -> DatasetCard:
        if task not in CARDS:
            raise ValueError(f"unknown task {task!r}; known: {sorted(CARDS)}")
        return CARDS[task].model_copy(deep=True)

    @staticmethod
    def _config_problem(config: TrainConfig, card: DatasetCard | None) -> str | None:
        if card is None:
            return f"unknown task {config.task!r}; known: {sorted(CARDS)}"
        if config.bands != "all":
            unknown = [b for b in config.bands if b not in card.bands]
            if unknown:
                return f"bands {unknown} not in {config.task}; available: {card.bands}"
        return None

    @staticmethod
    def _write_checkpoint(run_dir: Path, config: TrainConfig, val_score: float) -> Path:
        path = run_dir / "checkpoints" / "best.ckpt"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fake": True,
            "task": config.task,
            "seed": config_seed(config),
            "metric_name": metric_for(CARDS[config.task]),
            "val_score": val_score,
            "config": config.model_dump(mode="json"),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
