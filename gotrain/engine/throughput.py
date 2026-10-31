"""A deliberately simple model of training time per epoch.

    seconds per image = height * width / pixels_per_second + seconds_per_image (overhead)
    (segmentation tasks use their own pixels_per_second: a U-Net decoder costs extra)
    minutes per epoch = (train_samples * train_subset_frac + val_samples) * minutes per image

Bands, backbone and batch size are ignored. Every epoch also runs the full validation set,
so val images count (at the same per-image cost: a rough but measured approximation).

REAL_T4 is calibrated on the SageMaker T4 (m-eurosat, resnet50, batch 64, fp16, 2,000 train
+ 1,000 val images): ~1.9 s per epoch at 64 px and ~10 s at 224 px. One pixels-per-minute
constant can't fit both (that would be ~388 Mpx/min at 64 px vs ~903 at 224 px): small
images are dominated by a fixed per-image cost. The two terms below fit both exactly:
    10.0 - 1.9 = 3,000 x (224^2 - 64^2) px / pixels_per_second  -> 1,024 Mpx/min
    1.9 = 3,000 x (64^2 / pixels_per_second + overhead)          -> 0.393 ms per image
Segmentation is measured on m-pv4ger-seg (U-Net resnet34, batch 16, 3,000 train + 403 val
images at 320 px): ~34 s per epoch. The classification rate predicts only ~21.8 s, so
segmentation gets its own pixel rate, keeping the same per-image overhead:
    34 = 3,403 x (320^2 / seg_pixels_per_second + 0.393 ms)        -> ~640 Mpx/min
One measurement, one task: m-chesapeake (256 px, 4,000 images) is extrapolated (~26 s).
**Update REAL_T4 here when new measurements arrive.**

The fake engine has its own, slower FAKE_TIMING (gotrain/engine/fake.py), and hints use
whichever engine is active, so dev-run hints match what the fake engine does. Once a task
has trials, their observed `minutes_per_epoch` beats any estimate.
"""

import math
from dataclasses import dataclass

from gotrain.config import get_settings
from gotrain.contracts import DatasetCard

SCREEN_MINUTES = 2.0  # a typical rung-1 short_minutes
SCREEN_EPOCHS = 3  # a screening trial needs at least this many epochs to rank configs


@dataclass(frozen=True)
class TimingModel:
    name: str
    pixels_per_minute: float
    seconds_per_image: float = 0.0
    counts_val: bool = True  # whether each epoch's validation pass is part of the time
    seg_pixels_per_minute: float | None = None  # None: same rate as classification

    def minutes_per_image(self, height: int, width: int, segmentation: bool = False) -> float:
        rate = self.pixels_per_minute
        if segmentation and self.seg_pixels_per_minute:
            rate = self.seg_pixels_per_minute
        return height * width / rate + self.seconds_per_image / 60


REAL_T4 = TimingModel(
    name="real T4 (calibrated on m-eurosat resnet50 and m-pv4ger-seg U-Net resnet34, fp16)",
    pixels_per_minute=1_024e6,
    seconds_per_image=3.93e-4,
    counts_val=True,
    seg_pixels_per_minute=640.2e6,
)


def active_timing() -> TimingModel:
    """The timing model of the engine in use (GOTRAIN_ENGINE)."""
    if get_settings().engine == "fake":
        from gotrain.engine.fake import FAKE_TIMING  # late import: fake imports this module

        return FAKE_TIMING
    return REAL_T4


def estimate_minutes_per_epoch(
    train_samples: int,
    height: int,
    width: int,
    frac: float = 1.0,
    val_samples: int = 0,
    timing: TimingModel = REAL_T4,
    segmentation: bool = False,
) -> float:
    images = train_samples * frac + (val_samples if timing.counts_val else 0)
    return images * timing.minutes_per_image(height, width, segmentation)


def card_minutes_per_epoch(
    card: DatasetCard, side: int | None = None, frac: float = 1.0, timing: TimingModel | None = None
) -> float:
    """Estimated epoch time for this task, at `side` px (default: the native image size)."""
    timing = timing or active_timing()
    height, width = (side, side) if side else card.image_size
    split = card.split_sizes
    seg = card.task_type == "segmentation"
    return estimate_minutes_per_epoch(
        split["train"], height, width, frac, split.get("val", 0), timing, seg
    )


def subset_for_epochs(
    card: DatasetCard,
    minutes: float,
    side: int | None = None,
    epochs: int = SCREEN_EPOCHS,
    timing: TimingModel | None = None,
) -> float | None:
    """The largest train_subset_frac (steps of 0.05, at least 0.05) that fits `epochs` epochs
    in `minutes`; None if even the validation passes alone don't fit."""
    timing = timing or active_timing()
    height, width = (side, side) if side else card.image_size
    per_image = timing.minutes_per_image(height, width, card.task_type == "segmentation")
    val_minutes = card.split_sizes.get("val", 0) * per_image if timing.counts_val else 0.0
    train_minutes = minutes / epochs - val_minutes
    if train_minutes <= 0:
        return None
    frac = math.floor(train_minutes / (card.split_sizes["train"] * per_image) * 20) / 20
    return min(1.0, max(0.05, frac))


def screening_advice(card: DatasetCard, timing: TimingModel | None = None) -> str | None:
    """A hint when SCREEN_EPOCHS full-data epochs don't fit in a SCREEN_MINUTES trial."""
    timing = timing or active_timing()
    per_epoch = card_minutes_per_epoch(card, timing=timing)
    if SCREEN_EPOCHS * per_epoch <= SCREEN_MINUTES:
        return None
    height, width = card.image_size
    frac = subset_for_epochs(card, SCREEN_MINUTES, timing=timing)
    subset = f"use train_subset_frac <= {frac:g}, or " if frac is not None else ""
    return (
        f"One full-data epoch ({card.split_sizes['train']} samples at {height}x{width} px) is "
        f"estimated at ~{per_epoch:.1f} min ({timing.name}). For {SCREEN_EPOCHS} epochs in a "
        f"{SCREEN_MINUTES:g}-min screening trial {subset}"
        f"short_minutes >= {SCREEN_EPOCHS * per_epoch:.1f}."
    )
