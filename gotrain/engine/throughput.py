"""A deliberately simple model of training time, from the pixels seen per epoch.

    minutes_per_epoch ~= train_samples * train_subset_frac * height * width / PIXELS_PER_MINUTE

Bands, backbone and batch size are ignored. PIXELS_PER_MINUTE is exactly the fake engine's
timing (it computes epoch time with this function), so hints match dev runs. For the real
engine it is a placeholder until measured on the T4. Once a task has trials, their observed
`minutes_per_epoch` is always the better number.
"""

import math

from gotrain.contracts import DatasetCard

PIXELS_PER_MINUTE = 20_000 * 64 * 64  # ~82 Mpx/min: 20k 64x64 images per minute
SCREEN_MINUTES = 2.0  # a typical rung-1 short_minutes
SCREEN_EPOCHS = 3  # a screening trial needs at least this many epochs to rank configs


def estimate_minutes_per_epoch(
    train_samples: int, height: int, width: int, frac: float = 1.0
) -> float:
    return train_samples * frac * height * width / PIXELS_PER_MINUTE


def card_minutes_per_epoch(card: DatasetCard) -> float:
    """Estimated full-data epoch time at the dataset's native image size."""
    height, width = card.image_size
    return estimate_minutes_per_epoch(card.split_sizes["train"], height, width)


def subset_for_epochs(full_epoch_min: float, minutes: float, epochs: int = SCREEN_EPOCHS) -> float:
    """The largest train_subset_frac (in steps of 0.05, at least 0.05) fitting `epochs` epochs."""
    frac = math.floor(minutes / (epochs * full_epoch_min) * 20) / 20
    return min(1.0, max(0.05, frac))


def screening_advice(card: DatasetCard) -> str | None:
    """A hint when SCREEN_EPOCHS full-data epochs don't fit in a SCREEN_MINUTES trial."""
    per_epoch = card_minutes_per_epoch(card)
    if SCREEN_EPOCHS * per_epoch <= SCREEN_MINUTES:
        return None
    height, width = card.image_size
    frac = subset_for_epochs(per_epoch, SCREEN_MINUTES)
    return (
        f"One full-data epoch ({card.split_sizes['train']} samples at {height}x{width} px) is "
        f"estimated at ~{per_epoch:.1f} min. For {SCREEN_EPOCHS} epochs in a "
        f"{SCREEN_MINUTES:g}-min screening trial use train_subset_frac <= {frac:g}, or "
        f"short_minutes >= {SCREEN_EPOCHS * per_epoch:.1f}."
    )
