"""Shared helpers for the slide figures. CPU only, TRAIN split only."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gotrain.engine.data import GeoBenchDataset  # noqa: E402

# Reference palette (dataviz skill, light mode) + text tokens
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#6250d6", "#e34948"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e4e3df"
NEUTRAL = "#b9b8b1"

RGB_BANDS = {
    "m-eurosat": ("B04", "B03", "B02"),
    "m-brick-kiln": ("B04", "B03", "B02"),
    "m-pv4ger-seg": ("Red", "Green", "Blue"),
    "m-chesapeake": ("Red", "Green", "Blue"),
}

# Chesapeake classes -> semantic colours drawn from the same palette
CHESA_COLORS = {
    "no data": "#d9d8d3", "water": BLUE, "tree canopy": GREEN, "low vegetation": YELLOW,
    "barren": ORANGE, "impervious other": VIOLET, "impervious roads": MAGENTA,
}
PANEL_COLOR = RED

_DS: dict = {}


def dataset(task: str) -> GeoBenchDataset:
    if task not in _DS:
        _DS[task] = GeoBenchDataset(task, "train", norm="none")  # TRAIN only, raw units
    return _DS[task]


def stretch(img: np.ndarray) -> np.ndarray:
    """Per-channel 2-98% stretch (same as scratch/explore.py)."""
    out = np.zeros(img.shape, dtype=np.float64)
    for c in range(img.shape[-1]):
        ch = img[..., c].astype(np.float64)
        lo, hi = np.nanpercentile(ch, 2), np.nanpercentile(ch, 98)
        out[..., c] = np.clip((ch - lo) / (hi - lo if hi > lo else 1.0), 0, 1)
    return np.nan_to_num(out)


def tile(task: str, i: int) -> tuple[np.ndarray, np.ndarray]:
    """(stretched RGB HxWx3, label) for train tile i."""
    ds = dataset(task)
    x, y = ds[int(i)]
    x = x.numpy()
    bands = list(ds.all_bands)
    rgb = np.stack([x[bands.index(b)] for b in RGB_BANDS[task]], axis=-1)
    return stretch(rgb), np.asarray(y)


def setup_fonts() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 20, "text.color": INK,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def slide(title: str, subtitle: str | None = None):
    """A 1920x1080 white figure with a large title."""
    setup_fonts()
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.text(0.035, 0.935, title, fontsize=44, fontweight="bold", color=INK, va="center")
    if subtitle:
        fig.text(0.035, 0.875, subtitle, fontsize=22, color=INK2, va="center")
    return fig
