"""GEO-Bench data access for the real engine: dataset cards and a PyTorch dataset.

Three layers, top to bottom:

1. Shared helpers: locate a task on disk, open a split (the test split is refused unless
   evaluate() explicitly allows it), pack a sample into an (H, W, C) array, and apply the
   upstream errata tables (BAND_ORDER_FIXES, CLASS_NAME_FIXES).
2. Recon: `inspect_task()` measures a task on TRAIN/VAL only and writes stats/<task>.json
   (normalisation stats) and stats/<task>_card.json (the full recon card).
   `describe_dataset()` turns that recon card into a contract-valid `DatasetCard`.
3. `GeoBenchDataset`: a map-style dataset for torch's DataLoader. It selects bands by name,
   normalises them, and returns channels-first float32 tensors.

torch and geobench are imported lazily, so this module (and describe_dataset on a cached
card) works on a machine without the ML extras.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from gotrain.config import REPO_ROOT, get_settings
from gotrain.contracts import DatasetCard

# Engine split names -> geobench split names. The test split is never opened unless the caller
# passes allow_test=True, and only train.evaluate() does that.
TEST_SPLIT = "test"
GEOBENCH_SPLITS = {"train": "train", "val": "valid", TEST_SPLIT: TEST_SPLIT}
IGNORE_INDEX = -100  # mask pixels outside [0, n_classes); torch's cross-entropy default
NODATA_RULE = "pixels where every band is exactly 0, plus NaN/inf values"
NEARLY_EMPTY_RATIO = 0.01  # a band whose p99 is <1% of the largest band's p99
# Flat (low-contrast) tiles. A tile's contrast is the MEDIAN over bands of
# (p98 - p2 of the tile's valid pixels in that band) / (that band's TRAIN std): unit-free, so it
# works for reflectance, 8-bit and 0-1 data, and one odd band (e.g. near-empty B10) can't decide.
# Chosen from data: m-eurosat's contrast is bimodal (5% of tiles < 0.06, 90% > 0.5); 0.25 sits in
# the gap. For brick-kiln's B04 it is ~90 DN, above the 50 DN that undercounted (gotcha 23).
FLAT_THRESHOLD = 0.25
FLAT_DEFINITION = (
    "a tile is nearly flat when the median over bands of (p98 - p2 of its valid pixels in that "
    f"band) / (that band's train std) is below {FLAT_THRESHOLD}"
)

# Upstream errata: geo-bench README "Known issues", issues #28 and #29. The pixel data is fine,
# but the stored band names are wrong for most channels. These are the TRUE names, in order.
BAND_ORDER_FIXES: dict[str, list[str]] = {
    "m-eurosat": [
        "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A",
    ],
    "m-brick-kiln": [
        "B01", "B02", "B03", "B04", "B05", "B07", "B8A", "B08", "B11", "B12",
        "TCI_R", "TCI_G", "TCI_B",
    ],
}  # fmt: skip
ERRATA_URL = "https://github.com/ServiceNow/geo-bench#known-issues"

# Upstream errata (found by us, 2026-10-31): m-eurosat's label integers follow the original
# alphabetical EuroSAT order, but geobench's class_names list is in a different order. Evidence,
# train split only: docs/figures/m-eurosat_per_label.png (6 tiles per label index) and mean NDVI
# per label (label 9 = -0.25, i.e. water; label 1 = +0.71, i.e. forest). Same strings as
# geobench, reordered. See docs/data_notes.md gotcha 14.
CLASS_NAME_FIXES: dict[str, list[str]] = {
    "m-eurosat": [
        "Annual Crop", "Forest", "Herbaceous Vegetation", "Highway", "Industrial Buildings",
        "Pasture", "Permanent Crop", "Residential Buildings", "River", "Sea and Lake",
    ],
    # m-chesapeake (found by us, 2026-10-31): the task specs store only "0".."6". The masks are
    # TorchGeo's raw Chesapeake land-cover values (GEO-Bench's converter reads them through
    # ChesapeakeCVPRDataModule(class_set=7), which does not remap). TorchGeo draws 0 as
    # transparent and fills out-of-raster areas with 0: class 0 is exactly 125 whole blank
    # 256x256 train tiles. GEO-Bench's converter lists the names starting at "water" with
    # "no-data" LAST, which is off by one. Evidence: docs/figures/m-chesapeake_samples.png
    # (tile #49, rippled water, is class 1; 2-6 match canopy, fields, barren, roofs, roads).
    # See docs/data_notes.md gotcha 22.
    "m-chesapeake": [
        "no data", "water", "tree canopy", "low vegetation", "barren",
        "impervious other", "impervious roads",
    ],
}  # fmt: skip

# Facts measured by one-off checks outside inspect_task (data, not logic). describe_dataset()
# adds them to the card's notes so they survive a re-run of scratch/explore.py.
# Segmentation classes that mean "no data", not land cover (data, not logic). Training and the
# official `score` keep ALL classes (comparable with GEO-Bench baselines; decision on
# docs/contract-requests.md #9); metrics.json and diagnose() add `miou_excl_nodata` as an extra.
NODATA_CLASSES: dict[str, int] = {"m-chesapeake": 0}

DATASET_FACTS: dict[str, list[str]] = {
    "m-brick-kiln": [
        "Kilns are about 100-150 m across (10-15 px of a 64 px tile). Mean NDVI is 0.40 for "
        "'not brick kiln' and 0.38 for 'brick kiln' tiles.",
    ],
    "m-chesapeake": [
        "Class 0 'no data' marks blank tiles (125 whole train tiles; the imagery there is all "
        "zero). It stays in training and in the official 7-class mIoU (score, comparable with "
        "GEO-Bench); metrics.json and diagnose() also report miou_excl_nodata (6 classes).",
        "Some masks have hand-drawn straight edges that do not follow the image (e.g. a barren "
        "polygon over grass in train tile #808).",
    ],
}

# Standard Sentinel-2 MSI native resolutions, used instead of the stored metadata when band
# names are corrected (the stored resolutions belong to the mislabelled names).
# The 13 Sentinel-2 bands in the order TorchGeo's SSL4EO weights expect (B1..B8, B8a, B9..B12;
# verified in torchgeo 0.10.1), spelled the GEO-Bench way. Bands a task lacks are zero-filled.
SSL4EO_S2_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12",
]  # fmt: skip

S2_NATIVE_RES_M = {
    **dict.fromkeys(["B02", "B03", "B04", "B08"], 10),
    **dict.fromkeys(["B05", "B06", "B07", "B8A", "B11", "B12"], 20),
    **dict.fromkeys(["B01", "B09", "B10"], 60),
}

# geobench band-info class -> sensor slug used in cards and memory lessons.
SENSOR_SLUGS = {
    "Sentinel2": "sentinel-2",
    "Sentinel1": "sentinel-1",
    "Landsat8": "landsat-8",
    "HyperSpectralBands": "hyperspectral",
    "ElevationBand": "elevation",
    "CloudProbability": "cloud-probability",
}
AERIAL_LETTERS = {
    "red": "r", "green": "g", "blue": "b", "nir": "n", "near-infrared": "n", "nearinfrared": "n",
}  # fmt: skip


# ---------------------------------------------------------------------------------------------
# 1. Shared helpers
# ---------------------------------------------------------------------------------------------
def find_task_dir(task_name: str, data_dir: str | Path | None = None) -> Path:
    """Locate <root>/<benchmark>/<task_name>/task_specs.pkl without import-time env tricks."""
    roots = [Path(data_dir)] if data_dir else []
    # Read GEO_BENCH_DIR directly: locating data must not need the agent's GOTRAIN_HOME.
    if os.environ.get("GEO_BENCH_DIR"):
        roots.append(Path(os.environ["GEO_BENCH_DIR"]).expanduser())
    roots += [REPO_ROOT / "data", Path.home() / "dataset" / "geobench"]
    for root in roots:
        for cand in sorted(root.glob(f"*/{task_name}")):
            if (cand / "task_specs.pkl").exists():
                return cand
    raise FileNotFoundError(
        f"task {task_name!r} not found under {[str(r) for r in roots]}. "
        "Download it with scratch/download_tasks.py or set GEO_BENCH_DIR."
    )


def load_task(task_name: str, data_dir: str | Path | None = None) -> Any:
    task_dir = find_task_dir(task_name, data_dir)  # a clear FileNotFoundError first
    import geobench

    return geobench.load_task_specs(task_dir)


def open_split(task: Any, split: str, *, allow_test: bool = False) -> Any:
    """Open 'train' or 'val' (engine names). The test split needs allow_test=True."""
    if split not in GEOBENCH_SPLITS:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    if split == TEST_SPLIT and not allow_test:
        raise PermissionError(
            "the test split is sacred: only evaluate() (called by final_test) may open it"
        )
    # Pass ALL band names: get_dataset() defaults to ("red", "green", "blue"), which silently
    # selects 3 bands (and crashes on datasets without RGB).
    band_names = [bi.name for bi in task.bands_info]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the band-order warning; handled by BAND_ORDER_FIXES
        return task.get_dataset(split=GEOBENCH_SPLITS[split], band_names=band_names)


def sample_array(sample: Any) -> tuple[np.ndarray, list[str], bool]:
    """(H, W, C) array, raw channel names, and whether any band had to be resampled."""
    names = [bi.name for bi in sample.band_info_list]
    shapes = {b.data.shape[:2] for b in sample.bands}
    arr, ch_names = sample.pack_to_3d(band_names=names, resample=True)
    return arr, list(ch_names), len(shapes) > 1


def channel_names(task_name: str, raw_names: list[str]) -> tuple[list[str], bool]:
    """True channel names (errata applied) and whether a fix was applied."""
    fix = BAND_ORDER_FIXES.get(task_name)
    if fix is not None and len(fix) == len(raw_names):
        return list(fix), True
    return list(raw_names), False


def raw_class_names(task: Any, n_classes: int) -> list[str]:
    lt = task.label_type
    # MultiLabelClassification stores its names in `class_name` (upstream typo).
    names = getattr(lt, "class_names", None) or getattr(lt, "class_name", None)
    return list(names) if names else [str(i) for i in range(n_classes)]


def corrected_class_names(task_name: str, raw_names: list[str]) -> tuple[list[str], bool]:
    """Class names indexed by label integer (errata applied) and whether a fix was applied."""
    fix = CLASS_NAME_FIXES.get(task_name)
    if fix is not None and len(fix) == len(raw_names):
        return list(fix), True
    return list(raw_names), False


def class_names_for(task_name: str, task: Any) -> list[str]:
    """The ONLY way to map label integers to names: never use label_type.class_names directly."""
    n = int(task.label_type.n_classes)
    return corrected_class_names(task_name, raw_class_names(task, n))[0]


def task_type(task: Any) -> str:
    kind = type(task.label_type).__name__
    if "MultiLabel" in kind:
        return "multilabel"
    if "Segmentation" in kind:
        return "segmentation"
    if "Classification" in kind:
        return "classification"
    return kind.lower()


def sensor(task: Any) -> str:
    """Sensor slug, e.g. 'sentinel-2' or 'aerial-rgb' (used to match memory lessons).

    Any task with Sentinel-2 bands is 'sentinel-2', even if it also carries derived channels
    (m-brick-kiln's 8-bit TCI true-colour bands): combos.check_combo only allows ssl4eo_s2
    weights on cards whose sensor normalises to 'sentinel2'.
    """
    if any(type(bi).__name__ == "Sentinel2" for bi in task.bands_info):
        return "sentinel-2"
    kinds: list[str] = []
    for bi in task.bands_info:
        kind = type(bi).__name__
        if kind == "SpectralBand":
            names = [bi2.name.strip().lower() for bi2 in task.bands_info]
            letters = [AERIAL_LETTERS.get(n) for n in names]
            slug = "aerial-" + ("".join(letters) if all(letters) else f"{len(names)}band")
        else:
            slug = SENSOR_SLUGS.get(kind, kind.lower())
        if slug not in kinds:
            kinds.append(slug)
    return "+".join(kinds)


def label_value(label: Any) -> Any:
    """Band -> its pixel array; anything else (int, vector) unchanged."""
    return label.data if hasattr(label, "data") else label


def labels_only(ds: Any, idx: int) -> Any:
    """Read only the label of sample idx (fast path for hdf5), falling back to a full load."""
    from geobench.dataset import load_sample_hdf5

    name = ds.active_partition.partition_dict[ds.split][idx]
    path = Path(ds.dataset_dir) / f"{name}.hdf5"
    if path.exists():
        return label_value(load_sample_hdf5(path, label_only=True).label)
    return label_value(ds[idx].label)


# ---------------------------------------------------------------------------------------------
# 2a. Streaming statistics (Chan et al. parallel merge of per-image mean / M2, in float64)
# ---------------------------------------------------------------------------------------------
class RunningBandStats:
    def __init__(self, n_bands: int, pixels_per_image_for_percentiles: int = 256, seed: int = 0):
        self.n = np.zeros(n_bands)
        self.mean = np.zeros(n_bands)
        self.m2 = np.zeros(n_bands)
        self.min = np.full(n_bands, np.inf)
        self.max = np.full(n_bands, -np.inf)
        self.reservoir: list[np.ndarray] = []
        self.k = pixels_per_image_for_percentiles
        self.rng = np.random.default_rng(seed)
        self.total_pixels = 0
        self.nodata_pixels = 0
        self.nonfinite_values = 0
        self.excluded_pixels = 0  # nodata pixels + pixels with any NaN/inf value
        self.constant_in_image = np.zeros(n_bands, dtype=int)
        self.images = 0

    def update(self, arr_hwc: np.ndarray) -> None:
        x = arr_hwc.reshape(-1, arr_hwc.shape[-1]).astype(np.float64)
        self.images += 1
        self.total_pixels += x.shape[0]
        nonfinite = ~np.isfinite(x)
        self.nonfinite_values += int(nonfinite.sum())
        x[nonfinite] = np.nan
        nodata = np.all(x == 0, axis=1)
        self.nodata_pixels += int(nodata.sum())
        x[nodata] = np.nan
        self.excluded_pixels += int((nodata | nonfinite.any(axis=1)).sum())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            nb = np.sum(np.isfinite(x), axis=0).astype(float)
            mb = np.nanmean(x, axis=0)
            m2b = np.nansum((x - mb) ** 2, axis=0)
            self.min = np.fmin(self.min, np.nanmin(x, axis=0))
            self.max = np.fmax(self.max, np.nanmax(x, axis=0))
            self.constant_in_image += (np.nanmax(x, axis=0) == np.nanmin(x, axis=0)).astype(int)
        ok = nb > 0
        delta = np.where(ok, mb - self.mean, 0.0)
        tot = self.n + nb
        safe_tot = np.where(tot > 0, tot, 1)
        self.mean = np.where(ok, self.mean + delta * nb / safe_tot, self.mean)
        self.m2 = np.where(
            ok, self.m2 + np.nan_to_num(m2b) + delta**2 * self.n * nb / safe_tot, self.m2
        )
        self.n = tot
        pick = self.rng.choice(x.shape[0], size=min(self.k, x.shape[0]), replace=False)
        self.reservoir.append(x[pick])

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.m2 / np.maximum(self.n, 1))

    def percentiles(self, q: float) -> np.ndarray:
        r = np.concatenate(self.reservoir, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanpercentile(r, q, axis=0)


# ---------------------------------------------------------------------------------------------
# 2b. Recon: measure a task (TRAIN/VAL only) and write stats/<task>.json + <task>_card.json
# ---------------------------------------------------------------------------------------------
def _label_counts(ds_train: Any, kind: str) -> Counter:
    counts: Counter = Counter()
    for i in range(len(ds_train)):
        lab = labels_only(ds_train, i)
        if kind == "segmentation":
            vals, cnt = np.unique(np.asarray(lab), return_counts=True)
            counts.update({int(v): int(c) for v, c in zip(vals, cnt, strict=True)})
        elif kind == "multilabel":
            for c in np.flatnonzero(np.asarray(lab)):
                counts[int(c)] += 1
        else:
            counts[int(lab)] += 1
    return counts


def _mask_summary(counts: Counter, n_classes: int, notes: list[str]) -> dict:
    values = sorted(counts)
    ignore = [v for v in values if v < 0 or v >= n_classes]
    summary = {
        "values": {str(v): counts[v] for v in values},
        "is_0_1": set(values) <= {0, 1},
        "is_0_255": 255 in values and set(values) <= {0, 255},
        "ignore_values": ignore,
    }
    if summary["is_0_255"]:
        notes.append("Masks use 0/255, not 0/1: divide by 255 (or map 255->1) before the loss.")
    elif ignore:
        notes.append(f"Mask values outside [0, {n_classes - 1}]: {ignore}. Treat as ignore_index.")
    return summary


def _distribution_notes(dist: dict, kind: str, notes: list[str]) -> None:
    unit = {"segmentation": "of pixels", "multilabel": "of images"}.get(kind, "of samples")
    pcts = {k: v["pct"] for k, v in dist.items()}
    for name, p in pcts.items():
        if dist[name]["count"] == 0:
            notes.append(f"class '{name}' never appears in train: the model cannot learn it.")
        elif p < 1:
            notes.append(f"severe imbalance: class '{name}' is {p:.2f}% {unit}")
        elif p < 5:
            notes.append(f"imbalance: class '{name}' is {p:.2f}% {unit}")
    nonzero = [p for p in pcts.values() if p > 0]
    if len(nonzero) >= 2 and max(nonzero) / min(nonzero) > 10 and kind != "multilabel":
        notes.append(
            f"largest/smallest class ratio is {max(nonzero) / min(nonzero):.0f}x: "
            "use class weights or Dice/focal loss, and report per-class metrics."
        )


def _scale_note(bands: list[str], p99: np.ndarray, notes: list[str]) -> None:
    usable = [i for i in range(len(bands)) if np.isfinite(p99[i]) and p99[i] > 0]
    if len(usable) < 2:
        return
    lo = min(usable, key=lambda i: p99[i])
    hi = max(usable, key=lambda i: p99[i])
    if p99[hi] / p99[lo] > 20:
        notes.append(
            f"band value scales differ by ~{p99[hi] / p99[lo]:.0f}x (p99: {bands[lo]} "
            f"{p99[lo]:g} vs {bands[hi]} {p99[hi]:g}): per-band normalisation is mandatory."
        )


def tile_spread(arr_hwc: np.ndarray) -> np.ndarray:
    """Per-band p98 - p2 over the tile's valid pixels (nodata and NaN/inf excluded); NaN if none."""
    x = arr_hwc.reshape(-1, arr_hwc.shape[-1]).astype(np.float64)
    x = x[np.isfinite(x).all(axis=1) & ~np.all(x == 0, axis=1)]
    if len(x) == 0:
        return np.full(arr_hwc.shape[-1], np.nan)
    return np.percentile(x, 98, axis=0) - np.percentile(x, 2, axis=0)


def tile_classes(label: Any, kind: str, n_classes: int) -> list[int]:
    """Classes present in one tile: its label (classification), its positive labels
    (multilabel), or the classes with at least one pixel (segmentation)."""
    lab = np.asarray(label)
    if kind == "segmentation":
        return [int(v) for v in np.unique(lab) if 0 <= v < n_classes]
    if kind == "multilabel":
        return [int(c) for c in np.flatnonzero(lab)]
    return [int(lab)]


def flat_tile_fraction(
    spreads: np.ndarray, std: np.ndarray, classes: list[list[int]], class_names: list[str]
) -> dict:
    """Share of sampled train tiles that are nearly flat, overall and per class (see
    FLAT_DEFINITION). A tile with no valid pixel at all counts as flat."""
    safe_std = np.where(np.asarray(std) > 1e-6, std, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        contrast = np.nanmedian(spreads / safe_std, axis=1)
    flat = ~(contrast >= FLAT_THRESHOLD)  # NaN (all-nodata tile) counts as flat
    per_class = {}
    for c, name in enumerate(class_names):
        members = np.array([c in cl for cl in classes], dtype=bool)
        per_class[name] = round(float(flat[members].mean()), 4) if members.any() else None
    return {
        "definition": FLAT_DEFINITION,
        "threshold": FLAT_THRESHOLD,
        "n_tiles": int(len(flat)),
        "overall": round(float(flat.mean()), 4) if len(flat) else 0.0,
        "per_class": per_class,
    }


def majority_class_baseline(
    kind: str, train_counts: Counter, val_counts: Counter, n_val: int, class_names: list[str]
) -> dict:
    """Score on VAL of always predicting the most common TRAIN class, with the official metric.

    classification: accuracy = share of val samples in that class.
    segmentation:   mIoU as in contract v2: the majority class gets IoU = its share of valid val
                    pixels, every other class present in val gets IoU 0, classes absent from val
                    are skipped.
    multilabel:     micro-F1 of predicting every label that is positive in >= 50% of train images.
    """
    k = len(class_names)
    major = max(range(k), key=lambda c: train_counts.get(c, 0))
    if kind == "multilabel":
        n_train = max(train_counts.get("_n", 0), 1)
        on = [c for c in range(k) if train_counts.get(c, 0) / n_train >= 0.5]
        tp = sum(val_counts.get(c, 0) for c in on)
        fp = sum(n_val - val_counts.get(c, 0) for c in on)
        fn = sum(val_counts.get(c, 0) for c in range(k) if c not in on)
        score = 2 * tp / max(2 * tp + fp + fn, 1)
        return {"predict": [class_names[c] for c in on], "metric": "f1", "score": round(score, 4)}
    if kind == "segmentation":
        valid = sum(val_counts.get(c, 0) for c in range(k))
        present = [c for c in range(k) if val_counts.get(c, 0) > 0 or c == major]
        score = (val_counts.get(major, 0) / max(valid, 1)) / max(len(present), 1)
        metric = "miou"
    else:
        score = val_counts.get(major, 0) / max(n_val, 1)
        metric = "accuracy"
    return {"predict": class_names[major], "metric": metric, "score": round(score, 4)}


def shortcut_card_fields(majority: dict, flat: dict) -> dict:
    """Recon-card keys for contract request #11 (names final, agreed with Hilal).

    majority_class_baseline  float: VAL score of always predicting the most common TRAIN class
    flat_tile_fraction       {class: fraction of sampled train tiles that are flat}; classes
                             with no sampled tile are left out
    The rest is provenance kept next to them (what was predicted, metric, definition, sample).
    """
    return {
        "majority_class_baseline": majority["score"],
        "majority_class_baseline_detail": {
            "predict": majority["predict"],
            "metric": majority["metric"],
        },
        "flat_tile_fraction": {k: v for k, v in flat["per_class"].items() if v is not None},
        "flat_tile_overall": flat["overall"],
        "flat_tile_threshold": flat["threshold"],
        "flat_tile_definition": flat["definition"],
        "flat_tile_n": flat["n_tiles"],
    }


def inspect_task(
    task_name: str,
    data_dir: str | Path | None = None,
    max_stats_samples: int = 2000,
    seed: int = 0,
    stats_dir: str | Path | None = None,
) -> dict:
    """Measure one task on TRAIN + VAL. Writes stats/<task>.json and stats/<task>_card.json."""
    task = load_task(task_name, data_dir)
    kind = task_type(task)
    n_classes = int(task.label_type.n_classes)
    class_names, names_fixed = corrected_class_names(task_name, raw_class_names(task, n_classes))
    ds_train, ds_val = open_split(task, "train"), open_split(task, "val")
    n_train, n_val = len(ds_train), len(ds_val)
    rng = np.random.default_rng(seed)
    notes: list[str] = []

    # --- one sample in detail ---
    s0 = ds_train[0]
    arr0, raw_names, resampled = sample_array(s0)
    bands, fixed = channel_names(task_name, raw_names)
    lab0 = label_value(s0.label)
    if fixed:
        notes.append(
            f"Band names corrected: the published data mislabels channels ({ERRATA_URL}). "
            f"True order: {', '.join(bands)}."
        )
    if names_fixed:
        notes.append(
            "Class names corrected: geobench's class_names list does not match the label "
            "integers (see data_notes.md gotcha 14). Label order: "
            + ", ".join(f"{i} {n}" for i, n in enumerate(class_names))
            + "."
        )
    if resampled:
        notes.append(
            "Bands are stored at different native sizes; pack_to_3d(resample=True) upsamples "
            "them (cubic) to the largest size. train.py must do the same."
        )
    if isinstance(lab0, np.ndarray) and lab0.ndim >= 2:
        label_format = (
            f"mask {lab0.shape[0]}x{lab0.shape[1]} ({lab0.dtype}), one class index per pixel"
        )
    elif isinstance(lab0, np.ndarray) and lab0.ndim == 1:
        label_format = f"multi-hot vector of length {lab0.shape[0]} ({lab0.dtype})"
    else:
        label_format = f"single integer class index ({type(lab0).__name__})"
    sample_bands = [
        {
            "band": name,
            "min": float(np.nanmin(arr0[..., i])),
            "max": float(np.nanmax(arr0[..., i])),
            "mean": float(np.nanmean(arr0[..., i].astype(np.float64))),
        }
        for i, name in enumerate(bands)
    ]
    native_res = {
        bands[i]: (S2_NATIVE_RES_M.get(bands[i]) if fixed else bi.spatial_resolution)
        for i, bi in enumerate(s0.band_info_list)
        if i < len(bands)
    }

    # --- labels over the FULL train split (label-only reads are cheap) ---
    counts = _label_counts(ds_train, kind)
    mask_unique_values = None
    if kind == "segmentation":
        mask_unique_values = _mask_summary(counts, n_classes, notes)
    total = sum(counts[c] for c in range(n_classes)) or 1
    denom = n_train if kind == "multilabel" else total
    class_distribution = {
        class_names[c]: {
            "count": int(counts.get(c, 0)),
            "pct": round(100 * counts.get(c, 0) / denom, 3),
        }
        for c in range(n_classes)
    }
    _distribution_notes(class_distribution, kind, notes)

    # --- pixel statistics on a random subset of TRAIN (seeded) ---
    n_stats = min(max_stats_samples, n_train)
    rs = RunningBandStats(len(bands), seed=seed)
    train_hashes: dict[str, int] = {}
    dup_train = 0
    dtypes = set()
    spreads, sample_classes = [], []
    for i in rng.choice(n_train, size=n_stats, replace=False):
        sample = ds_train[int(i)]
        arr, _, _ = sample_array(sample)
        dtypes.add(str(arr.dtype))
        rs.update(arr)
        spreads.append(tile_spread(arr))
        sample_classes.append(tile_classes(label_value(sample.label), kind, n_classes))
        h = hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()
        dup_train += h in train_hashes
        train_hashes[h] = int(i)

    # --- duplicates between val and the sampled train images (leakage check) ---
    leak = 0
    for i in rng.choice(n_val, size=min(max_stats_samples, n_val), replace=False):
        arr, _, _ = sample_array(ds_val[int(i)])
        leak += hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest() in train_hashes
    if dup_train:
        notes.append(f"{dup_train} exact duplicate image(s) inside the sampled train set.")
    if leak:
        notes.append(
            f"{leak} validation image(s) identical to a train image: val scores may be optimistic."
        )

    p1, p99 = rs.percentiles(1), rs.percentiles(99)
    mean, std = rs.mean, rs.std
    value_range = {
        b: {
            "min": float(rs.min[i]),
            "max": float(rs.max[i]),
            "p1": float(p1[i]),
            "p99": float(p99[i]),
        }
        for i, b in enumerate(bands)
    }
    constant_bands = [
        b for i, b in enumerate(bands) if std[i] < 1e-6 or rs.constant_in_image[i] > 0.5 * rs.images
    ]
    if constant_bands:
        notes.append(f"constant (or mostly constant) bands: {constant_bands}. Consider dropping.")
    nodata_fraction = rs.nodata_pixels / max(rs.total_pixels, 1)
    if nodata_fraction > 0:
        notes.append(
            f"nodata ({NODATA_RULE}): {100 * nodata_fraction:.3f}% of pixels; excluded from stats."
        )
    if rs.nonfinite_values:
        notes.append(
            f"{rs.nonfinite_values} NaN/inf values found; excluded from stats. "
            "train.py must handle them."
        )
    _scale_note(bands, p99, notes)
    flat = flat_tile_fraction(np.asarray(spreads), std, sample_classes, class_names)
    train_counts = Counter(counts)
    train_counts["_n"] = n_train
    majority = majority_class_baseline(
        kind, train_counts, _label_counts(ds_val, kind), n_val, class_names
    )
    if n_train < 1000:
        notes.append(f"small train set ({n_train}): prefer pretrained backbones and augmentation.")

    # --- save the stats file read by train.py (contract: BandStats) ---
    stats_dir = Path(stats_dir) if stats_dir else get_settings().stats_dir
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "task": task_name,
        "split": "train",
        "n_samples": int(n_stats),
        "seed": seed,
        "bands": bands,
        "mean": [round(float(v), 6) for v in mean],
        "std": [round(float(v), 6) for v in std],
        "excluded_pixels": int(rs.excluded_pixels),
    }
    (stats_dir / f"{task_name}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    card = {
        "task": task_name,
        "task_type": kind,
        "sensor": sensor(task),
        "bands": bands,
        "n_classes": n_classes,
        "class_names": class_names,
        "image_size": [int(arr0.shape[0]), int(arr0.shape[1])],
        "split_sizes": {"train": n_train, "val": n_val},
        "class_distribution": class_distribution,
        "band_mean": stats["mean"],
        "band_std": stats["std"],
        "notes": notes,
        # recon extras
        "dtype": sorted(dtypes),
        "value_range": value_range,
        "label_format": label_format,
        "mask_unique_values": mask_unique_values,
        "nodata_fraction": round(nodata_fraction, 6),
        "constant_bands": constant_bands,
        "stats_n_samples": int(n_stats),
        "spatial_resolution_m": float(task.spatial_resolution) if task.spatial_resolution else None,
        "native_band_resolution_m": native_res,
        "band_names_corrected": fixed,
        "class_names_corrected": names_fixed,
        "excluded_pixels": int(rs.excluded_pixels),
        **shortcut_card_fields(majority, flat),
        "example_sample": {
            "name": s0.sample_name,
            "shape_hwc": list(arr0.shape),
            "dtype": str(arr0.dtype),
            "bands": sample_bands,
        },
    }
    card_path = stats_dir / f"{task_name}_card.json"
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    return card


# ---------------------------------------------------------------------------------------------
# 2c. describe_dataset: recon card -> contract DatasetCard
# ---------------------------------------------------------------------------------------------
def _extra_notes(recon: dict) -> list[str]:
    """Facts the flat contract fields would lose, as short notes for the agent."""
    notes = []
    res = recon.get("spatial_resolution_m")
    h, w = recon["image_size"]
    if res:
        notes.append(
            f"Ground resolution {res:g} m/pixel: one {h}x{w} image covers "
            f"{h * res:g} m x {w * res:g} m."
        )
    notes.append("Label integer i means class_names[i] (errata already applied).")
    if recon["sensor"] == "sentinel-2":
        zero = [b for b in SSL4EO_S2_BANDS if b not in recon["bands"]]
        extra = [b for b in recon["bands"] if b not in SSL4EO_S2_BANDS]
        msg = "With weights='ssl4eo_s2' the engine feeds the 13 SSL4EO bands in TorchGeo order"
        msg += f"; zero-filled (missing here): {', '.join(zero)}" if zero else "; all present"
        if extra:
            msg += f"; left out (not Sentinel-2 reflectance): {', '.join(extra)}"
        notes.append(msg + ".")
    ranges = recon.get("value_range") or {}
    p99 = {b: r["p99"] for b, r in ranges.items() if np.isfinite(r["p99"]) and r["p99"] > 0}
    if len(p99) >= 2:
        top = max(p99.values())
        empty = [f"{b} (p99 {v:g})" for b, v in p99.items() if v < NEARLY_EMPTY_RATIO * top]
        if empty:
            notes.append(
                f"Nearly empty band(s): {', '.join(empty)} vs max p99 {top:g}. Mostly noise after "
                "normalisation; a candidate to drop via `bands`."
            )
    unit = {"segmentation": "pixels", "multilabel": "images (can sum > 1)"}.get(
        recon["task_type"], "samples"
    )
    notes.append(f"class_distribution = fraction of train {unit}.")
    notes.extend(_shortcut_notes(recon))
    notes.extend(DATASET_FACTS.get(recon["task"], []))
    if recon.get("excluded_pixels"):
        notes.append(f"{recon['excluded_pixels']} sampled train pixels excluded from stats.")
    return notes


def _shortcut_notes(recon: dict) -> list[str]:
    """Majority-class baseline and flat-tile fractions as plain facts (no advice in the card)."""
    notes = []
    if recon.get("majority_class_baseline") is not None:
        detail = recon.get("majority_class_baseline_detail", {})
        what = detail.get("predict", "the most common train class")
        what = what if isinstance(what, str) else ", ".join(what)
        notes.append(
            f"Majority-class baseline on val: always predicting '{what}' scores "
            f"{recon['majority_class_baseline']:.3f} {detail.get('metric', '')}".rstrip()
            + "."
        )
    flat = recon.get("flat_tile_fraction")
    if flat is not None:
        by_class = sorted(flat.items(), key=lambda kv: -kv[1])[:10]
        notes.append(
            f"Flat tiles ({recon.get('flat_tile_definition', 'low contrast')}): "
            f"{recon.get('flat_tile_overall', 0):.1%} of {recon.get('flat_tile_n', '?')} sampled "
            "train tiles; by class: " + ", ".join(f"{k} {v:.1%}" for k, v in by_class) + "."
        )
    return notes


def to_dataset_card(recon: dict) -> DatasetCard:
    """Flatten a recon card (stats/<task>_card.json) into the contract's DatasetCard.

    Four recon fields are richer than the contract allows, so they are flattened:
      class_distribution {name: {count, pct}} -> {name: fraction}
      dtype [list]                            -> "int16" (comma-joined if several)
      value_range {band: {min, max, p1, p99}} -> (min over bands, max over bands)
      mask_unique_values {values: {...}, ...} -> sorted list of mask values
    """
    mask = recon.get("mask_unique_values")
    ranges = recon.get("value_range") or {}
    value_range = None
    if ranges:
        value_range = (
            min(r["min"] for r in ranges.values()),
            max(r["max"] for r in ranges.values()),
        )
    dtype = recon.get("dtype")
    return DatasetCard(
        task=recon["task"],
        task_type=recon["task_type"],
        sensor=recon["sensor"],
        bands=recon["bands"],
        n_classes=recon["n_classes"],
        class_names=recon["class_names"],
        image_size=tuple(recon["image_size"]),
        split_sizes=recon["split_sizes"],
        class_distribution={
            name: round(d["pct"] / 100, 6) for name, d in recon["class_distribution"].items()
        },
        band_mean=recon["band_mean"],
        band_std=recon["band_std"],
        notes=list(recon.get("notes", [])) + _extra_notes(recon),
        dtype=", ".join(dtype) if isinstance(dtype, list) else dtype,
        value_range=value_range,
        label_format=recon.get("label_format"),
        mask_unique_values=sorted(int(v) for v in mask["values"]) if mask else None,
        # contract request #11 (final names); ignored by DatasetCard until Hilal adds the fields
        majority_class_baseline=recon.get("majority_class_baseline"),
        flat_tile_fraction=recon.get("flat_tile_fraction"),
        nodata_fraction=recon.get("nodata_fraction"),
        constant_bands=recon.get("constant_bands"),
        stats_n_samples=recon.get("stats_n_samples"),
    )


def describe_dataset(task: str) -> DatasetCard:
    """Contract entry point. Uses the cached recon card; measures the task if there is none."""
    path = get_settings().stats_dir / f"{task}_card.json"
    if path.exists():
        recon = json.loads(path.read_text(encoding="utf-8"))
    else:
        recon = inspect_task(task)  # raises FileNotFoundError for unknown/undownloaded tasks
    return to_dataset_card(recon)


# ---------------------------------------------------------------------------------------------
# 2d. One-time .npy cache of TRAIN and VAL (never the test split)
# ---------------------------------------------------------------------------------------------
# Loading every tile from its own hdf5 file each epoch dominates training time on the T4
# (brick-kiln speed check). scripts/build_cache.py writes each split once as float32 arrays:
#   <cache root>/<task>/<split>_x.npy   [N, H, W, C] float32, all bands in card order, raw units
#   <cache root>/<task>/<split>_y.npy   labels (int64 [N] | float32 [N, K] | int16/int32 masks)
#   <cache root>/<task>/<split>_meta.json   format, fingerprint, bands, shapes
# GeoBenchDataset memory-maps them when the fingerprint still matches the source files;
# otherwise it reads hdf5 as before. Results are identical either way (tested).
CACHE_FORMAT = 1
CACHE_SPLITS = ("train", "val")  # the test split is NEVER cached


def cache_root() -> Path:
    """GOTRAIN_DATA_CACHE, default ~/data_cache (outside the repo and the shared ~/data)."""
    return Path(os.environ.get("GOTRAIN_DATA_CACHE", "~/data_cache")).expanduser()


def split_fingerprint(task_name: str, split: str, ds: Any, bands: list[str]) -> str:
    """Changes whenever the cached content could: format, task, split, bands, sample list, and
    the total size of the split's hdf5 files (a re-download or edit changes it)."""
    names = list(ds.active_partition.partition_dict[ds.split])
    total = sum(
        (Path(ds.dataset_dir) / f"{n}.hdf5").stat().st_size
        for n in names
        if (Path(ds.dataset_dir) / f"{n}.hdf5").exists()
    )
    payload = json.dumps([CACHE_FORMAT, task_name, split, bands, names, total])
    return hashlib.sha256(payload.encode()).hexdigest()


def check_cache(task_name: str, split: str, ds: Any, bands: list[str]) -> tuple:
    """(memory-mapped (x, y), None) if a valid cache exists, else (None, reason).

    Every reason the engine falls back to hdf5 is named, so a slow run can be explained.
    The cache location is cache_root(): $GOTRAIN_DATA_CACHE, default ~/data_cache. A different
    GEO_BENCH_DIR does NOT invalidate it by itself: the fingerprint covers the sample list and
    the files' total size, not their folder, so an identical copy of the data still matches.
    """
    if split not in CACHE_SPLITS:
        return None, f"the {split} split is never cached"
    folder = cache_root() / task_name
    meta_path = folder / f"{split}_meta.json"
    if not meta_path.exists():
        return None, f"no cache at {folder} (build it: scripts/build_cache.py {task_name})"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("format") != CACHE_FORMAT:
            return None, f"cache format {meta.get('format')} != {CACHE_FORMAT} (rebuild it)"
        if meta.get("bands") != bands:
            return None, f"cache was built for bands {meta.get('bands')} (rebuild it)"
        if meta.get("fingerprint") != split_fingerprint(task_name, split, ds, bands):
            return None, "stale cache: the source files changed since it was built (rebuild it)"
        x = np.load(folder / f"{split}_x.npy", mmap_mode="r")
        y = np.load(folder / f"{split}_y.npy", mmap_mode="r")
    except (OSError, ValueError) as exc:
        return None, f"cache unreadable: {type(exc).__name__}: {exc}"
    if len(x) != len(ds) or len(y) != len(ds):
        return None, f"cache has {len(x)} rows but the split has {len(ds)} (rebuild it)"
    return (x, y), None


def load_cache(task_name: str, split: str, ds: Any, bands: list[str]) -> tuple | None:
    """Memory-mapped (x, y) if a valid cache exists for this split, else None."""
    return check_cache(task_name, split, ds, bands)[0]


def cache_bytes(n: int, image_hwc: tuple, label_shape: tuple, label_itemsize: int) -> int:
    """Disk size of one cached split: float32 images + labels."""
    return int(n * (np.prod(image_hwc) * 4 + np.prod(label_shape or (1,)) * label_itemsize))


def _label_store_dtype(kind: str, label: Any) -> Any:
    if kind == "segmentation":
        return np.int16 if np.asarray(label).max(initial=0) < 32767 else np.int32
    return np.float32 if kind == "multilabel" else np.int64


def build_split_cache(
    task_name: str, split: str, data_dir: str | Path | None = None, progress: Any = None
) -> dict:
    """Write <cache>/<task>/<split>_{x,y}.npy + meta (atomically). TRAIN/VAL only."""
    if split not in CACHE_SPLITS:
        raise PermissionError(f"only {CACHE_SPLITS} are cached; refusing split={split!r}")
    task = load_task(task_name, data_dir)
    ds = open_split(task, split)
    kind = task_type(task)
    first = ds[0]
    arr0, raw_names, _ = sample_array(first)
    bands = channel_names(task_name, raw_names)[0]
    lab0 = np.asarray(label_value(first.label))
    label_dtype = _label_store_dtype(kind, lab0)
    folder = cache_root() / task_name
    folder.mkdir(parents=True, exist_ok=True)
    tmp_x, tmp_y = folder / f".{split}_x.tmp.npy", folder / f".{split}_y.tmp.npy"
    x = np.lib.format.open_memmap(tmp_x, mode="w+", dtype=np.float32, shape=(len(ds), *arr0.shape))
    y = np.lib.format.open_memmap(tmp_y, mode="w+", dtype=label_dtype, shape=(len(ds), *lab0.shape))
    for i in range(len(ds)):
        sample = first if i == 0 else ds[i]
        x[i] = sample_array(sample)[0]
        y[i] = np.asarray(label_value(sample.label))
        if progress and (i + 1) % 500 == 0:
            progress(i + 1, len(ds))
    x.flush()
    y.flush()
    del x, y
    meta = {
        "format": CACHE_FORMAT,
        "task": task_name,
        "split": split,
        "n": len(ds),
        "bands": bands,
        "image_shape_hwc": list(arr0.shape),
        "source_dtype": str(arr0.dtype),
        "label_dtype": np.dtype(label_dtype).name,
        "fingerprint": split_fingerprint(task_name, split, ds, bands),
        "bytes": (tmp_x.stat().st_size + tmp_y.stat().st_size),
    }
    tmp_x.replace(folder / f"{split}_x.npy")
    tmp_y.replace(folder / f"{split}_y.npy")
    (folder / f"{split}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


# ---------------------------------------------------------------------------------------------
# 3. PyTorch dataset
# ---------------------------------------------------------------------------------------------
Norm = Literal["zscore", "div10000", "none"] | tuple[Sequence[float], Sequence[float]]


def subset_indices(n: int, frac: float, seed: int) -> list[int]:
    """A seeded random subset of range(n) (at least one item), in sorted order."""
    if frac >= 1.0:
        return list(range(n))
    k = max(1, round(n * frac))
    return sorted(np.random.default_rng(seed).choice(n, size=k, replace=False).tolist())


class GeoBenchDataset:
    """Map-style dataset: item i -> (x float32 [C, H, W], y).

    y is a long scalar (classification), a float multi-hot vector (multilabel) or a long
    [H, W] mask (segmentation; values outside [0, n_classes) become IGNORE_INDEX).

    bands: "all" or band names (true names, errata applied), in the order the model wants.
    zero_fill: band names in `bands` that this task lacks; fed as all-zero channels.
    use_cache: read the split from the on-disk .npy cache (scripts/build_cache.py) when one
           exists and still matches the source files; never for the test split.
    norm:  "zscore"   (x - mean) / std with stats/<task>.json (TRAIN-only stats)
           "div10000" x / 10000, Sentinel-2 reflectance scale (what SSL4EO weights expect)
           "none"     raw values as float32
           (mean, std) explicit per-band arrays (models.input_normalisation, checkpoints)
    """

    def __init__(
        self,
        task_name: str,
        split: str,
        bands: Literal["all"] | Sequence[str] = "all",
        norm: Norm = "zscore",
        indices: Sequence[int] | None = None,
        data_dir: str | Path | None = None,
        allow_test: bool = False,
        cache: bool = False,
        zero_fill: Sequence[str] = (),
        use_cache: bool = True,
    ) -> None:
        self.task_name = task_name
        self.split = split
        self.task = load_task(task_name, data_dir)
        self.ds = open_split(self.task, split, allow_test=allow_test)
        self.task_type = task_type(self.task)
        self.n_classes = int(self.task.label_type.n_classes)
        self.class_names = class_names_for(task_name, self.task)
        self.all_bands = channel_names(task_name, sample_array(self.ds[0])[1])[0]
        self.bands = list(self.all_bands) if bands == "all" else list(bands)
        self.zero_fill = [b for b in self.bands if b in set(zero_fill)]
        unknown = [b for b in self.bands if b not in self.all_bands and b not in self.zero_fill]
        if unknown:
            raise ValueError(f"bands {unknown} not in {task_name}; available: {self.all_bands}")
        # -1 marks a zero-filled channel (a band the model expects but this task lacks)
        self.channel_idx = [
            -1 if b in self.zero_fill else self.all_bands.index(b) for b in self.bands
        ]
        self.norm = norm
        self.mean, self.std = self._norm_params(norm)
        self.indices = list(indices) if indices is not None else list(range(len(self.ds)))
        self._cache: dict[int, tuple[np.ndarray, Any]] | None = {} if cache else None
        # on-disk .npy cache (train/val only; load_cache returns None for the test split)
        if use_cache:
            self.disk_cache, self.cache_reason = check_cache(
                task_name, split, self.ds, self.all_bands
            )
        else:
            self.disk_cache, self.cache_reason = None, "disabled (use_cache=False)"

    def _norm_params(self, norm: Norm) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.bands)
        if isinstance(norm, tuple):  # explicit per-band (mean, std), e.g. from a checkpoint
            mean, std = (np.asarray(v, np.float32) for v in norm)
            if mean.shape != (n,) or std.shape != (n,):
                raise ValueError(f"norm arrays must have one value per band ({n})")
            return mean, np.where(std < 1e-6, 1.0, std).astype(np.float32)
        if norm == "div10000":
            return np.zeros(n, np.float32), np.full(n, 10000.0, np.float32)
        if norm == "none":
            return np.zeros(n, np.float32), np.ones(n, np.float32)
        if norm != "zscore":
            raise ValueError(f"norm must be zscore, div10000 or none; got {norm!r}")
        from gotrain.engine.interface import load_band_stats

        stats = load_band_stats(self.task_name)
        if stats.bands != self.all_bands:
            raise ValueError(
                f"stats/{self.task_name}.json bands {stats.bands} != dataset bands "
                f"{self.all_bands}; re-run scratch/explore.py"
            )
        if self.zero_fill:
            raise ValueError("zero-filled bands need explicit (mean, std) normalisation")
        mean = np.asarray(stats.mean, np.float32)[self.channel_idx]
        std = np.asarray(stats.std, np.float32)[self.channel_idx]
        return mean, np.where(std < 1e-6, 1.0, std).astype(np.float32)

    def __getstate__(self) -> dict:
        """Pickled for DataLoader workers (forkserver/spawn): never ship the memory-mapped cache
        itself (pickling a memmap copies the whole array); each worker reopens it."""
        state = self.__dict__.copy()
        state["_cache_files"] = self._cache_files() if self.disk_cache is not None else None
        state["disk_cache"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        files = state.pop("_cache_files", None)
        self.__dict__.update(state)
        if files is not None:  # the parent already validated the fingerprint
            self.disk_cache = tuple(np.load(f, mmap_mode="r") for f in files)

    def _cache_files(self) -> tuple[str, str]:
        folder = cache_root() / self.task_name
        return str(folder / f"{self.split}_x.npy"), str(folder / f"{self.split}_y.npy")

    def __len__(self) -> int:
        return len(self.indices)

    def sample_name(self, i: int) -> str:
        return self.ds.active_partition.partition_dict[self.ds.split][self.indices[i]]

    def raw(self, i: int) -> tuple[np.ndarray, Any]:
        """(H, W, C) array with ALL bands in true order (unnormalised) and the raw label."""
        idx = self.indices[i]
        if self.disk_cache is not None:
            x, y = self.disk_cache
            label = np.asarray(y[idx])
            return np.asarray(x[idx]), (label.item() if label.ndim == 0 else label)
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]
        sample = self.ds[idx]
        item = (sample_array(sample)[0], label_value(sample.label))
        if self._cache is not None:
            self._cache[idx] = item
        return item

    def _target(self, label: Any) -> Any:
        import torch

        if self.task_type == "segmentation":
            mask = np.asarray(label).squeeze().astype(np.int64)
            if mask.max(initial=0) == 255 and set(np.unique(mask).tolist()) <= {0, 255}:
                mask = mask // 255
            mask[(mask < 0) | (mask >= self.n_classes)] = IGNORE_INDEX
            return torch.from_numpy(mask)
        if self.task_type == "multilabel":
            return torch.as_tensor(np.asarray(label), dtype=torch.float32)
        return torch.tensor(int(label), dtype=torch.long)

    def __getitem__(self, i: int) -> tuple[Any, Any]:
        import torch

        arr, label = self.raw(i)
        x = np.zeros(arr.shape[:2] + (len(self.bands),), dtype=np.float32)
        for out_c, in_c in enumerate(self.channel_idx):
            if in_c >= 0:
                x[..., out_c] = arr[..., in_c]
        x = np.nan_to_num((x - self.mean) / self.std, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        return x, self._target(label)
