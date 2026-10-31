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
}  # fmt: skip

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
AERIAL_LETTERS = {"red": "r", "green": "g", "blue": "b", "nir": "n", "near-infrared": "n"}


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
    for i in rng.choice(n_train, size=n_stats, replace=False):
        arr, _, _ = sample_array(ds_train[int(i)])
        dtypes.add(str(arr.dtype))
        rs.update(arr)
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
    if recon.get("excluded_pixels"):
        notes.append(f"{recon['excluded_pixels']} sampled train pixels excluded from stats.")
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

    def __len__(self) -> int:
        return len(self.indices)

    def sample_name(self, i: int) -> str:
        return self.ds.active_partition.partition_dict[self.ds.split][self.indices[i]]

    def raw(self, i: int) -> tuple[np.ndarray, Any]:
        """(H, W, C) array with ALL bands in true order (unnormalised) and the raw label."""
        idx = self.indices[i]
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
