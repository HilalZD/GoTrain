"""GEO-Bench data reconnaissance for GoTrain.

    python scratch/explore.py m-eurosat m-pv4ger-seg [--data-dir data]

For each task this prints a readable summary, writes
    stats/<task>.json        train-split per-band mean/std (read by train.py)
    stats/<task>_card.json   the full DatasetCard dict (future inspect_dataset tool)
    docs/figures/<task>_samples.png   8 training samples

Rules this file enforces:
  * Only the TRAIN and VALID splits are ever opened. `_open_split` refuses "test".
  * No task-specific logic: task type, bands, classes and labels come from geobench's
    own task metadata. The one exception is BAND_ORDER_FIXES, which is upstream errata
    (data, not logic) for two published datasets with mislabelled channels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

ALLOWED_SPLITS = ("train", "valid")  # geobench's names; "valid" is the validation split
NODATA_RULE = "pixels where every band is exactly 0, plus NaN/inf values"

# Upstream errata: geo-bench README "Known issues", issues #28 and #29. The pixel data is fine,
# but the stored band names are wrong for most channels. These are the TRUE channel names in order.
BAND_ORDER_FIXES = {
    "m-eurosat": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"],
    "m-brick-kiln": ["B01", "B02", "B03", "B04", "B05", "B07", "B8A", "B08", "B11", "B12", "TCI_R", "TCI_G", "TCI_B"],
}
ERRATA_URL = "https://github.com/ServiceNow/geo-bench#known-issues"

SENSOR_NAMES = {
    "Sentinel2": "Sentinel-2", "Sentinel1": "Sentinel-1", "Landsat8": "Landsat-8",
    "HyperSpectralBands": "hyperspectral", "ElevationBand": "elevation",
    "SpectralBand": "optical (aerial or other)", "CloudProbability": "cloud probability",
}


# ----------------------------------------------------------------------------------------------
# Loading helpers
# ----------------------------------------------------------------------------------------------
def find_task_dir(task_name: str, data_dir: str | None = None) -> Path:
    """Locate <root>/<benchmark>/<task_name>/task_specs.pkl without relying on import-time env."""
    roots = [Path(data_dir)] if data_dir else []
    if os.environ.get("GEO_BENCH_DIR"):
        roots.append(Path(os.environ["GEO_BENCH_DIR"]))
    roots.append(Path.home() / "dataset" / "geobench")
    for root in roots:
        for cand in sorted(root.glob(f"*/{task_name}")):
            if (cand / "task_specs.pkl").exists():
                return cand
    raise FileNotFoundError(f"Task {task_name} not found under {[str(r) for r in roots]}. "
                            "Run scratch/download_tasks.py first or pass --data-dir.")


def load_task(task_name: str, data_dir: str | None = None):
    import geobench
    return geobench.load_task_specs(find_task_dir(task_name, data_dir))


def _open_split(task, split: str):
    """Open a split. The test split is refused here, so nothing in this file can touch it."""
    if split not in ALLOWED_SPLITS:
        raise PermissionError(f"explore.py only reads {ALLOWED_SPLITS}; refusing split={split!r}")
    # Pass ALL band names: geobench's get_dataset() defaults to ("red","green","blue"),
    # which silently selects 3 bands (and crashes on datasets without RGB).
    band_names = [bi.name for bi in task.bands_info]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the band-order warning; we handle it via BAND_ORDER_FIXES
        return task.get_dataset(split=split, band_names=band_names)


def _sample_array(sample) -> tuple[np.ndarray, list[str], bool]:
    """Return (H, W, C) array, raw channel names, and whether any band had to be resampled."""
    names = [bi.name for bi in sample.band_info_list]
    shapes = {b.data.shape[:2] for b in sample.bands}
    arr, ch_names = sample.pack_to_3d(band_names=names, resample=True)
    return arr, list(ch_names), len(shapes) > 1


def _channel_names(task_name: str, raw_names: list[str]) -> tuple[list[str], bool]:
    fix = BAND_ORDER_FIXES.get(task_name)
    if fix is not None and len(fix) == len(raw_names):
        return list(fix), True
    return raw_names, False


def _label_value(label):
    """Band -> its pixel array; anything else (int, vector) unchanged."""
    return label.data if hasattr(label, "data") else label


def _labels_only(ds, idx: int):
    """Read only the label of sample idx (fast path for hdf5), falling back to a full load."""
    from geobench.dataset import load_sample_hdf5
    name = ds.active_partition.partition_dict[ds.split][idx]
    path = Path(ds.dataset_dir) / f"{name}.hdf5"
    if path.exists():
        return _label_value(load_sample_hdf5(path, label_only=True).label)
    return _label_value(ds[idx].label)


# ----------------------------------------------------------------------------------------------
# Streaming statistics (Chan et al. parallel merge of per-image mean / M2, accumulated in float64)
# ----------------------------------------------------------------------------------------------
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
        self.m2 = np.where(ok, self.m2 + np.nan_to_num(m2b) + delta**2 * self.n * nb / safe_tot, self.m2)
        self.n = tot
        pick = self.rng.choice(x.shape[0], size=min(self.k, x.shape[0]), replace=False)
        self.reservoir.append(x[pick])

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.m2 / np.maximum(self.n, 1))

    def percentiles(self, q) -> np.ndarray:
        r = np.concatenate(self.reservoir, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanpercentile(r, q, axis=0)


# ----------------------------------------------------------------------------------------------
# The reusable function (becomes the agent's inspect_dataset tool)
# ----------------------------------------------------------------------------------------------
def _task_type(task) -> str:
    kind = type(task.label_type).__name__
    if "MultiLabel" in kind:
        return "multilabel"
    if "Segmentation" in kind:
        return "segmentation"
    if "Classification" in kind:
        return "classification"
    return kind.lower()


def _class_names(task, n_classes: int) -> list[str]:
    lt = task.label_type
    names = getattr(lt, "class_names", None) or getattr(lt, "class_name", None)  # multilabel uses a typo'd attr upstream
    return list(names) if names else [str(i) for i in range(n_classes)]


def _sensor(task) -> str:
    kinds = []
    for bi in task.bands_info:
        k = SENSOR_NAMES.get(type(bi).__name__, type(bi).__name__)
        if k not in kinds:
            kinds.append(k)
    return " + ".join(kinds)


def inspect_task(task_name: str, data_dir: str | None = None, max_stats_samples: int = 2000,
                 seed: int = 0, stats_dir: str | Path = "stats", verbose: bool = True) -> dict:
    """Inspect one GEO-Bench task using only train + valid. Saves stats/<task>.json. Returns a DatasetCard dict."""
    task = load_task(task_name, data_dir)
    task_type = _task_type(task)
    n_classes = int(task.label_type.n_classes)
    class_names = _class_names(task, n_classes)
    ds_train, ds_val = _open_split(task, "train"), _open_split(task, "valid")
    n_train, n_val = len(ds_train), len(ds_val)
    rng = np.random.default_rng(seed)
    notes: list[str] = []

    # --- one sample in detail ---
    s0 = ds_train[0]
    arr0, raw_names, resampled = _sample_array(s0)
    bands, fixed = _channel_names(task_name, raw_names)
    lab0 = _label_value(s0.label)
    if fixed:
        notes.append(f"Band names corrected: the published data mislabels channels ({ERRATA_URL}). "
                     f"True order: {', '.join(bands)}.")
    if resampled:
        notes.append("Bands are stored at different native sizes; pack_to_3d(resample=True) upsamples them "
                     "(cubic) to the largest size. train.py must do the same.")
    if isinstance(lab0, np.ndarray) and lab0.ndim >= 2:
        label_format = f"mask {lab0.shape[0]}x{lab0.shape[1]} ({lab0.dtype}), one class index per pixel"
    elif isinstance(lab0, np.ndarray) and lab0.ndim == 1:
        label_format = f"multi-hot vector of length {lab0.shape[0]} ({lab0.dtype})"
    else:
        label_format = f"single integer class index ({type(lab0).__name__})"
    sample_bands = []
    for i, name in enumerate(bands):
        ch = arr0[..., i].astype(np.float64)
        sample_bands.append({"band": name, "min": float(np.nanmin(ch)), "max": float(np.nanmax(ch)),
                             "mean": float(np.nanmean(ch))})
    native_res = {}
    for i, bi in enumerate(s0.band_info_list):
        if i < len(bands):
            native_res[bands[i]] = bi.spatial_resolution

    # --- labels over the FULL train split (label-only reads are cheap) ---
    counts: Counter = Counter()
    for i in range(n_train):
        lab = _labels_only(ds_train, i)
        if task_type == "segmentation":
            vals, cnt = np.unique(np.asarray(lab), return_counts=True)
            counts.update({int(v): int(c) for v, c in zip(vals, cnt)})
        elif task_type == "multilabel":
            for c in np.flatnonzero(np.asarray(lab)):
                counts[int(c)] += 1
        else:
            counts[int(lab)] += 1

    mask_unique_values = None
    unit = "samples"
    if task_type == "segmentation":
        unit = "pixels"
        values = sorted(counts)
        ignore = [v for v in values if v < 0 or v >= n_classes]
        mask_unique_values = {
            "values": {str(v): counts[v] for v in values},
            "is_0_1": set(values) <= {0, 1},
            "is_0_255": 255 in values and set(values) <= {0, 255},
            "ignore_values": ignore,
        }
        if mask_unique_values["is_0_255"]:
            notes.append("Masks use 0/255, not 0/1: divide by 255 (or map 255->1) before the loss.")
        elif ignore:
            notes.append(f"Mask values outside [0, {n_classes - 1}]: {ignore}. Treat as ignore_index in the loss.")
    total = sum(counts[c] for c in range(n_classes)) or 1
    denom = n_train if task_type == "multilabel" else total
    class_distribution = {
        class_names[c]: {"count": int(counts.get(c, 0)), "pct": round(100 * counts.get(c, 0) / denom, 3)}
        for c in range(n_classes)
    }
    pcts = {k: v["pct"] for k, v in class_distribution.items()}
    for name, p in pcts.items():
        what = "of pixels" if unit == "pixels" else ("of images" if task_type == "multilabel" else "of samples")
        if class_distribution[name]["count"] == 0:
            notes.append(f"class '{name}' never appears in train: the model cannot learn it.")
        elif p < 1:
            notes.append(f"severe imbalance: class '{name}' is {p:.2f}% {what}")
        elif p < 5:
            notes.append(f"imbalance: class '{name}' is {p:.2f}% {what}")
    nonzero = [p for p in pcts.values() if p > 0]
    if len(nonzero) >= 2 and max(nonzero) / min(nonzero) > 10 and task_type != "multilabel":
        notes.append(f"largest/smallest class ratio is {max(nonzero) / min(nonzero):.0f}x: "
                     "use class weights or Dice/focal loss, and report per-class metrics.")

    # --- pixel statistics on a random subset of TRAIN (seeded) ---
    n_stats = min(max_stats_samples, n_train)
    stats_idx = rng.choice(n_train, size=n_stats, replace=False)
    rs = RunningBandStats(len(bands), seed=seed)
    train_hashes: dict[str, int] = {}
    dup_train = 0
    dtypes = set()
    for i in stats_idx:
        arr, _, _ = _sample_array(ds_train[int(i)])
        dtypes.add(str(arr.dtype))
        rs.update(arr)
        h = hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()
        if h in train_hashes:
            dup_train += 1
        train_hashes[h] = int(i)

    # --- duplicates between valid and the sampled train images (leakage check) ---
    n_val_check = min(max_stats_samples, n_val)
    leak = 0
    for i in rng.choice(n_val, size=n_val_check, replace=False):
        arr, _, _ = _sample_array(ds_val[int(i)])
        if hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest() in train_hashes:
            leak += 1
    if dup_train:
        notes.append(f"{dup_train} exact duplicate image(s) inside the sampled train set.")
    if leak:
        notes.append(f"{leak} validation image(s) identical to a train image: validation scores may be optimistic.")

    p1, p99 = rs.percentiles(1), rs.percentiles(99)
    mean, std = rs.mean, rs.std
    value_range = {b: {"min": float(rs.min[i]), "max": float(rs.max[i]),
                       "p1": float(p1[i]), "p99": float(p99[i])} for i, b in enumerate(bands)}
    constant_bands = [b for i, b in enumerate(bands)
                      if std[i] < 1e-6 or rs.constant_in_image[i] > 0.5 * rs.images]
    if constant_bands:
        notes.append(f"constant (or mostly constant) bands: {constant_bands}. Consider dropping them.")
    nodata_fraction = rs.nodata_pixels / max(rs.total_pixels, 1)
    if nodata_fraction > 0:
        notes.append(f"nodata ({NODATA_RULE}): {100 * nodata_fraction:.3f}% of pixels; excluded from stats.")
    if rs.nonfinite_values:
        notes.append(f"{rs.nonfinite_values} NaN/inf values found; excluded from stats. train.py must handle them.")
    finite_p99 = p99[np.isfinite(p99) & (p99 > 0)]
    if finite_p99.size >= 2 and finite_p99.max() / finite_p99.min() > 20:
        notes.append(f"band value scales differ by ~{finite_p99.max() / finite_p99.min():.0f}x "
                     "(e.g. 8-bit channels next to reflectance): per-band normalisation is mandatory.")
    if n_train < 1000:
        notes.append(f"small train set ({n_train}): prefer pretrained backbones and strong augmentation.")

    # --- save the stats file read by train.py ---
    stats_dir = Path(stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats = {"task": task_name, "split": "train", "n_samples": int(n_stats), "seed": seed,
             "bands": bands, "mean": [round(float(v), 6) for v in mean], "std": [round(float(v), 6) for v in std]}
    (stats_dir / f"{task_name}.json").write_text(json.dumps(stats, indent=2))

    card = {
        "task": task_name,
        "task_type": task_type,
        "sensor": _sensor(task),
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
        "example_sample": {"name": s0.sample_name, "shape_hwc": list(arr0.shape), "dtype": str(arr0.dtype),
                           "bands": sample_bands},
    }
    (stats_dir / f"{task_name}_card.json").write_text(json.dumps(card, indent=2))
    if verbose:
        print_card(card)
    return card


# ----------------------------------------------------------------------------------------------
# Readable summary
# ----------------------------------------------------------------------------------------------
def print_card(c: dict) -> None:
    line = "=" * 88
    print(f"\n{line}\n{c['task']}  [{c['task_type']}]  sensor: {c['sensor']}\n{line}")
    print(f"classes ({c['n_classes']}): {', '.join(c['class_names'])}")
    print(f"splits: train={c['split_sizes']['train']}  val={c['split_sizes']['val']}  (test never opened)")
    print(f"image size: {c['image_size'][0]}x{c['image_size'][1]}  |  resolution: {c['spatial_resolution_m']} m/pixel"
          f"  |  dtype: {', '.join(c['dtype'])}")
    print(f"label format: {c['label_format']}")
    if c["mask_unique_values"]:
        m = c["mask_unique_values"]
        print(f"mask values: {m['values']}  is_0_1={m['is_0_1']}  is_0_255={m['is_0_255']}  ignore={m['ignore_values']}")
    ex = c["example_sample"]
    print(f"\nexample sample '{ex['name']}': array shape (H, W, C) = {tuple(ex['shape_hwc'])}, {len(c['bands'])} bands")
    print(f"  {'band':<28}{'min':>10}{'max':>10}{'mean':>11}  |{'train mean':>12}{'train std':>11}{'p1':>9}{'p99':>9}")
    for i, b in enumerate(ex["bands"]):
        vr = c["value_range"][b["band"]]
        print(f"  {b['band']:<28}{b['min']:>10.1f}{b['max']:>10.1f}{b['mean']:>11.1f}  |"
              f"{c['band_mean'][i]:>12.2f}{c['band_std'][i]:>11.2f}{vr['p1']:>9.1f}{vr['p99']:>9.1f}")
    unit = "pixels" if c["task_type"] == "segmentation" else "samples"
    print(f"\nclass distribution on train ({unit}):")
    for name, d in c["class_distribution"].items():
        bar = "#" * int(round(d["pct"] / 2))
        print(f"  {name:<28}{d['count']:>12,}  {d['pct']:>7.2f}%  {bar}")
    print(f"\nstats computed on {c['stats_n_samples']} train images; nodata fraction {c['nodata_fraction']:.4%}; "
          f"constant bands: {c['constant_bands'] or 'none'}")
    print("notes:" if c["notes"] else "notes: none")
    for n in c["notes"]:
        print(f"  - {n}")


# ----------------------------------------------------------------------------------------------
# Image grid (generic)
# ----------------------------------------------------------------------------------------------
RGB_PATTERNS = {
    "red": [r"^red$", r"^b0?4$", r"^0?4 - red", r"^tci_r$"],
    "green": [r"^green$", r"^b0?3$", r"^0?3 - green", r"^tci_g$"],
    "blue": [r"^blue$", r"^b0?2$", r"^0?2 - blue", r"^tci_b$"],
}


def pick_rgb(bands: list[str], alt_names: list[tuple[str, ...]] | None = None) -> tuple[list[int], str | None]:
    """Indices of red/green/blue channels found BY NAME. Falls back to the first 3 bands with a warning."""
    alt_names = alt_names or [()] * len(bands)
    out = []
    for color, pats in RGB_PATTERNS.items():
        idx = None
        for pat in pats:
            for i, (b, alts) in enumerate(zip(bands, alt_names)):
                if any(re.search(pat, n.strip().lower()) for n in (b, *alts)):
                    idx = i
                    break
            if idx is not None:
                break
        out.append(idx)
    if None in out:
        warn = f"could not find red/green/blue by name in {bands}; showing the first 3 bands instead"
        n = len(bands)
        return [0, min(1, n - 1), min(2, n - 1)], warn
    return out, None


def _stretch(img: np.ndarray) -> np.ndarray:
    out = np.zeros(img.shape, dtype=np.float64)
    for c in range(img.shape[-1]):
        ch = img[..., c].astype(np.float64)
        lo, hi = np.nanpercentile(ch, 2), np.nanpercentile(ch, 98)
        out[..., c] = np.clip((ch - lo) / (hi - lo if hi > lo else 1.0), 0, 1)
    return np.nan_to_num(out)


def save_sample_grid(task_name: str, data_dir: str | None = None, n: int = 8, seed: int = 0,
                     fig_dir: str | Path = "docs/figures") -> Path:
    """Save a grid of n TRAIN samples: RGB (2-98% stretch) with class titles; mask overlay for segmentation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    task = load_task(task_name, data_dir)
    task_type = _task_type(task)
    n_classes = int(task.label_type.n_classes)
    class_names = _class_names(task, n_classes)
    ds = _open_split(task, "train")
    idx = np.random.default_rng(seed).choice(len(ds), size=min(n, len(ds)), replace=False)

    s0 = ds[int(idx[0])]
    _, raw_names, _ = _sample_array(s0)
    bands, fixed = _channel_names(task_name, raw_names)
    alts = [()] * len(bands) if fixed else [tuple(bi.alt_names) for bi in s0.band_info_list][: len(bands)]
    rgb, warn = pick_rgb(bands, alts)
    if warn:
        print(f"[warn] {task_name}: {warn}")

    seg = task_type == "segmentation"
    cols = 4
    per_row = cols // 2 if seg else cols
    rows = int(np.ceil(len(idx) / per_row))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.6 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    palette = plt.get_cmap("tab10" if n_classes <= 10 else "tab20")
    cmap = ListedColormap([palette(i % palette.N) for i in range(n_classes)])

    for k, i in enumerate(idx):
        s = ds[int(i)]
        arr, _, _ = _sample_array(s)
        img = _stretch(arr[..., rgb])
        lab = _label_value(s.label)
        if not seg:
            ax = axes.flat[k]
            ax.imshow(img)
            if task_type == "multilabel":
                title = ", ".join(class_names[c] for c in np.flatnonzero(np.asarray(lab))) or "(none)"
            else:
                title = class_names[int(lab)]
            ax.set_title(title if len(title) < 42 else title[:40] + "…", fontsize=8)
            continue
        mask = np.asarray(lab).squeeze().astype(np.int64)
        if mask.max() == 255 and set(np.unique(mask)) <= {0, 255}:
            mask = mask // 255
        r, c0 = divmod(k, per_row)
        a_img, a_ovl = axes[r, 2 * c0], axes[r, 2 * c0 + 1]
        a_img.imshow(img)
        a_img.set_title(f"train #{int(i)}", fontsize=8)
        a_ovl.imshow(img)
        if n_classes == 2:
            overlay = np.ma.masked_where(mask != 1, mask)
            a_ovl.imshow(overlay, cmap=ListedColormap([(1.0, 0.15, 0.1)]), alpha=0.55, interpolation="nearest")
        else:  # multi-class: show the mask itself (clearer than a blend); ignore values in light grey
            a_ovl.imshow(np.full(mask.shape + (3,), 0.85))
            valid = np.ma.masked_where((mask < 0) | (mask >= n_classes), mask)
            a_ovl.imshow(valid, cmap=cmap, vmin=0, vmax=n_classes - 1, interpolation="nearest")
        present = [class_names[v] for v in np.unique(mask) if 0 <= v < n_classes]
        frac = 100 * (mask == 1).mean() if n_classes == 2 else None
        t = f"{class_names[1]}: {frac:.1f}% px" if frac is not None else ", ".join(present)
        a_ovl.set_title(t if len(t) < 42 else t[:40] + "…", fontsize=8)

    rgb_names = ", ".join(bands[j] for j in rgb)
    fig.suptitle(f"{task_name} · {task_type} · 8 train samples · RGB = [{rgb_names}] · 2–98% stretch",
                 fontsize=10, y=0.995)
    if seg and n_classes > 2:
        fig.legend(handles=[Patch(color=cmap(c), label=class_names[c]) for c in range(n_classes)],
                   loc="lower center", ncol=min(n_classes, 7), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.04 if seg and n_classes > 2 else 0, 1, 0.97), h_pad=1.5)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"{task_name}_samples.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[ok] saved {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="GEO-Bench data reconnaissance (train + valid only).")
    ap.add_argument("tasks", nargs="+")
    ap.add_argument("--data-dir", default=None, help="GEO_BENCH_DIR root (default: $GEO_BENCH_DIR)")
    ap.add_argument("--max-stats-samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats-dir", default="stats")
    ap.add_argument("--fig-dir", default="docs/figures")
    ap.add_argument("--no-grid", action="store_true", help="skip the image grids")
    args = ap.parse_args(argv)
    for t in args.tasks:
        inspect_task(t, args.data_dir, args.max_stats_samples, args.seed, args.stats_dir)
        if not args.no_grid:
            save_sample_grid(t, args.data_dir, seed=args.seed, fig_dir=args.fig_dir)


if __name__ == "__main__":
    sys.exit(main())
