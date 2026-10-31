"""GEO-Bench data reconnaissance for GoTrain: readable summaries and figures.

    python scratch/explore.py m-eurosat m-pv4ger-seg [--data-dir data]

For each task this prints a readable summary, writes
    stats/<task>.json        train-split per-band mean/std (contract: BandStats)
    stats/<task>_card.json   the full recon card (describe_dataset() flattens it)
    docs/figures/<task>_samples.png   8 training samples
    docs/figures/<task>_per_label.png 6 training tiles per label (single-label classification only)

All measuring and loading logic lives in gotrain/engine/data.py (inspect_task, the errata tables
BAND_ORDER_FIXES / CLASS_NAME_FIXES, the guarded split opener). This script only prints and plots.
Only the TRAIN and VAL splits are ever opened.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a fresh clone too

from gotrain.engine.data import (  # noqa: E402
    channel_names as _channel_names,
    corrected_class_names as _corrected_class_names,
    inspect_task as _measure_task,
    label_value as _label_value,
    labels_only as _labels_only,
    load_task,
    open_split as _open_split,
    raw_class_names as _class_names,
    task_type as _task_type,
    sample_array as _sample_array,
)


def inspect_task(task_name: str, data_dir: str | None = None, max_stats_samples: int = 2000,
                 seed: int = 0, stats_dir: str | Path = "stats", verbose: bool = True) -> dict:
    """Measure one task (gotrain.engine.data.inspect_task) and print the summary."""
    card = _measure_task(task_name, data_dir, max_stats_samples, seed, stats_dir)
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
    class_names, _ = _corrected_class_names(task_name, _class_names(task, n_classes))
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


NIR_PATTERNS = [r"^nir$", r"^b0?8$", r"^0?8 - nir"]


def _find_band(bands: list[str], patterns: list[str]) -> int | None:
    for pat in patterns:
        for i, b in enumerate(bands):
            if re.search(pat, b.strip().lower()):
                return i
    return None


def _mean_ndvi(arr: np.ndarray, red: int, nir: int) -> float:
    r, n = arr[..., red].astype(np.float64), arr[..., nir].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (n - r) / (n + r)
    return float(np.nanmean(np.where(np.isfinite(ndvi), ndvi, np.nan)))


def save_label_grid(task_name: str, data_dir: str | None = None, per_label: int = 6, seed: int = 0,
                    max_ndvi_per_label: int = 200, fig_dir: str | Path = "docs/figures") -> Path | None:
    """Single-label classification only: one row per label index with `per_label` TRAIN tiles, titled with
    the class name and mean NDVI (if red and NIR bands exist). This is the evidence for CLASS_NAME_FIXES."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    task = load_task(task_name, data_dir)
    if _task_type(task) != "classification":
        return None
    n_classes = int(task.label_type.n_classes)
    raw_names = _class_names(task, n_classes)
    names, names_fixed = _corrected_class_names(task_name, raw_names)
    ds = _open_split(task, "train")
    by_label: dict[int, list[int]] = {c: [] for c in range(n_classes)}
    for i in range(len(ds)):
        by_label[int(_labels_only(ds, i))].append(i)

    _, raw_bands, _ = _sample_array(ds[0])
    bands, _ = _channel_names(task_name, raw_bands)
    rgb, _ = pick_rgb(bands)
    red, nir = _find_band(bands, RGB_PATTERNS["red"]), _find_band(bands, NIR_PATTERNS)
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(n_classes, per_label, figsize=(1.7 * per_label + 3.2, 1.75 * n_classes), squeeze=False)
    print(f"\nper-label check for {task_name} (train only):")
    for c in range(n_classes):
        members = by_label[c]
        show = rng.choice(members, size=min(per_label, len(members)), replace=False) if members else []
        ndvi_idx = rng.choice(members, size=min(max_ndvi_per_label, len(members)), replace=False) if members else []
        ndvi = None
        if red is not None and nir is not None and len(ndvi_idx):
            ndvi = float(np.mean([_mean_ndvi(_sample_array(ds[int(i)])[0], red, nir) for i in ndvi_idx]))
        for k in range(per_label):
            ax = axes[c, k]
            ax.set_xticks([])
            ax.set_yticks([])
            if k < len(show):
                ax.imshow(_stretch(_sample_array(ds[int(show[k])])[0][..., rgb]))
        label = f"{c}: {names[c]}"
        if names_fixed:
            label += f"\n(geobench: {raw_names[c]})"
        if ndvi is not None:
            label += f"\nNDVI {ndvi:+.2f}"
        axes[c, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8, labelpad=8)
        ndvi_txt = f"{ndvi:+.2f}" if ndvi is not None else "n/a"
        print(f"  label {c}: {names[c]:<26} n={len(members):<5} mean NDVI {ndvi_txt}")

    fig.suptitle(f"{task_name} · {per_label} train tiles per label index · RGB 2–98% stretch"
                 f"{' · NDVI on ≤' + str(max_ndvi_per_label) + ' tiles/label' if red is not None and nir is not None else ''}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"{task_name}_per_label.png"
    fig.savefig(out, dpi=100)
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
            save_label_grid(t, args.data_dir, seed=args.seed, fig_dir=args.fig_dir)


if __name__ == "__main__":
    sys.exit(main())
