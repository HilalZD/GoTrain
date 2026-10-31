"""diagnose(task, run_dir): how a finished run fails, from its saved validation predictions.

Return format (docs/gameplan-changes.md, same keys as FakeEngine.diagnose):
    task, metric_name, class_names,
    per_class         {class: {"recall"|"iou": v, ...}}  the first key is the headline metric
                      ("iou" for segmentation, "recall" otherwise; precision/f1 follow)
    confusion_matrix  rows = true class, row-normalised (fractions summing to 1 per row)
    curve_verdict     "overfitting" | "underfitting" | "fine"
    error_grid_path   PNG of the worst validation predictions, or None

Sources, all written by train(): run_dir/val_predictions.npz (best epoch), history.json and
checkpoints/best.ckpt (task type, class names, bands, normalisation). No model is re-run:
the images for the grid are read back from the VAL split only.

Confusion matrix per task type:
    classification  sample counts (true -> predicted)
    segmentation    pixel counts (true -> predicted), ignore pixels excluded
    multilabel      row i = images whose truth contains label i; entry j = fraction of those
                    images where label j was predicted (rows need not sum to 1, labels overlap)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

N_WORST = 8
RGB_PATTERNS = {
    "red": [r"^red$", r"^b0?4$", r"^tci_r$"],
    "green": [r"^green$", r"^b0?3$", r"^tci_g$"],
    "blue": [r"^blue$", r"^b0?2$", r"^tci_b$"],
}


# ---------------------------------------------------------------------------------------------
# Curve verdict
# ---------------------------------------------------------------------------------------------
def curve_verdict(val_scores: list[float], tolerance: float = 0.01) -> str:
    """'overfitting' if val peaked early and fell; 'underfitting' if it was still climbing.

    Same rule as FakeEngine so both engines speak the same language:
      overfitting:  best epoch in the first 75% of epochs AND last score < best - tolerance
      underfitting: fewer than 5 epochs, OR the score rose by > tolerance over the last 20%
      fine:         otherwise
    """
    n = len(val_scores)
    if n == 0:
        return "underfitting"
    best = max(val_scores)
    best_epoch = val_scores.index(best) + 1
    if best_epoch < 0.75 * n and val_scores[-1] < best - tolerance:
        return "overfitting"
    if n < 5 or val_scores[-1] - val_scores[max(int(0.8 * n) - 1, 0)] > tolerance:
        return "underfitting"
    return "fine"


# ---------------------------------------------------------------------------------------------
# Per-class metrics and confusion matrices
# ---------------------------------------------------------------------------------------------
def _row_normalise(counts: np.ndarray) -> list[list[float]]:
    rows = counts.sum(axis=1, keepdims=True).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.where(rows > 0, counts / rows, 0.0)
    return np.round(norm, 4).tolist()


def _prf(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> tuple:
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
    return recall, precision, f1


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, names: list[str]) -> tuple:
    n = len(names)
    counts = np.bincount(n * y_true.astype(int) + y_pred.astype(int), minlength=n * n)
    counts = counts.reshape(n, n)
    tp = np.diag(counts).astype(float)
    recall, precision, f1 = _prf(tp, counts.sum(0) - tp, counts.sum(1) - tp)
    per_class = {
        name: {
            "recall": round(float(r), 4),
            "precision": round(float(p), 4),
            "f1": round(float(f), 4),
        }
        for name, r, p, f in zip(names, recall, precision, f1, strict=True)
    }
    return per_class, _row_normalise(counts)


def segmentation_report(confusion: np.ndarray, names: list[str]) -> tuple:
    from gotrain.engine.train import iou_per_class

    iou = iou_per_class(confusion)
    tp = np.diag(confusion).astype(float)
    recall, precision, _ = _prf(tp, confusion.sum(0) - tp, confusion.sum(1) - tp)
    per_class = {
        name: {
            "iou": round(float(np.nan_to_num(i)), 4),
            "recall": round(float(r), 4),
            "precision": round(float(p), 4),
            "pixel_share": round(float(confusion[c].sum() / max(confusion.sum(), 1)), 4),
        }
        for c, (name, i, r, p) in enumerate(zip(names, iou, recall, precision, strict=True))
    }
    return per_class, _row_normalise(confusion)


def multilabel_report(y_true: np.ndarray, probs: np.ndarray, names: list[str]) -> tuple:
    true, pred = y_true > 0.5, probs > 0.5
    tp = (true & pred).sum(0).astype(float)
    recall, precision, f1 = _prf(tp, (~true & pred).sum(0), (true & ~pred).sum(0))
    per_class = {
        name: {
            "recall": round(float(r), 4),
            "precision": round(float(p), 4),
            "f1": round(float(f), 4),
        }
        for name, r, p, f in zip(names, recall, precision, f1, strict=True)
    }
    rows = []
    for i in range(len(names)):
        has_i = true[:, i]
        rows.append(
            np.round(pred[has_i].mean(0), 4).tolist() if has_i.any() else [0.0] * len(names)
        )
    return per_class, rows


# ---------------------------------------------------------------------------------------------
# Error grid
# ---------------------------------------------------------------------------------------------
def rgb_indices(bands: list[str]) -> list[int]:
    """Red/green/blue channels by name; falls back to the first three bands."""
    out = []
    for patterns in RGB_PATTERNS.values():
        hit = next(
            (i for p in patterns for i, b in enumerate(bands) if re.search(p, b.strip().lower())),
            None,
        )
        out.append(hit)
    if None in out:
        n = len(bands)
        return [0, min(1, n - 1), min(2, n - 1)]
    return out


def stretch(img: np.ndarray) -> np.ndarray:
    """Per-channel 2-98 percentile stretch to [0, 1] for display."""
    out = np.zeros(img.shape, dtype=np.float64)
    for c in range(img.shape[-1]):
        ch = img[..., c].astype(np.float64)
        lo, hi = np.nanpercentile(ch, 2), np.nanpercentile(ch, 98)
        out[..., c] = np.clip((ch - lo) / (hi - lo if hi > lo else 1.0), 0, 1)
    return np.nan_to_num(out)


def worst_indices(task_type: str, preds: dict, k: int = N_WORST) -> list[int]:
    """Indices of the k worst validation samples."""
    if task_type == "segmentation":
        return np.argsort(preds["image_score"])[:k].tolist()
    if task_type == "multilabel":
        wrong = ((preds["probs"] > 0.5) != (preds["y_true"] > 0.5)).sum(1)
        return np.argsort(-wrong, kind="stable")[:k].tolist()
    true_prob = preds["probs"][np.arange(len(preds["y_true"])), preds["y_true"].astype(int)]
    return np.argsort(true_prob)[:k].tolist()  # confidently wrong first


def save_error_grid(path: Path, ckpt: dict, preds: dict, idx: list[int]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gotrain.engine.train import checkpoint_dataset

    ds = checkpoint_dataset(ckpt, "val")
    names = ckpt["class_names"]
    # ds.raw() returns the bands ON DISK (ds.all_bands). The model's own bands may include
    # zero-filled ones the task lacks (ssl4eo_s2 on m-brick-kiln: B06, B09, B10), so pick RGB here.
    rgb = rgb_indices(ds.all_bands)
    seg = ckpt["task_type"] == "segmentation"
    cols = 3 if seg else 4
    rows = len(idx) if seg else int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.2 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, i in enumerate(idx):
        arr, label = ds.raw(i)
        img = stretch(arr[..., rgb])
        if seg:
            truth = np.asarray(label).squeeze()
            vmax = max(len(names) - 1, 1)
            panels = [
                (img, f"val #{i} image"),
                (truth, "truth"),
                (preds["pred_masks"][i], f"prediction (mIoU {preds['image_score'][i]:.2f})"),
            ]
            for ax, (data, title) in zip(axes[k], panels, strict=True):
                kw = {} if data.ndim == 3 else {"cmap": "viridis", "vmin": 0, "vmax": vmax}
                ax.imshow(data, interpolation="nearest", **kw)
                ax.set_title(title, fontsize=8)
            continue
        ax = axes.flat[k]
        ax.imshow(img)
        if ckpt["task_type"] == "multilabel":
            t = ", ".join(names[j] for j in np.flatnonzero(preds["y_true"][i] > 0.5)) or "-"
            p = ", ".join(names[j] for j in np.flatnonzero(preds["probs"][i] > 0.5)) or "-"
            title = f"true: {t[:30]}\npred: {p[:30]}"
        else:
            yt, yp = int(preds["y_true"][i]), int(preds["y_pred"][i])
            title = f"true: {names[yt]}\npred: {names[yp]} ({preds['probs'][i][yp]:.2f})"
        ax.set_title(title, fontsize=8)
    head = "image | truth | prediction" if seg else "true vs predicted class"
    fig.suptitle(f"{ckpt['task']} · worst {len(idx)} validation predictions · {head}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------------------------
# Entry point (re-exported from gotrain.engine.train)
# ---------------------------------------------------------------------------------------------
def diagnose(task: str, run_dir: Path) -> dict:
    import torch

    run_dir = Path(run_dir)
    ckpt_path = run_dir / "checkpoints" / "best.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"no checkpoint in {run_dir}; only finished runs have one")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if ckpt["task"] != task:
        raise ValueError(f"run is for {ckpt['task']!r}, not {task!r}")
    preds = dict(np.load(run_dir / "val_predictions.npz"))
    names = ckpt["class_names"]
    kind = ckpt["task_type"]
    if kind == "segmentation":
        per_class, matrix = segmentation_report(preds["confusion"], names)
    elif kind == "multilabel":
        per_class, matrix = multilabel_report(preds["y_true"], preds["probs"], names)
    else:
        per_class, matrix = classification_report(preds["y_true"], preds["y_pred"], names)
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    scores = [h["val_score"] for h in history if h.get("val_score") is not None]
    grid = None
    try:
        grid = str(save_error_grid(run_dir / "errors.png", ckpt, preds, worst_indices(kind, preds)))
    except Exception as exc:  # the numbers are still useful without the picture
        (run_dir / "errors_failed.txt").write_text(f"{type(exc).__name__}: {exc}", "utf-8")
    from gotrain.engine.train import nodata_extras

    extras = nodata_extras(task, preds.get("confusion")) if kind == "segmentation" else {}
    return {
        **extras,  # e.g. m-chesapeake: miou_excl_nodata (extra; NOT the official score)
        "task": task,
        "metric_name": ckpt["metric_name"],
        "class_names": names,
        "per_class": per_class,
        "confusion_matrix": matrix,
        "curve_verdict": curve_verdict(scores),
        "error_grid_path": grid,
    }
