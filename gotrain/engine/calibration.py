"""Confidence flags: temperature calibration and a "needs review" threshold, fitted on VALIDATION.

    calibrate(task, run_dir) -> dict, also saved to run_dir/calibration.json (+ reliability.png)

1. Temperature scaling. One number T rescales the model's logits, softmax(logits / T), so that
   its confidence matches how often it is right (T > 1 softens over-confident models). The
   logits are log-probabilities: softmax(log p / T) == p^(1/T) renormalised, and for a binary
   U-Net with one sigmoid channel, log([1 - p, p]) gives exactly sigmoid(z / T). T minimises
   the validation negative log-likelihood (NLL). ECE (15 bins) and NLL are reported before/after.
2. Threshold. Classification: the lowest confidence t such that predictions with confidence
   >= t make <= 1% errors, i.e. the largest coverage at that error rate. Segmentation: the same
   rule per pixel, plus a tile-level needs_review flag when more than 20% of the pixels
   predicted as a target class (every class except the most common one) are uncertain.

All numbers are measured on the VALIDATION split, where T and the thresholds were fitted, so
they are optimistic for new data; the output says so. CPU only. The test split is never read.

Sources: classification uses the validation probabilities train() saved
(run_dir/val_predictions.npz). Segmentation predictions are not saved per pixel (too large),
so they are recomputed from run_dir/checkpoints/best.ckpt on the validation split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

ECE_BINS = 15
MAX_ERROR = 0.01
TILE_UNCERTAIN_SHARE = 0.20
PIXELS_PER_TILE = 2000  # pixels sampled per val tile to fit T and the pixel threshold
CURVE_POINTS = 50
EPS = 1e-7
VALIDATION_NOTE = (
    "Measured on the VALIDATION split, where the temperature and the threshold were fitted: "
    "expect slightly worse coverage/accuracy on new data."
)


# ---------------------------------------------------------------------------------------------
# Pure helpers (numpy only)
# ---------------------------------------------------------------------------------------------
def log_probs(probs: np.ndarray) -> np.ndarray:
    """Class probabilities [N, K] -> logits usable for temperature scaling."""
    return np.log(np.clip(probs, EPS, 1.0))


def softmax_t(logits: np.ndarray, t: float) -> np.ndarray:
    z = logits / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, y: np.ndarray, t: float = 1.0) -> float:
    p = softmax_t(logits, t)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0))))


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = ECE_BINS) -> float:
    """Expected calibration error: |accuracy - confidence| per bin, weighted by bin size."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def reliability(conf: np.ndarray, correct: np.ndarray, bins: int = ECE_BINS) -> dict:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    out = {"bin_conf": [], "bin_acc": [], "bin_count": []}
    for b in range(bins):
        m = idx == b
        out["bin_conf"].append(round(float(conf[m].mean()), 4) if m.any() else None)
        out["bin_acc"].append(round(float(correct[m].mean()), 4) if m.any() else None)
        out["bin_count"].append(int(m.sum()))
    return out


def fit_temperature(logits: np.ndarray, y: np.ndarray, lo: float = 0.05, hi: float = 20.0) -> float:
    """T minimising NLL: golden-section search on log T (NLL is smooth and unimodal in T)."""
    a, b = np.log(lo), np.log(hi)
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(logits, y, np.exp(c)), nll(logits, y, np.exp(d))
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll(logits, y, np.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll(logits, y, np.exp(d))
    return float(np.exp((a + b) / 2))


def choose_threshold(conf: np.ndarray, correct: np.ndarray, max_error: float = MAX_ERROR) -> dict:
    """Largest coverage whose most-confident predictions have error <= max_error."""
    order = np.argsort(-conf, kind="stable")
    c_sorted, ok_sorted = conf[order], correct[order].astype(float)
    k = np.arange(1, len(conf) + 1)
    err = 1 - np.cumsum(ok_sorted) / k
    valid = np.flatnonzero(err <= max_error)
    if len(valid) == 0:
        return {
            "confidence": 1.0,
            "coverage": 0.0,
            "confident_accuracy": None,
            "flagged_pct": 100.0,
            "max_error": max_error,
        }
    n = int(valid[-1]) + 1
    t = float(c_sorted[n - 1])
    keep = conf >= t  # ties at t are all kept, as at prediction time
    return {
        "confidence": t,  # unrounded: rounding up would flag the boundary prediction
        "coverage": round(float(keep.mean()), 4),
        "confident_accuracy": round(float(correct[keep].mean()), 4),
        "flagged_pct": round(100 * float(1 - keep.mean()), 2),
        "max_error": max_error,
    }


def coverage_curve(conf: np.ndarray, correct: np.ndarray, points: int = CURVE_POINTS) -> dict:
    """Accuracy of the most-confident fraction, for coverages from 1/points to 1."""
    order = np.argsort(-conf, kind="stable")
    acc = np.cumsum(correct[order].astype(float)) / np.arange(1, len(conf) + 1)
    cov = np.linspace(1 / points, 1, points)
    at = np.clip((cov * len(conf)).astype(int) - 1, 0, len(conf) - 1)
    return {
        "coverage": [round(float(c), 4) for c in cov],
        "accuracy": [round(float(acc[i]), 4) for i in at],
    }


def tile_needs_review(
    conf: np.ndarray,
    pred: np.ndarray,
    threshold: float,
    background: int,
    share: float = TILE_UNCERTAIN_SHARE,
) -> bool:
    """True when more than `share` of the tile's target-class pixels are below the threshold."""
    target = pred != background
    if not target.any():
        return False
    return bool(np.mean(conf[target] < threshold) > share)


# ---------------------------------------------------------------------------------------------
# calibrate()
# ---------------------------------------------------------------------------------------------
def _load(run_dir: Path) -> tuple[dict, Any]:
    import torch

    from gotrain.engine.train import load_checkpoint

    ckpt_path = run_dir / "checkpoints" / "best.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"no checkpoint in {run_dir}; only finished runs can be calibrated")
    return load_checkpoint(ckpt_path, torch.device("cpu"))


def _seg_pass(
    ckpt: dict,
    model: Any,
    rng: np.random.Generator | None,
    t: float | None,
    threshold: float | None,
    background: int | None,
) -> dict:
    """One pass over val. Sampling pass (t None): logits + labels of sampled pixels.
    Flag pass: per-tile needs_review and mIoU, at temperature t and pixel threshold."""
    import torch
    from torch.utils.data import DataLoader

    from gotrain.engine.data import IGNORE_INDEX
    from gotrain.engine.train import checkpoint_dataset, class_probs, miou_from_confusion

    ds = checkpoint_dataset(ckpt, "val")
    k = ckpt["n_classes"]
    out: dict[str, list] = {"logits": [], "y": [], "flag": [], "tile_miou": []}
    model.eval()
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=8, num_workers=0):
            probs = class_probs(model, x, ckpt["config"]["img_size"], ckpt["task_type"]).numpy()
            for p_img, y_img in zip(probs, y.numpy(), strict=True):
                lg = log_probs(p_img.reshape(k, -1).T)  # [H*W, K]
                lab = y_img.reshape(-1)
                valid = np.flatnonzero(lab != IGNORE_INDEX)
                if t is None:
                    pick = rng.choice(valid, size=min(PIXELS_PER_TILE, len(valid)), replace=False)
                    out["logits"].append(lg[pick])
                    out["y"].append(lab[pick])
                    continue
                cal = softmax_t(lg, t)
                conf, pred = cal.max(axis=1), cal.argmax(axis=1)
                out["flag"].append(tile_needs_review(conf, pred, threshold, background))
                cm = np.bincount(k * lab[valid] + pred[valid], minlength=k * k).reshape(k, k)
                out["tile_miou"].append(miou_from_confusion(cm))
    return out


def save_reliability_png(path: Path, before: dict, after: dict, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.plot([0, 1], [0, 1], color="#898781", lw=1, label="perfect calibration")
    for rel, color, name in ((before, "#b9b8b1", "before"), (after, "#2a78d6", "after (T)")):
        pts = [
            (c, a) for c, a in zip(rel["bin_conf"], rel["bin_acc"], strict=True) if c is not None
        ]
        ax.plot(*zip(*pts, strict=True), marker="o", ms=5, lw=2, color=color, label=name)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def calibrate(task: str, run_dir: Path) -> dict:
    """Fit T and the needs-review threshold on VALIDATION; save calibration.json + a PNG."""
    run_dir = Path(run_dir)
    ckpt, model = _load(run_dir)
    if ckpt["task"] != task:
        raise ValueError(f"run is for {ckpt['task']!r}, not {task!r}")
    kind = ckpt["task_type"]
    if kind == "multilabel":
        raise ValueError("calibration supports classification and segmentation, not multilabel")
    result: dict[str, Any] = {
        "task": task,
        "task_type": kind,
        "fitted_on": "val",
        "note": VALIDATION_NOTE,
    }
    if kind == "classification":
        preds = np.load(run_dir / "val_predictions.npz")
        logits, y = log_probs(preds["probs"]), preds["y_true"].astype(int)
    else:
        sample = _seg_pass(ckpt, model, np.random.default_rng(0), None, None, None)
        logits, y = np.concatenate(sample["logits"]), np.concatenate(sample["y"]).astype(int)
    t = fit_temperature(logits, y)
    p0, p1 = softmax_t(logits, 1.0), softmax_t(logits, t)
    correct = p1.argmax(1) == y  # argmax is unchanged by T
    conf0, conf1 = p0.max(1), p1.max(1)
    threshold = choose_threshold(conf1, correct)
    result |= {
        "temperature": round(t, 4),
        "n_val": int(len(y)) if kind == "classification" else int(len(sample["y"])),
        "ece_before": round(ece(conf0, correct), 4),
        "ece_after": round(ece(conf1, correct), 4),
        "nll_before": round(nll(logits, y, 1.0), 4),
        "nll_after": round(nll(logits, y, t), 4),
        "accuracy": round(float(correct.mean()), 4),
        "threshold": threshold,
        "curve": coverage_curve(conf1, correct),
        "reliability": {
            "before": reliability(conf0, correct),
            "after": reliability(conf1, correct),
        },
    }
    if kind == "segmentation":
        result["unit"] = "pixels (sampled from every val tile)"
        result["n_pixels_sampled"] = int(len(y))
        background = int(np.bincount(y, minlength=ckpt["n_classes"]).argmax())
        flags = _seg_pass(ckpt, model, None, t, threshold["confidence"], background)
        flagged = np.asarray(flags["flag"], dtype=bool)
        tiles = np.asarray(flags["tile_miou"])
        result["tiles"] = {
            "rule": (
                f"needs_review when > {TILE_UNCERTAIN_SHARE:.0%} of the pixels predicted as "
                f"a target class (not '{ckpt['class_names'][background]}') are below the "
                "pixel threshold"
            ),
            "background_class": ckpt["class_names"][background],
            "flagged_pct": round(100 * float(flagged.mean()), 2),
            "mean_tile_miou_flagged": round(float(tiles[flagged].mean()), 4)
            if flagged.any()
            else None,
            "mean_tile_miou_unflagged": round(float(tiles[~flagged].mean()), 4)
            if (~flagged).any()
            else None,
        }
        result["threshold"]["background_class"] = background
    else:
        result["unit"] = "images"
    (run_dir / "calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    save_reliability_png(
        run_dir / "reliability.png",
        result["reliability"]["before"],
        result["reliability"]["after"],
        f"{task} · T = {t:.2f} · ECE {result['ece_before']:.3f} -> {result['ece_after']:.3f} (val)",
    )
    return result


# ---------------------------------------------------------------------------------------------
# predict() extras
# ---------------------------------------------------------------------------------------------
def confidence_extras(checkpoint_path: Path, probs: np.ndarray, task_type: str) -> dict:
    """{"confidence", "needs_review"} if the checkpoint's run has calibration.json, else {}.

    classification: confidence float32 [N], needs_review bool [N]
    segmentation:   confidence float32 [N, H, W], needs_review bool [N, H, W] (per pixel)
    """
    path = Path(checkpoint_path).parent.parent / "calibration.json"
    if not path.exists() or task_type not in ("classification", "segmentation"):
        return {}
    cal = json.loads(path.read_text(encoding="utf-8"))
    k = probs.shape[1]
    lg = log_probs(np.moveaxis(probs, 1, -1).reshape(-1, k))
    conf = softmax_t(lg, cal["temperature"]).max(axis=1)
    shape = (probs.shape[0],) + probs.shape[2:]
    conf = conf.reshape(shape).astype(np.float32)
    return {"confidence": conf, "needs_review": conf < cal["threshold"]["confidence"]}
