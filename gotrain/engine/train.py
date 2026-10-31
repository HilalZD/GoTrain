"""Config-driven training engine: train(), evaluate() (and diagnose(), re-exported).

train(config, budget_minutes, run_dir) -> TrainResult, scored on VALIDATION.

How one trial runs:
1. Check the config against the task's DatasetCard (loss fits the task type, bands exist,
   backbone/weights combination is buildable, U-Net input size divisible by 32). A bad config
   returns status "error" with a hint, before any heavy work.
2. Build the datasets (train subset via train_subset_frac; the FULL, fixed val set) and the
   model (gotrain/engine/models.py).
3. Train with AdamW. The learning rate follows a warm-up + cosine curve whose progress is
   max(time used / time available, steps done / max_epochs steps): the schedule always
   finishes, whether the budget or max_epochs runs out first. Time for the last validation
   and checkpoint is reserved, so the run stops cleanly inside its budget.
4. Validate after every epoch on the full val set; keep the best checkpoint.
5. Write run_dir/{history.json, metrics.json, checkpoints/best.ckpt, val_predictions.npz}
   (+ config.json if the caller has not written one).

Status: "ok" = at least one full epoch was trained and validated (stopping at the budget is
the normal end of a time-budgeted trial); "timeout" = the budget ran out inside the first
epoch: empty history, no checkpoint (the tools fail fast on 0-epoch timeouts); "error" =
anything else, with a message. The first batch always runs, and the time reserved for the
final validation is estimated without warm-up batches and capped at half the training window.

history.json rows = EpochRecord fields + train_s (training time of that epoch, measured
before validation) and val_s (validation time), for planning time budgets.
train() never raises.

Metrics: accuracy (classification), micro-F1 "f1" (multilabel), mIoU "miou" (segmentation).
"""

from __future__ import annotations

import contextlib
import json
import math
import random
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from gotrain.contracts import DatasetCard, EpochRecord, TrainConfig, TrainResult
from gotrain.engine.combos import check_combo
from gotrain.engine.data import (
    IGNORE_INDEX,
    NODATA_CLASSES,
    TEST_SPLIT,
    GeoBenchDataset,
    describe_dataset,
    subset_indices,
)
from gotrain.engine.interface import load_band_stats
from gotrain.engine.models import UNET_STRIDE, ModelSpec, build_model, ssl4eo_inputs
from gotrain.engine.models import input_normalisation as model_normalisation

METRIC = {"classification": "accuracy", "multilabel": "f1", "segmentation": "miou"}
LOSSES = {
    "classification": ("ce", "focal", "bce"),
    "multilabel": ("bce", "focal"),
    "segmentation": ("ce", "dice_ce", "focal", "bce", "dice_bce"),
}
BINARY_SEG_LOSSES = ("bce", "dice_bce")  # one logit per pixel; needs exactly 2 classes
WARMUP_FRACTION = 0.05
SAVE_MARGIN_S = 3.0
RESERVE_CAP = 0.5  # the validation reserve never takes more than half the training window
VAL_COST_RATIO = 0.5  # before the first validation: a val sample ~ half a train sample's cost
CKPT_NAME = "checkpoints/best.ckpt"
EVAL_BATCH = 32


# ---------------------------------------------------------------------------------------------
# Config checks (cheap, before any data or model is loaded)
# ---------------------------------------------------------------------------------------------
def selected_bands(config: TrainConfig, card: DatasetCard) -> list[str]:
    return list(card.bands) if config.bands == "all" else list(config.bands)


def n_outputs(card: DatasetCard, loss: str) -> int:
    if card.task_type == "segmentation" and loss in BINARY_SEG_LOSSES:
        return 1
    return card.n_classes


def config_problem(config: TrainConfig, card: DatasetCard) -> str | None:
    """Why this config can't run on this task (a hint the agent can act on), or None."""
    allowed = LOSSES[card.task_type]
    if config.loss not in allowed:
        return f"loss {config.loss!r} does not fit a {card.task_type} task; use one of {allowed}"
    if card.task_type == "segmentation" and config.loss in BINARY_SEG_LOSSES:
        if card.n_classes != 2:
            return (
                f"loss {config.loss!r} is for binary masks but {card.task} has "
                f"{card.n_classes} classes; use 'dice_ce', 'ce' or 'focal'"
            )
    if config.bands != "all":
        unknown = [b for b in config.bands if b not in card.bands]
        if unknown:
            return f"bands {unknown} not in {card.task}; available: {card.bands}"
    if card.task_type == "segmentation" and config.img_size % UNET_STRIDE:
        return f"segmentation img_size must be a multiple of {UNET_STRIDE}, got {config.img_size}"
    combo = check_combo(card, config.backbone, config.weights)  # Hilal's single source of truth
    if combo:
        return combo
    if config.weights == "ssl4eo_s2":
        inputs, zero, _ = ssl4eo_inputs(selected_bands(config, card))
        if len(zero) == len(inputs):
            return "weights 'ssl4eo_s2' need at least one Sentinel-2 band (B01..B12, B8A)"
    return None


def model_spec(config: TrainConfig, card: DatasetCard, stats: Any) -> tuple[ModelSpec, list]:
    """The model's input spec and the selected bands it leaves out (ssl4eo_s2 only).

    imagenet/none: the selected bands, z-scored with the TRAIN stats.
    ssl4eo_s2:     always the 13 SSL4EO bands in TorchGeo order; missing ones zero-filled.
    """
    selected = selected_bands(config, card)
    zero, dropped = [], []
    bands = selected
    if config.weights == "ssl4eo_s2":
        bands, zero, dropped = ssl4eo_inputs(selected)
    present = [b for b in bands if b not in zero]
    mean = {b: stats.mean[stats.bands.index(b)] for b in present}
    std = {b: stats.std[stats.bands.index(b)] for b in present}
    spec = ModelSpec(
        task_type=card.task_type,
        backbone=config.backbone,
        weights=config.weights,
        bands=bands,
        n_outputs=n_outputs(card, config.loss),
        band_mean=[mean.get(b, 0.0) for b in bands],
        band_std=[std.get(b, 1.0) for b in bands],
        zero_fill=zero,
    )
    return spec, dropped


def amp(device: Any) -> Any:
    """fp16 mixed precision on CUDA (the T4 has no bf16); full precision on CPU."""
    import torch

    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


# ---------------------------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------------------------
def _soft_dice(probs: Any, onehot: Any, valid: Any, eps: float = 1.0) -> Any:
    """1 - mean Dice over classes. probs/onehot: [B, C, H, W]; valid: [B, 1, H, W]."""
    p, t = probs * valid, onehot * valid
    inter = (p * t).sum(dim=(0, 2, 3))
    denom = p.sum(dim=(0, 2, 3)) + t.sum(dim=(0, 2, 3))
    return 1 - ((2 * inter + eps) / (denom + eps)).mean()


def compute_loss(name: str, task_type: str, logits: Any, y: Any, n_classes: int) -> Any:
    import torch
    import torch.nn.functional as F
    from torchvision.ops import sigmoid_focal_loss

    logits = logits.float()
    if task_type == "multilabel":
        if name == "focal":
            return sigmoid_focal_loss(logits, y, reduction="mean")
        return F.binary_cross_entropy_with_logits(logits, y)
    if task_type == "classification":
        if name == "bce":
            return F.binary_cross_entropy_with_logits(logits, F.one_hot(y, n_classes).float())
        ce = F.cross_entropy(logits, y, reduction="none")
        if name == "focal":
            return ((1 - torch.exp(-ce)) ** 2 * ce).mean()
        return ce.mean()
    # segmentation; y: [B, H, W] with IGNORE_INDEX on pixels to skip
    valid = y != IGNORE_INDEX
    if logits.shape[1] == 1:  # binary: bce / dice_bce
        t = y.clamp(min=0).float().unsqueeze(1)
        v = valid.unsqueeze(1).float()
        bce = (F.binary_cross_entropy_with_logits(logits, t, reduction="none") * v).sum()
        bce = bce / v.sum().clamp(min=1)
        if name == "bce":
            return bce
        return bce + _soft_dice(torch.sigmoid(logits), t, v)
    ce = F.cross_entropy(logits, y, ignore_index=IGNORE_INDEX, reduction="none")
    if name == "focal":
        return ((1 - torch.exp(-ce)) ** 2 * ce)[valid].mean()
    ce_mean = ce[valid].mean() if valid.any() else ce.sum() * 0
    if name == "ce":
        return ce_mean
    onehot = F.one_hot(y.clamp(min=0), n_classes).permute(0, 3, 1, 2).float()
    return ce_mean + _soft_dice(logits.softmax(1), onehot, valid.unsqueeze(1).float())


# ---------------------------------------------------------------------------------------------
# Batch transforms (on the device)
# ---------------------------------------------------------------------------------------------
def resize(x: Any, size: int) -> Any:
    import torch.nn.functional as F

    if x.shape[-1] == size and x.shape[-2] == size:
        return x
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


def resize_mask(y: Any, size: int) -> Any:
    import torch.nn.functional as F

    if y.shape[-1] == size and y.shape[-2] == size:
        return y
    return F.interpolate(y[:, None].float(), size=(size, size), mode="nearest")[:, 0].long()


def augment(x: Any, y: Any, ops: list[str], segmentation: bool) -> tuple[Any, Any]:
    """Per-sample flips / rot90 (applied to masks too) and brightness/contrast jitter."""
    import torch

    if not ops:
        return x, y
    xs, ys = [], []
    square = x.shape[-1] == x.shape[-2]
    for xi, yi in zip(x, y, strict=True):
        if "flip" in ops:
            if torch.rand(()) < 0.5:
                xi = xi.flip(-1)
                yi = yi.flip(-1) if segmentation else yi
            if torch.rand(()) < 0.5:
                xi = xi.flip(-2)
                yi = yi.flip(-2) if segmentation else yi
        if "rot90" in ops and square:
            k = int(torch.randint(4, ()))
            xi = torch.rot90(xi, k, dims=(-2, -1))
            yi = torch.rot90(yi, k, dims=(-2, -1)) if segmentation else yi
        if "color_jitter" in ops:  # inputs are normalised, so jitter in normalised units
            xi = xi * (0.9 + 0.2 * torch.rand(())) + (torch.rand(()) - 0.5) * 0.2
        xs.append(xi)
        ys.append(yi)
    return torch.stack(xs), torch.stack(ys)


# ---------------------------------------------------------------------------------------------
# Inference and metrics (shared by validation, evaluate() and diagnose())
# ---------------------------------------------------------------------------------------------
def _tta_views(x: Any) -> list[tuple[Any, Any]]:
    """(view, undo) pairs: 4 rotations x {no flip, horizontal flip} = the 8 dihedral views."""
    import torch

    views = []
    for k in range(4):
        for flip in (False, True):

            def fwd(t, k=k, flip=flip):
                t = t.flip(-1) if flip else t
                return torch.rot90(t, k, dims=(-2, -1))

            def back(t, k=k, flip=flip):
                t = torch.rot90(t, -k, dims=(-2, -1))
                return t.flip(-1) if flip else t

            views.append((fwd, back))
    return views


def class_probs(model: Any, x: Any, img_size: int, task_type: str, tta: bool = False) -> Any:
    """Class probabilities for a batch of model inputs (already normalised).

    classification: softmax, [N, K]          multilabel: sigmoid per class, [N, K]
    segmentation:   [N, K, H, W] at the input's native size; a binary U-Net with one output
                    channel p becomes the two classes [1 - p, p].
    tta=True averages PROBABILITIES over the 8 flip/rot90 views; segmentation outputs are
    rotated/flipped back before averaging. Every image is processed independently of the
    rest of the batch (the model must be in eval mode).
    """
    import torch
    import torch.nn.functional as F

    native = x.shape[-2:]
    views = _tta_views(x) if tta and x.shape[-1] == x.shape[-2] else [(None, None)]
    total = None
    for fwd, back in views:
        xv = fwd(x) if fwd else x
        with amp(x.device):
            logits = model(resize(xv, img_size)).float()
        if task_type == "segmentation":
            if logits.shape[-2:] != native:
                logits = F.interpolate(logits, size=native, mode="bilinear", align_corners=False)
            if logits.shape[1] == 1:
                p1 = torch.sigmoid(logits)
                probs = torch.cat([1 - p1, p1], dim=1)
            else:
                probs = logits.softmax(1)
            probs = back(probs) if back else probs
        elif task_type == "multilabel":
            probs = torch.sigmoid(logits)
        else:
            probs = logits.softmax(1)
        total = probs if total is None else total + probs
    return total / len(views)


def run_inference(
    model: Any,
    loader: Any,
    task_type: str,
    n_classes: int,
    img_size: int,
    device: Any,
    tta: bool = False,
) -> dict:
    """Predictions over a whole loader, as numpy arrays.

    classification: y_true [N], y_pred [N], probs [N, C]
    multilabel:     y_true [N, C], probs [N, C]
    segmentation:   confusion [C, C] pixel counts (rows = true), pred_masks [N, H, W] uint8,
                    image_score [N] (mean IoU over classes present in that image)
    """
    import torch

    model.eval()
    seg = task_type == "segmentation"
    out: dict[str, list] = {"y_true": [], "y_pred": [], "probs": [], "image_score": []}
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            probs_t = class_probs(model, x, img_size, task_type, tta)
            if task_type == "classification":
                probs = probs_t.cpu().numpy()
                out["probs"].append(probs)
                out["y_pred"].append(probs.argmax(1))
                out["y_true"].append(y.numpy())
            elif task_type == "multilabel":
                out["probs"].append(probs_t.cpu().numpy())
                out["y_true"].append(y.numpy())
            else:
                pred, yt = probs_t.argmax(1).cpu().numpy(), y.numpy()
                for p_img, t_img in zip(pred, yt, strict=True):
                    valid = t_img != IGNORE_INDEX
                    cm = np.bincount(
                        n_classes * t_img[valid] + p_img[valid], minlength=n_classes**2
                    ).reshape(n_classes, n_classes)
                    confusion += cm
                    out["image_score"].append(miou_from_confusion(cm))
                out["y_pred"].append(pred.astype(np.uint8))
    if seg:
        return {
            "confusion": confusion,
            "pred_masks": np.concatenate(out["y_pred"]) if out["y_pred"] else np.zeros(0),
            "image_score": np.asarray(out["image_score"], dtype=np.float32),
        }
    res = {"y_true": np.concatenate(out["y_true"]), "probs": np.concatenate(out["probs"])}
    if task_type == "classification":
        res["y_pred"] = np.concatenate(out["y_pred"])
    return res


def iou_per_class(confusion: np.ndarray) -> np.ndarray:
    """IoU per class; NaN where the class is absent from both truth and prediction."""
    tp = np.diag(confusion).astype(np.float64)
    union = confusion.sum(0) + confusion.sum(1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, tp / union, np.nan)


def miou_from_confusion(confusion: np.ndarray) -> float:
    iou = iou_per_class(confusion)
    return float(np.nanmean(iou)) if np.isfinite(iou).any() else 0.0


def miou_excluding(confusion: np.ndarray, nodata: int) -> float:
    """Extra metric: mIoU over the other classes, with pixels whose TRUTH is `nodata` removed.

    Pixels of real classes predicted as `nodata` still count as errors for their true class.
    Not the official score (that keeps every class); reported next to it as miou_excl_nodata.
    """
    keep = [c for c in range(confusion.shape[0]) if c != nodata]
    cm = np.delete(confusion, nodata, axis=0)  # drop no-data truth rows
    tp = np.array([cm[i, c] for i, c in enumerate(keep)], dtype=np.float64)
    union = cm.sum(axis=1) + cm[:, keep].sum(axis=0) - tp
    present = union > 0
    return float(np.mean(tp[present] / union[present])) if present.any() else 0.0


def nodata_extras(task: str, confusion: np.ndarray | None) -> dict:
    """{"miou_excl_nodata", "nodata_class"} for tasks with a no-data class, else {}."""
    nodata = NODATA_CLASSES.get(task)
    if nodata is None or confusion is None:
        return {}
    return {
        "miou_excl_nodata": round(miou_excluding(confusion, nodata), 5),
        "nodata_class": nodata,
    }


def micro_f1(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> float:
    pred = probs > threshold
    true = y_true > 0.5
    tp = np.sum(pred & true)
    fp = np.sum(pred & ~true)
    fn = np.sum(~pred & true)
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def score(task_type: str, preds: dict) -> float:
    if task_type == "classification":
        return float(np.mean(preds["y_pred"] == preds["y_true"]))
    if task_type == "multilabel":
        return micro_f1(preds["y_true"], preds["probs"])
    return miou_from_confusion(preds["confusion"])


# ---------------------------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------------------------
def save_checkpoint(
    path: Path,
    model: Any,
    config: TrainConfig,
    card: DatasetCard,
    spec: ModelSpec,
    norm: tuple,
    val_score: float,
    epoch: int,
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": 1,
        "task": card.task,
        "task_type": card.task_type,
        "n_classes": card.n_classes,
        "class_names": list(card.class_names),
        "metric_name": METRIC[card.task_type],
        "config": config.model_dump(mode="json"),
        "spec": asdict(spec),
        "norm_mean": [float(v) for v in norm[0]],
        "norm_std": [float(v) for v in norm[1]],
        "val_score": float(val_score),
        "epoch": epoch,
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, device: Any) -> tuple[dict, Any]:
    """(checkpoint dict, model with its weights) without downloading pretrained weights."""
    import torch

    ckpt = torch.load(Path(path), map_location="cpu", weights_only=True)
    spec = ModelSpec(**{**ckpt["spec"], "weights": "none"})  # weights come from the file
    model = build_model(spec)
    model.load_state_dict(ckpt["model"])
    return ckpt, model.to(device)


def checkpoint_dataset(ckpt: dict, split: str, allow_test: bool = False) -> GeoBenchDataset:
    return GeoBenchDataset(
        ckpt["task"],
        split,
        bands=ckpt["spec"]["bands"],
        norm=(ckpt["norm_mean"], ckpt["norm_std"]),
        zero_fill=ckpt["spec"].get("zero_fill", []),
        allow_test=allow_test,
    )


# ---------------------------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------------------------
def reserve_seconds(
    val_seconds: float | None, sec_per_sample: float | None, n_val: int, window_s: float
) -> float:
    """Seconds to keep free for the final validation + checkpoint.

    Uses the measured validation time once there is one; before that, an estimate from the
    training speed (which excludes warm-up batches, see the loop). Capped at RESERVE_CAP of
    the training window (fixed when training starts), so a bad estimate can never eat the
    whole budget.
    """
    if val_seconds is not None:
        estimate = 1.3 * val_seconds
    elif sec_per_sample is not None:
        estimate = VAL_COST_RATIO * sec_per_sample * n_val
    else:
        estimate = 0.0
    return min(estimate + SAVE_MARGIN_S, RESERVE_CAP * max(window_s, 0.0))


def lr_factor(progress: float) -> float:
    """Linear warm-up for the first 5% of progress, then cosine decay to 0 at progress 1."""
    if progress < WARMUP_FRACTION:
        return max(progress / WARMUP_FRACTION, 0.01)
    p = (progress - WARMUP_FRACTION) / (1 - WARMUP_FRACTION)
    return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _loaders(train_ds: Any, val_ds: Any, batch_size: int, device: Any) -> tuple[Any, Any]:
    from torch.utils.data import DataLoader

    workers = 4 if device.type == "cuda" else 0
    pin = device.type == "cuda"
    train_dl = DataLoader(
        train_ds,
        batch_size,
        shuffle=True,
        drop_last=len(train_ds) > batch_size,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=workers > 0,
    )
    val_dl = DataLoader(val_ds, batch_size, num_workers=workers, pin_memory=pin)
    return train_dl, val_dl


def history_rows(history: list[EpochRecord], timings: list[dict]) -> list[dict]:
    """history.json rows: the contract's EpochRecord fields plus train_s and val_s."""
    return [h.model_dump() | t for h, t in zip(history, timings, strict=True)]


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _save_predictions(path: Path, preds: dict, sample_names: list[str]) -> None:
    np.savez_compressed(path, sample_names=np.asarray(sample_names), **preds)


def train(config: TrainConfig, budget_minutes: float, run_dir: Path) -> TrainResult:
    """Contract entry point. Never raises: failures come back as status 'error'."""
    t_start = time.monotonic()
    run_dir = Path(run_dir)
    metric = "unknown"
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        if not (run_dir / "config.json").exists():
            (run_dir / "config.json").write_text(config.model_dump_json(indent=2), "utf-8")
        card = describe_dataset(config.task)
        metric = METRIC[card.task_type]
        problem = config_problem(config, card)
        if problem:
            return _error(run_dir, config, metric, t_start, f"invalid config: {problem}")
        return _train(config, card, budget_minutes, run_dir, t_start)
    except Exception as exc:  # the caller must never crash because of a trial
        msg = f"{type(exc).__name__}: {exc}"
        if "out of memory" in str(exc).lower():
            msg += " (hint: lower batch_size or img_size)"
        (run_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return _error(run_dir, config, metric, t_start, msg)


def _error(run_dir: Path, config: TrainConfig, metric: str, t_start: float, msg: str):
    minutes = (time.monotonic() - t_start) / 60
    _write_json(run_dir / "metrics.json", {"status": "error", "error": msg, "minutes": minutes})
    return TrainResult(
        run_id=run_dir.name,
        task=config.task,
        status="error",
        metric_name=metric,
        minutes_used=minutes,
        error=msg,
    )


def _train(
    config: TrainConfig, card: DatasetCard, budget_minutes: float, run_dir: Path, t_start: float
) -> TrainResult:
    import torch

    _seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"
    deadline = t_start + budget_minutes * 60

    # --- data and model ---
    bands = selected_bands(config, card)
    spec, dropped = model_spec(config, card, load_band_stats(config.task))
    norm = model_normalisation(spec)
    train_idx = subset_indices(card.split_sizes["train"], config.train_subset_frac, config.seed)
    inputs = {"bands": spec.bands, "norm": norm, "zero_fill": spec.zero_fill}
    train_ds = GeoBenchDataset(config.task, "train", indices=train_idx, **inputs)
    val_ds = GeoBenchDataset(config.task, "val", **inputs)
    train_dl, val_dl = _loaders(train_ds, val_ds, config.batch_size, device)
    model = build_model(spec).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")  # fp16 loss scaling
    seg = card.task_type == "segmentation"

    # --- loop ---
    max_steps = config.max_epochs * len(train_dl)
    train_start = time.monotonic()
    window_s = deadline - train_start  # training window, fixed once (caps the reserve)
    history: list[EpochRecord] = []
    timings: list[dict] = []  # per epoch: train_s, val_s (history.json extras)
    best = {"score": -1.0, "epoch": 0, "preds": None}
    step, full_epochs, val_seconds = 0, 0, None
    # Training speed WITHOUT warm-up: the first batch of every epoch (worker start-up, CUDA
    # init, cudnn.benchmark autotuning) is excluded from these running totals.
    timed_s, timed_samples = 0.0, 0
    stop_reason = "max_epochs"

    def reserve() -> float:  # seconds kept for the final validation + checkpoint
        speed = timed_s / timed_samples if timed_samples else None
        return reserve_seconds(val_seconds, speed, len(val_ds), window_s)

    def progress(now: float) -> float:
        used = now - train_start
        left = deadline - now - reserve()
        time_p = used / max(used + left, 1e-9) if left > 0 else 1.0
        return min(1.0, max(step / max(max_steps, 1), time_p))

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        losses, epoch_t0, finished_epoch = [], time.monotonic(), True
        last_end = epoch_t0
        for batch_i, (x, y) in enumerate(train_dl):
            now = time.monotonic()
            if step > 0 and deadline - now - reserve() <= 0:  # never before the first batch
                finished_epoch, stop_reason = False, "budget"
                break
            for group in opt.param_groups:
                group["lr"] = config.lr * lr_factor(progress(now))
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x, y = augment(x, y, config.augment, seg)
            x = resize(x, config.img_size)
            if seg:
                y = resize_mask(y, config.img_size)
            with amp(device):
                logits = model(x)
            loss = compute_loss(config.loss, card.task_type, logits, y, card.n_classes)
            if not torch.isfinite(loss):
                raise FloatingPointError("loss became NaN/inf (hint: lower lr)")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
            step += 1
            end = time.monotonic()
            if batch_i > 0:  # skip each epoch's warm-up batch; includes data-loading waits
                timed_s += end - last_end
                timed_samples += len(x)
            last_end = end
        if not losses:  # no time left for even one batch of this epoch
            break
        if not finished_epoch and full_epochs == 0:
            # First epoch cut short: report a 0-epoch timeout (empty history), which the tools
            # treat as "budget too small" and fail fast on. No partial validation.
            break
        full_epochs += finished_epoch
        train_s = time.monotonic() - epoch_t0  # measured before validation

        val_t0 = time.monotonic()
        preds = run_inference(
            model, val_dl, card.task_type, card.n_classes, config.img_size, device
        )
        val_score = score(card.task_type, preds)
        val_seconds = time.monotonic() - val_t0
        timings.append({"train_s": round(train_s, 2), "val_s": round(val_seconds, 2)})
        history.append(
            EpochRecord(
                epoch=epoch,
                train_loss=round(float(np.mean(losses)), 5),
                val_score=round(min(max(val_score, 0.0), 1.0), 5),
                elapsed_s=round(time.monotonic() - t_start, 1),
            )
        )
        if val_score > best["score"]:
            best = {"score": val_score, "epoch": epoch, "preds": preds}
            save_checkpoint(run_dir / CKPT_NAME, model, config, card, spec, norm, val_score, epoch)
        _write_json(run_dir / "history.json", history_rows(history, timings))
        if not finished_epoch:
            break  # stop_reason already "budget"
        if progress(time.monotonic()) >= 1.0:
            stop_reason = "max_epochs" if step >= max_steps else "budget"
            break

    # --- outputs ---
    minutes = (time.monotonic() - t_start) / 60
    has_ckpt = best["preds"] is not None
    if has_ckpt:
        names = [val_ds.sample_name(i) for i in range(len(val_ds))]
        _save_predictions(run_dir / "val_predictions.npz", best["preds"], names)
    status = "ok" if full_epochs >= 1 and has_ckpt else "timeout"
    val_score = round(min(max(best["score"], 0.0), 1.0), 5) if has_ckpt else None
    _write_json(
        run_dir / "metrics.json",
        {
            "status": status,
            "metric_name": METRIC[card.task_type],
            "val_score": val_score,
            "best_epoch": best["epoch"],
            "epochs_run": len(history),
            "full_epochs": full_epochs,
            "stop_reason": stop_reason,
            **nodata_extras(config.task, (best["preds"] or {}).get("confusion")),
            "minutes_used": round(minutes, 3),
            "budget_minutes": budget_minutes,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "device": str(device),
            "bands": bands,
            "model_inputs": spec.bands,
            "zero_filled_bands": spec.zero_fill,
            "dropped_bands": dropped,
            "normalisation": {"mean": norm[0].tolist(), "std": norm[1].tolist()},
        },
    )
    _write_json(run_dir / "history.json", history_rows(history, timings))
    return TrainResult(
        run_id=run_dir.name,
        task=config.task,
        status=status,
        metric_name=METRIC[card.task_type],
        val_score=val_score,
        history=history,
        minutes_used=minutes,
        checkpoint_path=(run_dir / CKPT_NAME) if has_ckpt else None,
        error=None if status == "ok" else "budget ran out inside the first epoch",
    )


# ---------------------------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------------------------
def checkpoint_list(checkpoint_path: Path | str | list) -> list[Path]:
    """One checkpoint or a list of them (an ensemble) -> a non-empty list of paths."""
    paths = [checkpoint_path] if isinstance(checkpoint_path, str | Path) else list(checkpoint_path)
    if not paths:
        raise ValueError("no checkpoints given")
    return [Path(p) for p in paths]


def load_ensemble(task: str, checkpoint_path: Path | list, device: Any) -> list[tuple[dict, Any]]:
    """Load every checkpoint (eval mode). Members must share task and class count; backbones,
    weights, bands and image sizes may differ (each model gets its own inputs)."""
    members = [load_checkpoint(p, device) for p in checkpoint_list(checkpoint_path)]
    wrong = [c["task"] for c, _ in members if c["task"] != task]
    if wrong:
        raise ValueError(f"checkpoint(s) for {sorted(set(wrong))} cannot score task {task!r}")
    if len({(c["n_classes"], c["task_type"]) for c, _ in members}) > 1:
        counts = sorted({c["n_classes"] for c, _ in members})
        raise ValueError(f"ensemble members disagree on the classes ({counts}); not combinable")
    for _, model in members:
        model.eval()
    return members


def checkpoint_inputs(raw: Any, ckpt: dict, bands: list[str]) -> Any:
    """Raw images [N, C, H, W] (dataset units, channels = `bands`) -> this checkpoint's inputs.

    Selects/reorders its bands by name, zero-fills the ones it expects but the task lacks
    (ssl4eo_s2), and applies its own normalisation (train z-score, or /10000 for ssl4eo_s2),
    exactly as GeoBenchDataset does during training.
    """
    import torch

    zero = set(ckpt["spec"].get("zero_fill", []))
    channels = [
        torch.zeros_like(raw[:, 0]) if b in zero else raw[:, bands.index(b)]
        for b in ckpt["spec"]["bands"]
    ]
    x = torch.stack(channels, dim=1)
    mean = torch.tensor(ckpt["norm_mean"], dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    std = torch.tensor(ckpt["norm_std"], dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return torch.nan_to_num((x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)


def ensemble_probs(members: list, raw: Any, bands: list[str], tta: bool) -> Any:
    """Average of every member's class probabilities (each with its own inputs and size)."""
    total = None
    for ckpt, model in members:
        x = checkpoint_inputs(raw, ckpt, bands)
        probs = class_probs(model, x, ckpt["config"]["img_size"], ckpt["task_type"], tta)
        total = probs if total is None else total + probs
    return total / len(members)


def score_from_per_sample(task_type: str, per_sample: dict) -> float:
    """The split's metric, computed only from per_sample (as contract v2 requires).

    classification: mean(correct)
    segmentation:   dataset-level mIoU: SUM intersection and union over all images per class,
                    then average sum(I_k) / sum(U_k) over classes with sum(U_k) > 0
                    (not the mean of per-image IoUs)
    multilabel:     micro-F1 = 2 TP / (2 TP + FP + FN), summed over images
    """
    if task_type == "classification":
        correct = per_sample["correct"]
        return float(np.mean(correct)) if len(correct) else 0.0
    if task_type == "segmentation":
        inter = np.asarray(per_sample["intersection"], dtype=np.float64).sum(axis=0)
        union = np.asarray(per_sample["union"], dtype=np.float64).sum(axis=0)
        present = union > 0
        return float(np.mean(inter[present] / union[present])) if present.any() else 0.0
    tp, fp, fn = (float(np.sum(per_sample[k])) for k in ("tp", "fp", "fn"))
    return 2 * tp / max(2 * tp + fp + fn, 1.0)


def per_sample_batch(task_type: str, probs: np.ndarray, y: np.ndarray, k: int) -> dict:
    """per_sample entries for one batch (lists, one item per image, in dataset order)."""
    if task_type == "classification":
        return {"correct": (probs.argmax(1) == y).astype(int).tolist()}
    if task_type == "multilabel":
        pred, true = probs > 0.5, y > 0.5
        return {
            "tp": (pred & true).sum(1).tolist(),
            "fp": (pred & ~true).sum(1).tolist(),
            "fn": (~pred & true).sum(1).tolist(),
        }
    pred = probs.argmax(1)
    valid = y != IGNORE_INDEX
    inter, union = [], []
    for c in range(k):
        p_c, t_c = pred == c, y == c
        inter.append(((p_c & t_c) & valid).sum(axis=(1, 2)))
        union.append(((p_c | t_c) & valid).sum(axis=(1, 2)))
    return {
        "intersection": np.stack(inter, axis=1).astype(int).tolist(),
        "union": np.stack(union, axis=1).astype(int).tolist(),
    }


def evaluate(task: str, checkpoint_path: Path | list, split: str, tta: bool = False) -> dict:
    """Contract v2: score one checkpoint or an ensemble on the FULL 'val' or test split.

    The test split is only ever requested by tools/final_test.py. Ensembles average the
    members' class probabilities; tta averages over the 8 flip/rot90 views. `score` is
    computed from `per_sample` (see score_from_per_sample).
    """
    import torch
    from torch.utils.data import DataLoader

    if split not in ("val", TEST_SPLIT):  # engine names; data.py maps "val" to geobench "valid"
        raise ValueError(f"split must be 'val' or the test split, got {split!r}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    members = load_ensemble(task, checkpoint_path, device)
    first = members[0][0]
    kind, k = first["task_type"], first["n_classes"]
    raw_ds = GeoBenchDataset(task, split, bands="all", norm="none", allow_test=split == TEST_SPLIT)
    per_sample: dict[str, list] = {}
    with torch.no_grad():
        for raw, y in DataLoader(raw_ds, batch_size=EVAL_BATCH, num_workers=0):
            probs = ensemble_probs(members, raw.to(device), raw_ds.bands, tta).cpu().numpy()
            for key, values in per_sample_batch(kind, probs, y.numpy(), k).items():
                per_sample.setdefault(key, []).extend(values)
    return {
        "task": task,
        "split": split,
        "metric_name": first["metric_name"],
        "score": round(min(max(score_from_per_sample(kind, per_sample), 0.0), 1.0), 5),
        "n_samples": len(raw_ds),
        "per_sample": per_sample,
        "n_checkpoints": len(members),
        "tta": tta,
    }


def predict(task: str, checkpoint_path: Path | list, images: np.ndarray, tta: bool = False) -> dict:
    """Contract v2: predictions for caller-supplied images; reads no dataset split.

    images: float N x C x H x W in the dataset's raw units, C and order = the card's bands.
    Each checkpoint normalises them its own way. Returns numpy arrays:
      probabilities float32 [N, K] or [N, K, H, W];  labels int64 [N] or [N, H, W]
      (multilabel: labels are the 0/1 decisions at 0.5, [N, K]).
    """
    import torch

    bands = list(describe_dataset(task).bands)
    arr = np.asarray(images, dtype=np.float32)
    if arr.ndim != 4 or arr.shape[1] != len(bands):
        raise ValueError(
            f"images must be N x {len(bands)} x H x W with bands in the order {bands}; "
            f"got shape {arr.shape}"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    members = load_ensemble(task, checkpoint_path, device)
    chunks = []
    with torch.no_grad():
        for i in range(0, len(arr), EVAL_BATCH):
            raw = torch.from_numpy(arr[i : i + EVAL_BATCH]).to(device)
            chunks.append(ensemble_probs(members, raw, bands, tta).cpu().numpy())
    probs = np.concatenate(chunks).astype(np.float32)
    if members[0][0]["task_type"] == "multilabel":
        labels = (probs > 0.5).astype(np.int64)
    else:
        labels = probs.argmax(1).astype(np.int64)
    return {"probabilities": probs, "labels": labels, "n_checkpoints": len(members), "tta": tta}


# Integration rule (Hilal): diagnose() lives in diagnostics.py; the agent side imports it from here.
from gotrain.engine.diagnostics import diagnose  # noqa: E402, F401
