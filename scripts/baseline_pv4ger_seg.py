"""Hand baseline for m-pv4ger-seg: U-Net (ImageNet ResNet-34 encoder), Dice + BCE, full resolution.

    python scripts/baseline_pv4ger_seg.py            # full run (GPU)
    python scripts/baseline_pv4ger_seg.py --smoke    # ~1 minute CPU check

What it does, step by step:
1. Data: 3-band aerial RGB at 0.1 m/pixel, 320x320, kept at full resolution (thin panel
   edges disappear if we downsample). Each band is z-scored with stats/m-pv4ger-seg.json
   (TRAIN-only mean/std). Masks are 0 (background) / 1 (solar panel).
2. Model: segmentation_models_pytorch U-Net with a ResNet-34 encoder pretrained on ImageNet,
   one output channel = the logit of "this pixel is a panel".
3. Loss: Dice + BCE. BCE scores every pixel independently; Dice scores the overlap of the
   predicted and true panel regions, so it does not let the 80% background dominate.
4. Augmentation: the same random flips + 90-degree rotation on image and mask.
5. Metrics on the full VAL split, accumulated over all pixels: IoU of the panel class, IoU of
   background, and mIoU (their mean). Pixel accuracy is NOT reported as a goal: predicting
   "no panel" everywhere already scores 80%.
   The test split is never opened.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

from gotrain.config import get_settings
from gotrain.engine.data import GeoBenchDataset

TASK = "m-pv4ger-seg"


def build_model() -> nn.Module:
    import segmentation_models_pytorch as smp

    return smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1)


def augment(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Same random flip/rot90 for each image and its mask. x: [B, C, H, W], y: [B, H, W]."""
    xs, ys = [], []
    for xi, yi in zip(x, y, strict=True):
        if torch.rand(()) < 0.5:
            xi, yi = xi.flip(-1), yi.flip(-1)
        if torch.rand(()) < 0.5:
            xi, yi = xi.flip(-2), yi.flip(-2)
        k = int(torch.randint(4, ()))
        xs.append(torch.rot90(xi, k, dims=(-2, -1)))
        ys.append(torch.rot90(yi, k, dims=(-2, -1)))
    return torch.stack(xs), torch.stack(ys)


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """logits [B, 1, H, W], target [B, H, W] in {0, 1}. Soft Dice over the batch + BCE."""
    t = target.unsqueeze(1).float()
    bce = F.binary_cross_entropy_with_logits(logits, t)
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)
    return bce + dice


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Dataset-level IoU: intersections and unions summed over every VAL pixel."""
    model.eval()
    inter, union = torch.zeros(2), torch.zeros(2)
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            pred = (model(x).squeeze(1) > 0).long()  # logit > 0  <=>  probability > 0.5
        for c in (0, 1):
            inter[c] += ((pred == c) & (y == c)).sum().item()
            union[c] += ((pred == c) | (y == c)).sum().item()
        correct += (pred == y).sum().item()
        total += y.numel()
    iou = (inter / union.clamp(min=1)).tolist()
    return {
        "iou_panel": iou[1],
        "iou_background": iou[0],
        "miou": sum(iou) / 2,
        "pixel_acc": correct / total,
    }


def lr_at(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="default: $GOTRAIN_HOME/baselines/...")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU check: 16 train/val tiles")
    args = ap.parse_args(argv)
    if args.smoke:
        args.epochs, args.batch_size, args.workers = 1, 4, 0

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = GeoBenchDataset(TASK, "train", norm="zscore")
    val_ds = GeoBenchDataset(TASK, "val", norm="zscore")
    if args.smoke:
        train_ds, val_ds = Subset(train_ds, range(16)), Subset(val_ds, range(16))
    workers = {"num_workers": args.workers}
    if args.workers > 0:  # forkserver: forking after CUDA init can deadlock (see engine)
        methods = multiprocessing.get_all_start_methods()
        workers |= {
            "pin_memory": device.type == "cuda",
            "persistent_workers": True,
            "multiprocessing_context": "forkserver" if "forkserver" in methods else "spawn",
        }
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, drop_last=True, **workers)
    val_dl = DataLoader(val_ds, args.batch_size * 2, **workers)
    print(f"[data] train {len(train_ds)} / val {len(val_ds)} tiles at full 320x320 resolution")

    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    total_steps = args.epochs * len(train_dl)
    warmup = min(len(train_dl), total_steps // 10)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = args.out or get_settings().home / "baselines" / f"pv4ger-seg-{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    history, best, step, t0 = [], -1.0, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()
        model.train()
        running = 0.0
        for x, y in train_dl:
            for group in opt.param_groups:
                group["lr"] = lr_at(step, total_steps, args.lr, warmup)
            x, y = augment(x.to(device, non_blocking=True), y.to(device, non_blocking=True))
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
            loss = dice_bce_loss(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            step += 1
        train_s = time.time() - epoch_t0  # measured before validation
        m = evaluate(model, val_dl, device)
        val_s = time.time() - epoch_t0 - train_s
        rec = {
            "epoch": epoch,
            "train_loss": round(running / len(train_dl), 4),
            **{k: round(v, 4) for k, v in m.items()},
            "train_s": round(train_s, 2),
            "val_s": round(val_s, 2),
            "elapsed_s": round(time.time() - t0, 1),
        }
        history.append(rec)
        print(
            f"[epoch {epoch:>3}] loss {rec['train_loss']:.4f}  panel IoU {m['iou_panel']:.4f}  "
            f"mIoU {m['miou']:.4f}  (pixel acc {m['pixel_acc']:.4f})  {rec['elapsed_s']:.0f}s"
        )
        if m["miou"] > best:
            best = m["miou"]
            torch.save({"model": model.state_dict(), "args": vars(args), **m}, out / "best.pt")

    best_rec = max(history, key=lambda r: r["miou"])
    metrics = {
        "task": TASK,
        "metric": "miou",
        "best": best_rec,
        "history": history,
        "args": {k: str(v) for k, v in vars(args).items()},
        "device": str(device),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"[done] best epoch {best_rec['epoch']}: mIoU {best_rec['miou']:.4f}, "
        f"panel IoU {best_rec['iou_panel']:.4f} -> {out}"
    )


if __name__ == "__main__":
    main()
