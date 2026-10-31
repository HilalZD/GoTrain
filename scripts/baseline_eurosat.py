"""Hand baseline for m-eurosat: SSL4EO Sentinel-2 ResNet50, fine-tuned on TRAIN, scored on VAL.

    python scripts/baseline_eurosat.py                    # full run (GPU)
    python scripts/baseline_eurosat.py --img-size 64      # no upsizing, for comparison
    python scripts/baseline_eurosat.py --smoke            # ~1 minute CPU check

What it does, step by step:
1. Data: all 13 bands, reordered BY NAME into the order the pretrained weights expect
   (m-eurosat stores B8A last; SSL4EO expects it between B08 and B09), scaled /10000 to match
   how the weights were pretrained (no mean subtraction). Train and val are loaded into memory
   once (2,000 + 1,000 tiles of 13x64x64 is ~650 MB as float32).
2. Model: TorchGeo's ResNet50_Weights.SENTINEL2_ALL_MOCO (13 input channels, self-supervised
   on SSL4EO-S12), with a fresh 10-class head.
3. Input size: tiles are 64x64 but the weights were pretrained on 224x224 crops, so batches
   are bilinearly upsized to --img-size on the GPU (default 224).
4. Training: AdamW, cosine LR with a short linear warm-up, random flips + 90-degree rotations
   (satellite images have no "up"), mixed precision on CUDA.
5. Every epoch is scored on the full VAL split; the best checkpoint and metrics are saved.
   The test split is never opened.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gotrain.config import get_settings
from gotrain.engine.data import GeoBenchDataset, subset_indices

TASK = "m-eurosat"


def torchgeo_band_order(weights) -> list[str]:
    """TorchGeo names bands 'B1', 'B8a'; GEO-Bench (after errata) uses 'B01', 'B8A'."""
    return [f"B{name[1:].upper().zfill(2)}" for name in weights.meta["bands"]]


def load_split(split: str, bands: list[str], limit: int | None) -> TensorDataset:
    """Read a split once into memory as (x float32 [N, 13, 64, 64], y long [N])."""
    ds = GeoBenchDataset(TASK, split, bands=bands, norm="div10000")
    if limit:
        ds.indices = subset_indices(len(ds), limit / len(ds), seed=0)
    xs, ys = zip(*(ds[i] for i in range(len(ds))), strict=True)
    print(f"[data] {split}: {len(ds)} tiles, bands {bands}")
    return TensorDataset(torch.stack(xs), torch.stack(ys))


def build_model(n_classes: int) -> nn.Module:
    from torchgeo.models import ResNet50_Weights, resnet50

    model = resnet50(weights=ResNet50_Weights.SENTINEL2_ALL_MOCO)
    model.reset_classifier(n_classes)  # timm API: new randomly initialised linear head
    return model


def augment(x: torch.Tensor) -> torch.Tensor:
    """Per-sample random horizontal/vertical flip and rotation by k*90 degrees."""
    out = torch.empty_like(x)
    for i in range(x.shape[0]):
        xi = x[i]
        if torch.rand(()) < 0.5:
            xi = xi.flip(-1)
        if torch.rand(()) < 0.5:
            xi = xi.flip(-2)
        out[i] = torch.rot90(xi, int(torch.randint(4, ())), dims=(-2, -1))
    return out


def resize(x: torch.Tensor, size: int) -> torch.Tensor:
    if x.shape[-1] == size:
        return x
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


def lr_at(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, img_size: int, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(resize(x, img_size))
        correct += int((logits.argmax(1) == y).sum())
        total += len(y)
    return correct / total


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="default: $GOTRAIN_HOME/baselines/...")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU check: 64 train/val tiles")
    args = ap.parse_args(argv)
    if args.smoke:
        args.epochs, args.batch_size, args.img_size = 1, 16, 64

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from torchgeo.models import ResNet50_Weights

    bands = torchgeo_band_order(ResNet50_Weights.SENTINEL2_ALL_MOCO)
    limit = 64 if args.smoke else None
    train_ds, val_ds = load_split("train", bands, limit), load_split("val", bands, limit)
    train_dl = DataLoader(
        train_ds, args.batch_size, shuffle=True, drop_last=True, pin_memory=device.type == "cuda"
    )
    val_dl = DataLoader(val_ds, 256, pin_memory=device.type == "cuda")

    model = build_model(n_classes=10).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    total_steps = args.epochs * len(train_dl)
    warmup = min(len(train_dl), total_steps // 10)
    out = (
        args.out or get_settings().home / "baselines" / f"eurosat-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    out.mkdir(parents=True, exist_ok=True)

    history, best, step, t0 = [], 0.0, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()
        model.train()
        running = 0.0
        for x, y in train_dl:
            for group in opt.param_groups:
                group["lr"] = lr_at(step, total_steps, args.lr, warmup)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x = resize(augment(x), args.img_size)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            step += 1
        train_s = time.time() - epoch_t0  # measured before validation
        val_acc = evaluate(model, val_dl, args.img_size, device)
        val_s = time.time() - epoch_t0 - train_s
        rec = {
            "epoch": epoch,
            "train_loss": round(running / len(train_dl), 4),
            "val_acc": round(val_acc, 4),
            "train_s": round(train_s, 2),
            "val_s": round(val_s, 2),
            "elapsed_s": round(time.time() - t0, 1),
        }
        history.append(rec)
        print(
            f"[epoch {epoch:>3}] loss {rec['train_loss']:.4f}  val acc {val_acc:.4f}  "
            f"({rec['elapsed_s']:.0f}s)"
        )
        if val_acc > best:
            best = val_acc
            torch.save(
                {"model": model.state_dict(), "bands": bands, "args": vars(args)}, out / "best.pt"
            )

    metrics = {
        "task": TASK,
        "metric": "accuracy",
        "best_val_acc": best,
        "history": history,
        "args": {k: str(v) for k, v in vars(args).items()},
        "device": str(device),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[done] best val accuracy {best:.4f} -> {out}")


if __name__ == "__main__":
    main()
