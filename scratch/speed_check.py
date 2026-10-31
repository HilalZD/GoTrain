"""How long does one epoch spend LOADING tiles from hdf5 versus TRAINING on them?

    python scratch/speed_check.py m-brick-kiln --n 300            # on SageMaker (GPU if free)
    python scratch/speed_check.py m-eurosat --n 300 --scale-to 15063   # laptop proxy

Measures, on n random TRAIN tiles:
  load   GeoBenchDataset[i] one by one: hdf5 read + pack_to_3d + normalise (what each
         DataLoader worker does for every tile, every epoch)
  train  forward + backward + AdamW step on tiles already in memory, for resnet18/resnet50
         at the native size and at 224 px (the SSL4EO pretraining size)
and prints seconds per epoch for the full train split (or --scale-to N tiles).
Train split only; nothing is written.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gotrain.engine.data import GeoBenchDataset, subset_indices  # noqa: E402


def time_loading(ds: GeoBenchDataset) -> float:
    t0 = time.perf_counter()
    for i in range(len(ds)):
        ds[i]
    return (time.perf_counter() - t0) / len(ds)


def time_training(xs, n_classes: int, backbone: str, size: int, device, batch: int = 32) -> float:
    import timm
    import torch
    import torch.nn.functional as F

    model = timm.create_model(backbone, pretrained=False, in_chans=xs.shape[1], num_classes=n_classes)
    model = model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    y = torch.zeros(batch, dtype=torch.long, device=device)
    steps, seen = 0, 0
    t0 = None
    for start in range(0, len(xs) - batch + 1, batch):
        x = xs[start : start + batch].to(device)
        if size != x.shape[-1]:
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss.item()
        steps += 1
        if steps == 1:  # skip the warm-up batch
            t0 = time.perf_counter()
            continue
        seen += batch
    return (time.perf_counter() - t0) / max(seen, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("task")
    ap.add_argument("--n", type=int, default=300, help="train tiles to time")
    ap.add_argument("--scale-to", type=int, default=None, help="tiles per epoch to report")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full = GeoBenchDataset(args.task, "train", norm="none", data_dir=args.data_dir)
    n_epoch = args.scale_to or len(full)
    idx = subset_indices(len(full), min(1.0, args.n / len(full)), seed=0)
    ds = GeoBenchDataset(args.task, "train", norm="none", indices=idx, data_dir=args.data_dir)
    raw_dtype = ds.raw(0)[0].dtype
    load_s = time_loading(ds)
    xs = torch.stack([ds[i][0] for i in range(len(ds))])
    print(f"{args.task}: timed {len(ds)} train tiles on {device}; epoch = {n_epoch} tiles; "
          f"tile {tuple(xs.shape[1:])} stored as {raw_dtype}")  # fmt: skip
    print(f"  load (1 process):  {1000 * load_s:6.2f} ms/tile -> {load_s * n_epoch:7.1f} s/epoch "
          f"(~{load_s * n_epoch / 4:6.1f} s with 4 DataLoader workers, if they scale)")  # fmt: skip
    for backbone in ("resnet18", "resnet50"):
        for size in sorted({xs.shape[-1], 224}):
            train_s = time_training(xs, ds.n_classes, backbone, size, device)
            print(f"  train {backbone:<9} @ {size:>3}px: {1000 * train_s:7.2f} ms/tile -> "
                  f"{train_s * n_epoch:7.1f} s/epoch  (load/train = {load_s / train_s:5.2f})")  # fmt: skip
    cache_gb = n_epoch * int(np.prod(xs.shape[1:])) * 4 / 1e9
    print(f"  a float32 .npy cache of {n_epoch} tiles would take {cache_gb:.2f} GB")


if __name__ == "__main__":
    main()
