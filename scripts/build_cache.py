"""Build the one-time .npy data cache for TRAIN and VAL (never the test split). CPU only.

    python scripts/build_cache.py m-eurosat m-brick-kiln m-pv4ger-seg m-chesapeake
    python scripts/build_cache.py m-brick-kiln --dry-run     # expected disk use, writes nothing

Why: reading every tile from its own hdf5 file each epoch dominates training time on the T4.
This reads each split once and writes <cache>/<task>/<split>_{x,y}.npy + <split>_meta.json
(float32 images, all bands in card order, raw units). The engine memory-maps them
automatically while their fingerprint matches the source files, and falls back to hdf5
otherwise, with identical results.

Cache root: $GOTRAIN_DATA_CACHE, default ~/data_cache. Data: $GEO_BENCH_DIR or --data-dir.
CPU only (CUDA_VISIBLE_DEVICES is cleared and torch is never imported), so it can run while
the GPU is busy with training. A split whose cache is already valid is skipped (--force
rebuilds it).
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # never touch the GPU, even by accident

import argparse  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gotrain.engine import data  # noqa: E402


def estimate(task_name: str, data_dir: str | None) -> dict:
    """Expected bytes per split, from one sample's shape (float32 images + labels)."""
    task = data.load_task(task_name, data_dir)
    kind = data.task_type(task)
    out = {}
    for split in data.CACHE_SPLITS:
        ds = data.open_split(task, split)
        sample = ds[0]
        arr, _, _ = data.sample_array(sample)
        label = np.asarray(data.label_value(sample.label))
        itemsize = np.dtype(data._label_store_dtype(kind, label)).itemsize
        out[split] = (len(ds), data.cache_bytes(len(ds), arr.shape, label.shape, itemsize))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("tasks", nargs="+")
    ap.add_argument("--data-dir", default=None, help="GEO_BENCH_DIR root (default: $GEO_BENCH_DIR)")
    ap.add_argument("--dry-run", action="store_true", help="only print the expected disk use")
    ap.add_argument("--force", action="store_true", help="rebuild even if the cache is valid")
    args = ap.parse_args()

    root = data.cache_root()
    print(
        f"cache root: {root}  (splits: {', '.join(data.CACHE_SPLITS)}; "
        "the test split is never cached)"
    )
    plan = {t: estimate(t, args.data_dir) for t in args.tasks}
    total = 0
    for task_name, splits in plan.items():
        line = ", ".join(f"{s} {n} tiles {b / 1e9:.2f} GB" for s, (n, b) in splits.items())
        task_total = sum(b for _, b in splits.values())
        total += task_total
        print(f"  {task_name:<14} {task_total / 1e9:6.2f} GB  ({line})")
    existing = next(q for q in (root, *root.parents) if q.exists())  # dry run creates nothing
    free = shutil.disk_usage(existing).free
    print(f"  total          {total / 1e9:6.2f} GB; free at {root}: {free / 1e9:.1f} GB")
    if args.dry_run:
        return
    if total > free * 0.9:
        raise SystemExit(
            "not enough free disk for the cache (keeping a 10% margin); nothing written"
        )
    root.mkdir(parents=True, exist_ok=True)

    for task_name in args.tasks:
        task = data.load_task(task_name, args.data_dir)
        for split in data.CACHE_SPLITS:
            ds = data.open_split(task, split)
            bands = data.channel_names(task_name, data.sample_array(ds[0])[1])[0]
            if not args.force and data.load_cache(task_name, split, ds, bands) is not None:
                print(f"[skip] {task_name}/{split}: cache is valid")
                continue
            t0 = time.time()

            def progress(done: int, n: int, t0=t0, name=f"{task_name}/{split}") -> None:
                print(f"       {name}: {done}/{n} tiles ({time.time() - t0:.0f}s)", flush=True)

            meta = data.build_split_cache(task_name, split, args.data_dir, progress=progress)
            print(
                f"[ok]   {task_name}/{split}: {meta['n']} tiles, {meta['bytes'] / 1e9:.2f} GB "
                f"in {time.time() - t0:.0f}s"
            )


if __name__ == "__main__":
    main()
