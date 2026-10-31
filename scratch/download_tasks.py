"""Download individual GEO-Bench tasks from Hugging Face (never the full benchmark).

Why this script exists
----------------------
geobench's own `geobench-download` command has no per-task option: it lists every
file in the Hugging Face repo `recursix/geo-bench-1.0` and downloads all of them
(~17.5 GB zipped, ~65 GB during unzip). Each task is a single zip on the Hub:

    classification_v1.0/<task>.zip    e.g. classification_v1.0/m-eurosat.zip
    segmentation_v1.0/<task>.zip      e.g. segmentation_v1.0/m-pv4ger-seg.zip

so we download only the zips we ask for with `huggingface_hub.hf_hub_download`,
unzip them next to where geobench expects them, and delete the zip.

Usage
-----
    python scratch/download_tasks.py m-eurosat m-pv4ger-seg --data-dir data
    export GEO_BENCH_DIR="$(pwd)/data"     # must be set BEFORE python starts

Resulting layout (what geobench expects under $GEO_BENCH_DIR):
    data/classification_v1.0/m-eurosat/{task_specs.pkl, default_partition.json, band_stats.json, *.hdf5}
    data/segmentation_v1.0/m-pv4ger-seg/...
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

REPO_ID = "recursix/geo-bench-1.0"
BENCHMARKS = ("classification_v1.0", "segmentation_v1.0")


def find_task_file(task: str) -> str:
    """Return the repo path of <task>.zip, looked up from the live repo listing."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    matches = [f for f in files if f.endswith(f"/{task}.zip") and f.split("/")[0] in BENCHMARKS]
    if not matches:
        available = sorted(f.split("/")[-1][:-4] for f in files if f.endswith(".zip"))
        raise SystemExit(f"Task '{task}' not found in {REPO_ID}. Available: {', '.join(available)}")
    return matches[0]


def safe_extract(zip_path: Path, dest: Path) -> None:
    """Extract a zip, refusing any member that would land outside `dest` (zip-slip guard)."""
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            target = (dest / member).resolve()
            if not target.is_relative_to(dest):  # separator-agnostic (Windows uses "\")
                raise RuntimeError(f"Unsafe path in zip: {member}")
        zf.extractall(dest)


def download_task(task: str, data_dir: Path, keep_zip: bool = False) -> Path:
    from huggingface_hub import hf_hub_download

    repo_path = find_task_file(task)
    benchmark = repo_path.split("/")[0]
    task_dir = data_dir / benchmark / task
    if (task_dir / "task_specs.pkl").exists():
        print(f"[skip] {task}: already present at {task_dir}")
        return task_dir

    free_gb = shutil.disk_usage(data_dir).free / 1e9
    print(f"[info] {task}: downloading {repo_path} (free disk: {free_gb:.1f} GB)")
    zip_path = Path(
        hf_hub_download(repo_id=REPO_ID, filename=repo_path, repo_type="dataset", local_dir=data_dir)
    )

    print(f"[info] {task}: extracting {zip_path.name}")
    safe_extract(zip_path, zip_path.parent)
    if not keep_zip:
        zip_path.unlink()

    if not (task_dir / "task_specs.pkl").exists():
        found = list((data_dir / benchmark).rglob("task_specs.pkl"))
        raise RuntimeError(
            f"Extracted {task} but {task_dir}/task_specs.pkl is missing. Found specs at: {found}"
        )
    n_files = sum(1 for _ in task_dir.glob("*.hdf5"))
    print(f"[ok]   {task}: {task_dir} ({n_files} hdf5 samples)")
    return task_dir


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasks", nargs="+", help="task names, e.g. m-eurosat m-pv4ger-seg")
    ap.add_argument("--data-dir", default="data", help="becomes $GEO_BENCH_DIR (default: ./data)")
    ap.add_argument("--keep-zip", action="store_true", help="keep the zip after extracting")
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        download_task(task, data_dir, keep_zip=args.keep_zip)

    print("\nNow set this in your shell (before starting python / restart the kernel):")
    print(f'  export GEO_BENCH_DIR="{data_dir}"')


if __name__ == "__main__":
    sys.exit(main())
