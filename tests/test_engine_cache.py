"""The on-disk .npy data cache (gotrain/engine/data.py, scripts/build_cache.py).

Contract: TRAIN and VAL only, never the test split; used automatically when valid; a stale or
foreign cache is ignored; results are identical to reading hdf5. Data tests need geobench and
the GEO-Bench files and skip otherwise.
"""

import json
from pathlib import Path

import pytest

from gotrain.engine import data

REPO = Path(__file__).resolve().parents[1]


def have(task: str) -> bool:
    """True only if geobench is importable AND the task is on disk (else data tests skip)."""
    try:
        import geobench  # noqa: F401

        data.find_task_dir(task)
        return True
    except (ImportError, FileNotFoundError):
        return False


needs_eurosat = pytest.mark.skipif(not have("m-eurosat"), reason="needs geobench + m-eurosat")
needs_pv4ger = pytest.mark.skipif(not have("m-pv4ger-seg"), reason="needs geobench + pv4ger")


# ------------------------------------------------------------------ no data needed
def test_test_split_is_never_cached():
    assert data.TEST_SPLIT not in data.CACHE_SPLITS
    assert data.load_cache("m-eurosat", data.TEST_SPLIT, ds=None, bands=[]) is None
    with pytest.raises(PermissionError, match="only"):
        data.build_split_cache("m-eurosat", data.TEST_SPLIT)


def test_cache_root_follows_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GOTRAIN_DATA_CACHE", str(tmp_path / "c"))
    assert data.cache_root() == tmp_path / "c"
    monkeypatch.delenv("GOTRAIN_DATA_CACHE")
    assert data.cache_root() == Path("~/data_cache").expanduser()


def test_cache_bytes():
    # 10 tiles of 64x64x13 float32 + int64 labels
    assert data.cache_bytes(10, (64, 64, 13), (), 8) == 10 * (64 * 64 * 13 * 4 + 8)


def test_missing_cache_means_hdf5(monkeypatch, tmp_path):
    monkeypatch.setenv("GOTRAIN_DATA_CACHE", str(tmp_path))
    assert data.load_cache("m-eurosat", "val", ds=None, bands=[]) is None


# ------------------------------------------------------------------ with data
@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    """A cache root holding eurosat train+val and pv4ger-seg val (whatever is on disk)."""
    root = tmp_path_factory.mktemp("data_cache")
    mp = pytest.MonkeyPatch()
    mp.setenv("GOTRAIN_DATA_CACHE", str(root))
    if have("m-eurosat"):
        for split in data.CACHE_SPLITS:
            data.build_split_cache("m-eurosat", split)
    if have("m-pv4ger-seg"):
        data.build_split_cache("m-pv4ger-seg", "val")
    yield root
    mp.undo()


@pytest.fixture
def use_cache_dir(cache_dir, monkeypatch):
    monkeypatch.setenv("GOTRAIN_DATA_CACHE", str(cache_dir))
    monkeypatch.setenv("GOTRAIN_STATS_DIR", str(REPO / "stats"))
    return cache_dir


def _same_items(task: str, split: str, every: int, **kw) -> None:
    import torch

    cached = data.GeoBenchDataset(task, split, **kw)
    direct = data.GeoBenchDataset(task, split, use_cache=False, **kw)
    assert cached.disk_cache is not None and direct.disk_cache is None
    assert len(cached) == len(direct)
    for i in range(0, len(direct), every):
        xc, yc = cached[i]
        xd, yd = direct[i]
        assert torch.equal(xc, xd), i
        assert torch.equal(yc, yd), i


@needs_eurosat
def test_cached_and_hdf5_paths_give_identical_classification_items(use_cache_dir):
    meta = json.loads((use_cache_dir / "m-eurosat" / "val_meta.json").read_text())
    assert meta["n"] == 1000 and meta["image_shape_hwc"] == [64, 64, 13]
    _same_items("m-eurosat", "val", every=37)
    _same_items("m-eurosat", "train", every=97, bands=["B04", "B8A", "B10"], norm="div10000")


@needs_pv4ger
def test_cached_and_hdf5_paths_give_identical_segmentation_items(use_cache_dir):
    meta = json.loads((use_cache_dir / "m-pv4ger-seg" / "val_meta.json").read_text())
    assert meta["label_dtype"] == "int16"
    _same_items("m-pv4ger-seg", "val", every=41)


@needs_eurosat
def test_stale_or_foreign_cache_is_ignored(use_cache_dir):
    meta_path = use_cache_dir / "m-eurosat" / "val_meta.json"
    original = meta_path.read_text()
    try:
        for change in ({"fingerprint": "0" * 64}, {"format": 999}, {"bands": ["B01"]}):
            meta_path.write_text(json.dumps(json.loads(original) | change))
            assert data.GeoBenchDataset("m-eurosat", "val").disk_cache is None, change
    finally:
        meta_path.write_text(original)
    assert data.GeoBenchDataset("m-eurosat", "val").disk_cache is not None


@needs_eurosat
def test_train_gives_the_same_score_with_and_without_cache(use_cache_dir, tmp_path, monkeypatch):
    from gotrain.contracts import TrainConfig
    from gotrain.engine import train as engine

    config = TrainConfig(
        task="m-eurosat",
        backbone="resnet18",
        weights="none",
        img_size=64,
        batch_size=16,
        lr=1e-3,
        max_epochs=1,
        loss="ce",
        train_subset_frac=0.05,
        augment=["flip"],
        reason="cache test",
    )
    # The LR schedule is tied to the wall clock; a huge budget keeps its time share ~0, so the
    # schedule depends only on steps and the two runs differ ONLY in where the data comes from.
    with_cache = engine.train(config, 600.0, tmp_path / "run-cache")
    metrics = json.loads((tmp_path / "run-cache" / "metrics.json").read_text())
    assert metrics["data_cache"]["train"] == {"used": True, "reason": None}
    assert metrics["data_cache"]["val"] == {"used": True, "reason": None}
    monkeypatch.setenv("GOTRAIN_DATA_CACHE", str(tmp_path / "no-cache-here"))
    without = engine.train(config, 600.0, tmp_path / "run-hdf5")
    metrics = json.loads((tmp_path / "run-hdf5" / "metrics.json").read_text())
    assert metrics["data_cache"]["train"]["used"] is False
    assert "no cache at" in metrics["data_cache"]["train"]["reason"]
    assert with_cache.status == without.status == "ok"
    assert with_cache.val_score == pytest.approx(without.val_score, abs=1e-6)


@needs_eurosat
def test_build_cache_script_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    import runpy
    import sys

    root = tmp_path / "cache-root"
    monkeypatch.setenv("GOTRAIN_DATA_CACHE", str(root))
    monkeypatch.setattr(sys, "argv", ["build_cache.py", "m-eurosat", "--dry-run"])
    runpy.run_path(str(REPO / "scripts" / "build_cache.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "m-eurosat" in out and "0.64 GB" in out and "never cached" in out
    assert not root.exists()


@needs_eurosat
def test_cached_dataset_pickles_small_and_reopens_its_cache(use_cache_dir):
    """DataLoader workers get a pickled copy: the memmapped cache must not be inside it."""
    import pickle

    ds = data.GeoBenchDataset("m-eurosat", "val")
    blob = pickle.dumps(ds)
    assert len(blob) < 5_000_000  # the val images alone are 213 MB
    copy = pickle.loads(blob)
    assert copy.disk_cache is not None
    import torch

    assert torch.equal(copy[3][0], ds[3][0])


@needs_eurosat
def test_worker_processes_give_the_same_batches(use_cache_dir):
    """2 worker processes (spawn, the strictest start method) vs reading hdf5 in-process."""
    import torch
    from torch.utils.data import DataLoader

    idx = list(range(0, 64))
    cached = data.GeoBenchDataset("m-eurosat", "val", indices=idx)
    direct = data.GeoBenchDataset("m-eurosat", "val", indices=idx, use_cache=False)
    workers = DataLoader(cached, 16, num_workers=2, multiprocessing_context="spawn")
    for (xw, yw), (xd, yd) in zip(workers, DataLoader(direct, 16), strict=True):
        assert torch.equal(xw, xd) and torch.equal(yw, yd)


def test_fallback_reasons_are_named(monkeypatch, tmp_path):
    monkeypatch.setenv("GOTRAIN_DATA_CACHE", str(tmp_path))
    assert "never cached" in data.check_cache("m-eurosat", data.TEST_SPLIT, None, [])[1]
    cache, reason = data.check_cache("m-eurosat", "val", None, [])
    assert cache is None and "no cache at" in reason and "build_cache.py m-eurosat" in reason


@needs_eurosat
def test_stale_cache_reason_and_loud_warning(use_cache_dir, capsys):
    from gotrain.engine import train as engine

    meta_path = use_cache_dir / "m-eurosat" / "val_meta.json"
    original = meta_path.read_text()
    try:
        meta_path.write_text(json.dumps(json.loads(original) | {"fingerprint": "0" * 64}))
        stale = data.GeoBenchDataset("m-eurosat", "val")
        assert stale.disk_cache is None and stale.cache_reason.startswith("stale cache")
        fresh = data.GeoBenchDataset("m-eurosat", "train")
        engine.warn_if_no_cache("m-eurosat", fresh, stale)
        err = capsys.readouterr().err
        assert "WARNING: data cache NOT used for m-eurosat/val: stale cache" in err
        assert "m-eurosat/train" not in err  # the train cache is valid: no warning
        report = engine.cache_report(fresh, stale)
        assert report["train"]["used"] is True and report["val"]["used"] is False
        assert report["root"] == str(use_cache_dir)
    finally:
        meta_path.write_text(original)
