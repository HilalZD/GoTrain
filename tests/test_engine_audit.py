"""Data Auditor (gotrain/engine/audit.py).

Unit tests feed the pure checks fake data. Acceptance tests need geobench + the real tasks:
raw m-eurosat must show both upstream bugs, the fixed version none, m-pv4ger-seg no alarms.
"""

import numpy as np
import pytest

from gotrain.engine import audit, data


@pytest.fixture(scope="module", autouse=True)
def isolated_cache_root(tmp_path_factory):
    """Audits cache their results under the data cache root: never the real ~/data_cache."""
    mp = pytest.MonkeyPatch()
    mp.setenv("GOTRAIN_DATA_CACHE", str(tmp_path_factory.mktemp("audit_cache_root")))
    yield
    mp.undo()


def have(task: str) -> bool:
    try:
        import geobench  # noqa: F401

        data.find_task_dir(task)
        return True
    except (ImportError, FileNotFoundError):
        return False


S2_TRUE = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B09",
    "B10",
    "B11",
    "B12",
    "B8A",
]


def fake_s2_pixels(n: int = 4000, seed: int = 0) -> np.ndarray:
    """Fake Sentinel-2 pixels in TRUE order (B8A last) with realistic relations."""
    rng = np.random.default_rng(seed)
    veg = rng.uniform(0, 1, n)  # vegetation amount drives NIR up and red down
    bright = rng.uniform(0.5, 1.5, n)
    vis = 800 * bright + rng.normal(0, 30, (3, n))
    red = vis[2] * (1.2 - 0.6 * veg)
    nir = (1500 + 2500 * veg) * bright
    swir = 1500 * bright * (1.3 - 0.5 * veg)
    cols = {
        "B01": vis[0] * 1.1 + rng.normal(0, 20, n),
        "B02": vis[0],
        "B03": vis[1],
        "B04": red,
        "B05": red * 1.1 + 200,
        "B06": nir * 0.85 + rng.normal(0, 20, n),
        "B07": nir * 0.95 + rng.normal(0, 20, n),
        "B08": nir + rng.normal(0, 20, n),
        "B09": rng.uniform(300, 1500, n),  # water vapour: weakly related
        "B10": rng.integers(5, 30, n).astype(float),  # cirrus: near zero
        "B11": swir + rng.normal(0, 40, n),
        "B12": swir * 0.7 + rng.normal(0, 40, n),
        "B8A": nir * 1.05 + rng.normal(0, 20, n),
    }
    return np.stack([cols[b] for b in S2_TRUE], axis=1)


# ------------------------------------------------------------------ names
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("08A - Vegetation Red Edge", "B8A"),
        ("10 - SWIR - Cirrus", "B10"),
        ("B8A", "B8A"),
        ("B01", "B01"),
        ("TCI_R", None),
        ("Red", None),
    ],
)
def test_s2_id(name, expected):
    assert audit.s2_id(name) == expected


def test_band_roles_from_s2_ids_and_words():
    assert audit.band_roles(S2_TRUE) == {"blue": 1, "green": 2, "red": 3, "nir": 7}
    assert audit.band_roles(["Red", "Green", "Blue", "NearInfrared"]) == {
        "red": 0,
        "green": 1,
        "blue": 2,
        "nir": 3,
    }


# ------------------------------------------------------------------ 1. bands (fake data)
def test_correct_band_names_raise_no_band_errors():
    assert audit.check_bands(fake_s2_pixels(), S2_TRUE) == []


def test_mislabelled_bands_are_flagged_with_a_suggestion():
    # GEO-Bench's raw eurosat naming: standard order although B8A is really last
    raw = S2_TRUE[:8] + ["B8A", "B09", "B10", "B11", "B12"]
    out = audit.check_bands(fake_s2_pixels(), raw)
    texts = " ".join(f["suggestion"] for f in out if f["severity"] == "error")
    assert "doesn't move with B08" in texts and "behaves like B8A" in texts
    assert "should be B10 (cirrus)" in texts and "look like B10 instead" in texts
    assert "SWIR B11/B12 don't move together" in texts


def test_eight_bit_channels_next_to_reflectance_are_reported():
    px = fake_s2_pixels()
    tci = np.clip(np.round(px[:, 3] / 8), 0, 255)  # a rendered 8-bit red, using 0-255
    out = audit.check_bands(np.column_stack([px, tci]), S2_TRUE + ["TCI_R"])
    info = [f for f in out if f["severity"] == "info"]
    assert info and info[0]["evidence"]["eight_bit_channels"] == ["TCI_R"]


def test_rgb_only_has_no_false_alarms():
    rng = np.random.default_rng(1)
    base = rng.uniform(0, 255, 3000)
    rgb = np.clip(np.column_stack([base + rng.normal(0, 10, 3000) for _ in range(3)]), 0, 255)
    assert audit.check_bands(np.round(rgb), ["Red", "Green", "Blue"]) == []


# ------------------------------------------------------------------ 2. class names (fake)
def test_class_names_contradicting_ndvi_are_flagged():
    ndvi = {"Forest": -0.25, "Sea and Lake": 0.65, "River": 0.34, "Highway": 0.4}
    out = audit.check_class_names(ndvi, {})
    flagged = {f["evidence"].get("class") for f in out if "class" in f["evidence"]}
    assert flagged == {"Forest", "Sea and Lake"}
    forest = next(f for f in out if f["evidence"].get("class") == "Forest")
    assert "'River'" in forest["suggestion"] and "'Sea and Lake'" in forest["suggestion"]
    assert any("out of order" in f["suggestion"] for f in out)


def test_plausible_class_names_raise_nothing():
    ndvi = {"Forest": 0.71, "Sea and Lake": -0.25, "River": 0.34, "Industrial Buildings": 0.22}
    assert audit.check_class_names(ndvi, {}) == []


# ------------------------------------------------------------------ 3-5 (fake)
def test_flat_tile_shortcut_warns_but_open_water_is_expected():
    majority = {"predict": "a", "metric": "accuracy", "score": 0.66}
    warn = audit.check_shortcuts(majority, {"not kiln": 0.138, "kiln": 0.0}, {})
    assert [f["severity"] for f in warn] == ["info", "warning"]
    water = audit.check_shortcuts(
        majority, {"Sea and Lake": 0.75, "Forest": 0.0}, {}, {"Sea and Lake": -0.25}
    )
    assert [f["severity"] for f in water] == ["info", "info"]


def test_leaks_and_nodata_classes():
    assert audit.check_leaks(100, 0, 50, 0) == []
    out = audit.check_leaks(3000, 75, 1000, 20)
    assert [f["evidence"].get("val_images_in_train") for f in out] == [None, 20]
    nod = audit.check_nodata_classes({"no data": 1.0, "water": 0.0})
    assert len(nod) == 1 and nod[0]["evidence"]["class"] == "no data"


# ------------------------------------------------------------------ acceptance (real data)
needs_eurosat = pytest.mark.skipif(not have("m-eurosat"), reason="needs geobench + m-eurosat")


@pytest.fixture(scope="module")
def eurosat_audits():
    if not have("m-eurosat"):
        pytest.skip("needs geobench + m-eurosat")
    return {fix: audit.audit_dataset("m-eurosat", apply_fixes=fix) for fix in (False, True)}


@needs_eurosat
def test_raw_eurosat_shows_both_upstream_bugs(eurosat_audits):
    raw = eurosat_audits[False]
    errors = [f for f in raw["findings"] if f["severity"] == "error"]
    assert {f["check"] for f in errors} == {"bands", "classes"}
    classes = {f["evidence"].get("class") for f in errors if f["check"] == "classes"}
    assert {"Forest", "Sea and Lake", "Residential Buildings"} <= classes


@needs_eurosat
def test_fixed_eurosat_has_no_band_or_class_errors(eurosat_audits):
    fixed = eurosat_audits[True]
    assert not [f for f in fixed["findings"] if f["severity"] in ("error", "warning")]
    assert fixed["ndvi_by_class"]["Sea and Lake"] < 0 < fixed["ndvi_by_class"]["Forest"]


@pytest.mark.skipif(not have("m-pv4ger-seg"), reason="needs geobench + m-pv4ger-seg")
def test_rgb_pv4ger_has_no_false_alarms():
    out = audit.audit_dataset("m-pv4ger-seg", n_samples=150, leak_limit=400)
    assert not [f for f in out["findings"] if f["severity"] in ("error", "warning")]
    assert any("No NIR band" in f["suggestion"] for f in out["findings"])


# ------------------------------------------------------------------ speed: hashing + result cache
def test_tile_hash_is_the_same_for_every_storage_dtype():
    """The data cache stores float32, hdf5 stores int16/uint8/float64: same image, same hash."""
    tile = np.arange(64 * 64 * 3).reshape(64, 64, 3) % 251
    hashes = {audit.tile_hash(tile.astype(t)) for t in (np.int16, np.uint8, np.float32, np.float64)}
    assert len(hashes) == 1


@pytest.fixture
def tmp_cache_root(monkeypatch, tmp_path):
    monkeypatch.setenv("GOTRAIN_DATA_CACHE", str(tmp_path / "cache"))
    return tmp_path / "cache"


@needs_eurosat
def test_hashes_from_data_cache_match_hdf5(tmp_cache_root):
    data.build_split_cache("m-eurosat", "val")
    cached = data.GeoBenchDataset("m-eurosat", "val", norm="none")
    direct = data.GeoBenchDataset("m-eurosat", "val", norm="none", use_cache=False)
    assert cached.disk_cache is not None
    assert audit._hashes(cached, 60) == audit._hashes(direct, 60)
    assert audit._label_counts(cached, "classification") == audit._label_counts(
        direct, "classification"
    )


@needs_eurosat
def test_audit_result_is_cached_and_invalidated(tmp_cache_root):
    kw = {"n_samples": 40, "leak_limit": 100}
    first = audit.audit_dataset("m-eurosat", **kw)
    again = audit.audit_dataset("m-eurosat", **kw)
    assert first["cached"] is False and again["cached"] is True
    assert again["findings"] == first["findings"]
    assert (tmp_cache_root / "m-eurosat" / "audit_fixed.json").exists()
    other = audit.audit_dataset("m-eurosat", n_samples=41, leak_limit=100)  # new settings
    assert other["cached"] is False
    raw = audit.audit_dataset("m-eurosat", apply_fixes=False, **kw)  # separate file per mode
    assert raw["cached"] is False and (tmp_cache_root / "m-eurosat" / "audit_raw.json").exists()
    assert audit.audit_dataset("m-eurosat", use_cache=False, **kw)["cached"] is False
