"""Tests for gotrain/engine/data.py (Serena's data module).

Card tests use the committed recon cards in stats/, so they run anywhere. Dataset tests need
the GEO-Bench files on disk and are skipped when they are missing.
"""

import json
import re
from pathlib import Path

import pytest

from gotrain.contracts import DatasetCard
from gotrain.engine import data

REPO = Path(__file__).resolve().parents[1]
STATS = REPO / "stats"


def recon(task: str) -> dict:
    return json.loads((STATS / f"{task}_card.json").read_text(encoding="utf-8"))


def have_task(task: str) -> bool:
    try:
        data.find_task_dir(task)
        return True
    except FileNotFoundError:
        return False


needs_eurosat = pytest.mark.skipif(not have_task("m-eurosat"), reason="m-eurosat not downloaded")
needs_pv4ger = pytest.mark.skipif(
    not have_task("m-pv4ger-seg"), reason="m-pv4ger-seg not downloaded"
)


# ------------------------------------------------------------------ recon card -> DatasetCard
@pytest.mark.parametrize("task", ["m-eurosat", "m-pv4ger-seg"])
def test_recon_card_flattens_to_valid_contract(task):
    card = data.to_dataset_card(recon(task))
    assert isinstance(card, DatasetCard)
    assert DatasetCard.model_validate(card.model_dump()) == card
    assert abs(sum(card.class_distribution.values()) - 1.0) < 1e-6
    assert isinstance(card.dtype, str)
    lo, hi = card.value_range
    assert lo <= hi
    assert set(card.split_sizes) == {"train", "val"}


def test_eurosat_card_keeps_the_facts_flattening_would_lose():
    card = data.to_dataset_card(recon("m-eurosat"))
    assert card.sensor == "sentinel-2"  # tools/inspect.py keys its weights hint on this slug
    assert card.class_names[1] == "Forest" and card.class_names[9] == "Sea and Lake"
    assert card.bands[-1] == "B8A"
    assert card.dtype == "int16"
    assert card.class_distribution["Forest"] == pytest.approx(0.1)
    assert card.mask_unique_values is None
    for fact in ("Band names corrected", "Class names corrected", "Nearly empty band(s): B10"):
        assert fact in card.notes
    assert "B10 29 vs B8A 5352" in card.notes  # the scale note names the real bands
    assert "ssl4eo_s2" in card.notes and "all present" in card.notes


def test_sentinel2_cards_pass_hilals_combo_check():
    from gotrain.engine.combos import check_combo, is_sentinel2

    eurosat = data.to_dataset_card(recon("m-eurosat"))
    assert is_sentinel2(eurosat) and check_combo(eurosat, "resnet50", "ssl4eo_s2") is None
    pv4ger = data.to_dataset_card(recon("m-pv4ger-seg"))
    assert not is_sentinel2(pv4ger)


def test_sensor_is_sentinel2_when_any_band_is_sentinel2():
    class Sentinel2:
        name = "B02"

    class SpectralBand:
        def __init__(self, name):
            self.name = name

    class Task:  # brick-kiln-like: Sentinel-2 bands plus derived 8-bit TCI channels
        bands_info = [Sentinel2(), SpectralBand("TCI_R"), SpectralBand("TCI_G")]

    assert data.sensor(Task()) == "sentinel-2"


def test_pv4ger_card_mask_and_distribution():
    card = data.to_dataset_card(recon("m-pv4ger-seg"))
    assert card.sensor == "aerial-rgb"
    assert card.mask_unique_values == [0, 1]
    assert card.class_distribution["solar pv"] == pytest.approx(0.1996, abs=1e-4)
    lo, hi = card.value_range
    assert 0.0 <= lo < 10.0 and hi == 255.0  # all-zero pixels count as nodata, so min > 0
    assert "fraction of train pixels" in card.notes


def test_describe_dataset_reads_cached_card(monkeypatch):
    monkeypatch.setenv("GOTRAIN_STATS_DIR", str(STATS))
    card = data.describe_dataset("m-eurosat")
    assert card.task == "m-eurosat" and card.n_classes == 10


def test_describe_dataset_unknown_task_raises_file_not_found(monkeypatch, tmp_path):
    monkeypatch.setenv("GOTRAIN_STATS_DIR", str(tmp_path))
    monkeypatch.setenv("GEO_BENCH_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        data.describe_dataset("m-no-such-task-xyz")


# ------------------------------------------------------------------ errata and guards
def test_errata_tables_are_permutations_or_full_lists():
    eurosat = data.CLASS_NAME_FIXES["m-eurosat"]
    assert len(eurosat) == len(set(eurosat)) == 10
    for bands in data.BAND_ORDER_FIXES.values():
        assert len(bands) == len(set(bands))


def test_class_names_fix_only_applies_on_length_match():
    assert data.corrected_class_names("m-eurosat", ["a", "b"]) == (["a", "b"], False)
    names, fixed = data.corrected_class_names("m-eurosat", [str(i) for i in range(10)])
    assert fixed and names[0] == "Annual Crop"


def test_open_split_refuses_test_without_permission():
    with pytest.raises(PermissionError, match="sacred"):
        data.open_split(object(), data.TEST_SPLIT)
    with pytest.raises(ValueError, match="'train' or 'val'"):
        data.open_split(object(), "valid")


def test_only_evaluate_may_allow_the_test_split():
    """Static guard: allow_test=<anything but False> only appears in engine/train.py."""
    pattern = re.compile(r"allow_test\s*=\s*(?!False\b)")
    allowed = {REPO / "gotrain" / "engine" / "train.py", REPO / "gotrain" / "engine" / "data.py"}
    offenders = [
        str(p.relative_to(REPO))
        for d in ("gotrain", "scripts")
        for p in (REPO / d).rglob("*.py")
        if p not in allowed and pattern.search(p.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"only train.evaluate may open the test split: {offenders}"


def test_subset_indices_is_seeded_and_sized():
    a = data.subset_indices(100, 0.25, seed=3)
    assert a == data.subset_indices(100, 0.25, seed=3) and len(a) == 25
    assert data.subset_indices(10, 1.0, seed=0) == list(range(10))
    assert len(data.subset_indices(10, 0.01, seed=0)) == 1


# ------------------------------------------------------------------ dataset (needs data)
@needs_eurosat
def test_eurosat_dataset_shapes_and_band_selection():
    torch = pytest.importorskip("torch")
    full = data.GeoBenchDataset("m-eurosat", "val", indices=range(4))
    x, y = full[0]
    assert x.shape == (13, 64, 64) and x.dtype == torch.float32
    assert y.dtype == torch.long and 0 <= int(y) < 10
    rgb = data.GeoBenchDataset("m-eurosat", "val", bands=["B04", "B03", "B02"], indices=range(4))
    xr, _ = rgb[0]
    idx = [full.all_bands.index(b) for b in ("B04", "B03", "B02")]
    assert torch.allclose(xr, x[idx])


@needs_eurosat
def test_eurosat_zscore_is_roughly_standardised_on_train():
    torch = pytest.importorskip("torch")
    ds = data.GeoBenchDataset("m-eurosat", "train", indices=range(0, 2000, 40))
    xs = torch.stack([ds[i][0] for i in range(len(ds))])
    per_band_mean = xs.mean(dim=(0, 2, 3))
    assert per_band_mean.abs().max() < 0.6


@needs_eurosat
def test_div10000_gives_reflectance_scale():
    ds = data.GeoBenchDataset("m-eurosat", "train", norm="div10000", indices=range(2))
    x, _ = ds[0]
    assert 0.0 <= float(x.min()) and float(x.max()) < 2.0


@needs_eurosat
def test_dataset_refuses_test_and_unknown_bands():
    with pytest.raises(PermissionError):
        data.GeoBenchDataset("m-eurosat", data.TEST_SPLIT)
    with pytest.raises(ValueError, match="available"):
        data.GeoBenchDataset("m-eurosat", "val", bands=["B99"])


@needs_pv4ger
def test_pv4ger_mask_target():
    torch = pytest.importorskip("torch")
    ds = data.GeoBenchDataset("m-pv4ger-seg", "val", indices=range(2))
    x, y = ds[0]
    assert x.shape == (3, 320, 320)
    assert y.shape == (320, 320) and y.dtype == torch.long
    assert set(torch.unique(y).tolist()) <= {0, 1}
