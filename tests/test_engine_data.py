"""Tests for gotrain/engine/data.py (Serena's data module).

Card tests use the committed recon cards in stats/, so they run anywhere. Dataset tests need
the GEO-Bench files on disk and are skipped when they are missing.
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from gotrain.contracts import DatasetCard
from gotrain.engine import data

REPO = Path(__file__).resolve().parents[1]
STATS = REPO / "stats"


def recon(task: str) -> dict:
    return json.loads((STATS / f"{task}_card.json").read_text(encoding="utf-8"))


def have_task(task: str) -> bool:
    """True only if geobench is importable AND the task is on disk (else data tests skip)."""
    try:
        import geobench  # noqa: F401

        data.find_task_dir(task)
        return True
    except (ImportError, FileNotFoundError):
        return False


needs_eurosat = pytest.mark.skipif(
    not have_task("m-eurosat"), reason="needs geobench + m-eurosat on disk"
)
needs_pv4ger = pytest.mark.skipif(
    not have_task("m-pv4ger-seg"), reason="needs geobench + m-pv4ger-seg on disk"
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


# ------------------------------------------------------------------ brick-kiln and chesapeake cards
@pytest.mark.parametrize("task", ["m-brick-kiln", "m-chesapeake"])
def test_new_recon_cards_flatten_to_valid_contract(task):
    card = data.to_dataset_card(recon(task))
    assert DatasetCard.model_validate(card.model_dump()) == card
    assert abs(sum(card.class_distribution.values()) - 1.0) < 1e-4  # pct stored to 3 decimals


def test_chesapeake_class_names_are_corrected():
    card = data.to_dataset_card(recon("m-chesapeake"))
    assert card.class_names[0] == "no data" and card.class_names[1] == "water"
    assert card.class_names[6] == "impervious roads"
    assert card.class_distribution["no data"] == pytest.approx(0.0417, abs=1e-4)
    assert card.sensor == "aerial-rgbn"
    assert "Class names corrected" in card.notes and "no data" in card.notes
    assert "20 validation image(s) identical" in card.notes
    names, fixed = data.corrected_class_names("m-chesapeake", [str(i) for i in range(7)])
    assert fixed and names == card.class_names


def test_brick_kiln_card_reports_baseline_and_flat_shortcut_facts():
    from gotrain.engine.combos import check_combo

    rec = recon("m-brick-kiln")
    assert rec["majority_class_baseline"] == pytest.approx(0.6637, abs=1e-4)
    assert rec["flat_tile_fraction"] == {"not brick kiln": 0.1378, "brick kiln": 0.0}
    card = data.to_dataset_card(rec)
    assert "always predicting 'not brick kiln' scores 0.664 accuracy" in card.notes
    assert "by class: not brick kiln 13.8%, brick kiln 0.0%" in card.notes
    assert "zero-filled (missing here): B06, B09, B10" in card.notes
    assert check_combo(card, "resnet50", "ssl4eo_s2") is None  # sensor reads as Sentinel-2


# ------------------------------------------------------------------ request #11: shortcut metrics
def test_majority_baseline_classification_uses_train_majority_on_val():
    from collections import Counter

    out = data.majority_class_baseline(
        "classification", Counter({0: 70, 1: 30}), Counter({0: 60, 1: 40}), 100, ["a", "b"]
    )
    assert out == {"predict": "a", "metric": "accuracy", "score": 0.6}


def test_majority_baseline_segmentation_follows_v2_miou():
    from collections import Counter

    # val pixels: 80 of class 0, 20 of class 1, class 2 absent -> (0.8 + 0) / 2 present classes
    out = data.majority_class_baseline(
        "segmentation", Counter({0: 900, 1: 100}), Counter({0: 80, 1: 20}), 5, ["x", "y", "z"]
    )
    assert out["metric"] == "miou" and out["score"] == pytest.approx(0.4)


def test_majority_baseline_multilabel_micro_f1():
    from collections import Counter

    train = Counter({0: 8, 1: 2, "_n": 10})  # label 0 positive in 80% of train -> always on
    out = data.majority_class_baseline("multilabel", train, Counter({0: 3, 1: 1}), 4, ["p", "q"])
    assert out["predict"] == ["p"] and out["score"] == pytest.approx(2 * 3 / (2 * 3 + 1 + 1))


def test_tile_spread_ignores_nodata_and_flags_empty_tiles():
    tile = np.zeros((4, 4, 2))
    tile[0, :, 0] = [10, 20, 30, 40]
    tile[0, :, 1] = [1, 1, 1, 1]
    spread = data.tile_spread(tile)  # only the 4 non-zero pixels count
    assert spread[0] == pytest.approx(
        np.percentile([10, 20, 30, 40], 98) - np.percentile([10, 20, 30, 40], 2)
    )
    assert spread[1] == 0.0
    assert np.isnan(data.tile_spread(np.zeros((2, 2, 3)))).all()


def test_flat_tile_fraction_per_class():
    std = np.array([10.0, 10.0])
    spreads = np.array([[1.0, 1.0], [50.0, 40.0], [2.0, 30.0], [np.nan, np.nan]])
    classes = [[0], [1], [1], [0]]  # contrasts: 0.1 flat, 4.5, 1.6, NaN (empty) -> flat
    out = data.flat_tile_fraction(spreads, std, classes, ["water", "land", "never"])
    assert out["threshold"] == data.FLAT_THRESHOLD and out["n_tiles"] == 4
    assert out["overall"] == 0.5
    assert out["per_class"] == {"water": 1.0, "land": 0.0, "never": None}


def test_eurosat_card_reports_baseline_and_flat_water():
    rec = recon("m-eurosat")
    assert rec["majority_class_baseline"] == pytest.approx(0.1)
    assert rec["flat_tile_fraction"]["Sea and Lake"] == pytest.approx(0.75)
    card = data.to_dataset_card(rec)
    assert "Majority-class baseline on val" in card.notes and "0.100 accuracy" in card.notes
    assert "Flat tiles (a tile is nearly flat when" in card.notes
    assert "by class: Sea and Lake 75.0%" in card.notes


def test_pv4ger_card_has_no_flat_tiles_and_a_miou_baseline():
    rec = recon("m-pv4ger-seg")
    assert rec["majority_class_baseline_detail"]["metric"] == "miou"
    assert rec["majority_class_baseline"] == pytest.approx(0.40, abs=0.01)
    assert rec["flat_tile_overall"] == 0.0
    assert set(rec["flat_tile_fraction"].values()) == {0.0}


def test_cards_without_the_new_fields_still_work():
    rec = recon("m-eurosat")
    rec.pop("majority_class_baseline")
    rec.pop("flat_tile_fraction")
    assert "Majority-class baseline" not in data.to_dataset_card(rec).notes


def test_near_infrared_band_name_gives_rgbn_slug():
    class SpectralBand:
        def __init__(self, name):
            self.name = name

    class Task:
        bands_info = [SpectralBand(n) for n in ("Red", "Green", "Blue", "NearInfrared")]

    assert data.sensor(Task()) == "aerial-rgbn"


def test_agent_discovers_shortcut_notes_through_its_own_tool(monkeypatch):
    """The agent's inspect_dataset tool (real engine) shows the baseline and flat-tile notes."""
    from gotrain.tools.inspect import inspect_dataset

    monkeypatch.setenv("GOTRAIN_ENGINE", "real")
    monkeypatch.setenv("GOTRAIN_STATS_DIR", str(STATS))
    out = inspect_dataset("m-eurosat")
    notes = out["dataset"]["notes"]
    assert "Majority-class baseline on val" in notes
    assert "Flat tiles" in notes and "Sea and Lake 75.0%" in notes


def test_card_notes_for_the_new_fields_are_facts_only():
    """Hilal (#11): facts only, no recipe or advice in the card."""
    notes = data._shortcut_notes(recon("m-brick-kiln"))
    assert len(notes) == 2
    for word in ("should", "must", "consider", "use ", "shortcut", "recall"):
        assert all(word not in n.lower() for n in notes), word


def test_shortcut_card_fields_use_the_final_names():
    fields = data.shortcut_card_fields(
        {"predict": "a", "metric": "accuracy", "score": 0.7},
        {
            "definition": "d",
            "threshold": 0.25,
            "n_tiles": 10,
            "overall": 0.1,
            "per_class": {"a": 0.2, "b": None},
        },
    )
    assert fields["majority_class_baseline"] == 0.7
    assert fields["flat_tile_fraction"] == {"a": 0.2}


def test_chesapeake_card_has_the_request_11_fields():
    rec = recon("m-chesapeake")
    assert rec["majority_class_baseline"] == pytest.approx(0.0527, abs=1e-4)
    assert rec["majority_class_baseline_detail"] == {"predict": "tree canopy", "metric": "miou"}
    flat = rec["flat_tile_fraction"]
    assert flat["no data"] == 1.0 and flat["water"] == pytest.approx(0.118, abs=1e-3)
    assert max(v for k, v in flat.items() if k not in ("no data", "water")) <= 0.0024
    notes = data.to_dataset_card(rec).notes
    assert "Majority-class baseline on val" in notes and "no data 100.0%" in notes
