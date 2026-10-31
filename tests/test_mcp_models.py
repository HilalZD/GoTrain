"""The models MCP server: list_models, classification, segmentation, clear band errors."""

import json

import numpy as np
import pytest
from conftest import make_config
from test_tools_models import budget  # noqa: F401, F811

from gotrain.gopilot import export_bundle, list_bundles
from gotrain.mcp.models_server import build_server, list_models, run_model
from gotrain.tools import register_model, run_trials


def export_task(task: str, minutes: float = 2.0, **config) -> dict:
    run_id = run_trials(task, [make_config(**config)], minutes)["trials"][0]["run_id"]
    version = register_model(run_id)["version"]
    return export_bundle(task, version)


def test_list_models_and_server_registration(budget):  # noqa: F811
    assert list_models() == {"n_models": 0, "models": []}
    card = export_task("m-eurosat")
    listed = list_models()
    assert listed["n_models"] == 1
    assert listed["models"][0]["tool_name"] == "classify_land_cover_eurosat"
    assert listed["models"][0]["bands"] == card["bands"]
    build_server()  # registers one tool per bundle plus list_models without raising


def test_classification_from_npy(budget, tmp_path):  # noqa: F811
    card = export_task("m-eurosat")
    path = tmp_path / "tile.npy"
    np.save(path, np.random.default_rng(1).uniform(0, 3000, (len(card["bands"]), 64, 64)))
    out = run_model(card, str(path))
    assert out["label"] in card["class_names"] and 0 < out["confidence"] <= 1
    assert set(out["probabilities"]) == set(card["class_names"])
    assert abs(sum(out["probabilities"].values()) - 1) < 0.01


def test_wrong_band_count_returns_a_clear_error_not_a_crash(budget, tmp_path):  # noqa: F811
    card = export_task("m-eurosat")
    path = tmp_path / "bad.npy"
    np.save(path, np.zeros((3, 64, 64)))
    out = run_model(card, str(path))
    assert "needs N x 13 x H x W" in out["error"] and "classify_land_cover_eurosat" in out["hint"]
    assert "no such file" in run_model(card, str(tmp_path / "missing.npy"))["error"]


def test_segmentation_returns_mask_geojson_and_areas(budget, tmp_path, monkeypatch):  # noqa: F811
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    monkeypatch.setenv("GOTRAIN_MCP_OUT", str(tmp_path / "out"))
    card = export_task(
        "m-pv4ger-seg", 10.0, weights="imagenet", img_size=320, loss="dice_bce", batch_size=16
    )
    n = len(card["bands"])
    path = tmp_path / "tile.tif"
    data = np.random.default_rng(2).uniform(0, 255, (n, 64, 64)).astype("float32")
    profile = {"driver": "GTiff", "height": 64, "width": 64, "count": n, "dtype": "float32"}
    profile |= {"crs": "EPSG:32633", "transform": from_origin(500000, 4000000, 0.5, 0.5)}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)

    out = run_model(card, str(path))
    assert "error" not in out, out
    assert out["georeferenced"] and out["geojson"]["type"] == "FeatureCollection"
    total = sum(v["pixels"] for v in out["area_per_class"].values())
    assert total == 64 * 64 and all("area_m2" in v for v in out["area_per_class"].values())
    with rasterio.open(out["mask_geotiff"]) as mask:
        assert mask.shape == (64, 64) and str(mask.crs) == "EPSG:32633"
    json.dumps(out)  # the whole answer is JSON-serialisable
    assert list_bundles()
