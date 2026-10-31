"""MCP server for the exported models (GoPilot stand-in interface), over stdio.

    GOTRAIN_HOME=~/gotrain_home python -m gotrain.mcp.models_server

One tool per exported bundle (name and description generated from its card, e.g.
detect_brick_kilns) plus list_models. Input: a GeoTIFF path or a .npy file (bands in the card's
order). Classification returns label, probabilities and confidence; segmentation returns a mask
GeoTIFF, GeoJSON polygons of the target classes and the area per class. `needs_review` is passed
on when the engine's predict() returns it. Wrong band count: a clear error, not a crash.
"""

import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from gotrain.gopilot import list_bundles, predict_bundle

BACKGROUND = re.compile(r"background|no[ _-]?data|unknown|^no\b|none|nodata", re.IGNORECASE)
MAX_POLYGONS = 200


def out_dir() -> Path:
    path = Path(os.environ.get("GOTRAIN_MCP_OUT") or tempfile.gettempdir()) / "gotrain_mcp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_image(path: str) -> tuple[np.ndarray, dict]:
    """(1 x C x H x W float32, georeferencing {"transform", "crs"} or {}) from .npy or GeoTIFF."""
    file = Path(path).expanduser()
    if not file.is_file():
        raise ValueError(f"no such file: {path}")
    if file.suffix.lower() == ".npy":
        array = np.load(file)
        if array.ndim == 3:
            array = array[None]
        return array.astype(np.float32), {}
    import rasterio

    with rasterio.open(file) as src:
        array = src.read().astype(np.float32)
        return array[None], {"transform": src.transform, "crs": src.crs}


def target_classes(class_names: list[str]) -> list[int]:
    targets = [i for i, n in enumerate(class_names) if not BACKGROUND.search(n)]
    return targets or list(range(1, len(class_names)))


def classify(card: dict, probs: np.ndarray, result: dict) -> dict:
    p = probs[0]
    idx = int(np.argmax(p))
    names = card["class_names"]
    out = {
        "label": names[idx],
        "label_index": idx,
        "probabilities": {n: round(float(v), 4) for n, v in zip(names, p, strict=True)},
        "confidence": float(result.get("confidence", p[idx])),
    }
    if "needs_review" in result:
        out["needs_review"] = bool(np.asarray(result["needs_review"]).reshape(-1)[0])
    return out


def pixel_area_m2(geo: dict) -> float | None:
    """Pixel area in m^2 when the GeoTIFF has a projected CRS in metres, else None."""
    crs, transform = geo.get("crs"), geo.get("transform")
    if crs is None or transform is None or not getattr(crs, "is_projected", False):
        return None
    return abs(transform.a * transform.e)


def segment(card: dict, name: str, probs: np.ndarray, result: dict, geo: dict) -> dict:
    from rasterio import features
    from rasterio.transform import Affine

    p = probs[0]  # K x H x W
    mask = p.argmax(axis=0).astype(np.uint8)
    transform = geo.get("transform") or Affine.identity()
    names = card["class_names"]
    mask_path = out_dir() / f"{name}_mask.tif"
    import rasterio

    profile = {"driver": "GTiff", "height": mask.shape[0], "width": mask.shape[1], "count": 1}
    profile |= {"dtype": "uint8", "transform": transform}
    if geo.get("crs") is not None:
        profile["crs"] = geo["crs"]
    with rasterio.open(mask_path, "w", **profile) as dst:
        dst.write(mask, 1)

    area_px = pixel_area_m2(geo)
    per_class, polygons, n_polygons = {}, [], 0
    for cls in range(len(names)):
        pixels = int((mask == cls).sum())
        per_class[names[cls]] = {"pixels": pixels, "fraction": round(pixels / mask.size, 4)}
        if area_px is not None:
            per_class[names[cls]]["area_m2"] = round(pixels * area_px, 2)
    for cls in target_classes(names):
        shapes = features.shapes(mask, mask=mask == cls, transform=transform)
        for geometry, _ in shapes:
            n_polygons += 1
            if len(polygons) < MAX_POLYGONS:
                polygons.append(
                    {"type": "Feature", "properties": {"class": names[cls]}, "geometry": geometry}
                )
    out = {
        "mask_geotiff": str(mask_path),
        "geojson": {"type": "FeatureCollection", "features": polygons},
        "n_polygons": n_polygons,
        "polygons_truncated": n_polygons > len(polygons),
        "area_per_class": per_class,
        "georeferenced": bool(geo),
        "confidence": float(result.get("confidence", p.max(axis=0).mean())),
    }
    if "needs_review" in result:
        out["needs_review"] = bool(np.asarray(result["needs_review"]).reshape(-1)[0])
    return out


def run_model(card: dict, path: str) -> dict:
    """Run one exported model on one image file. Errors come back as {"error": ...}."""
    try:
        images, geo = load_image(path)
        result = predict_bundle(card, images)
    except (ValueError, OSError) as exc:
        return {"error": str(exc), "hint": f"{card['tool_name']}: {card['description']}"}
    probs = np.asarray(result["probabilities"])
    if card["task_type"] == "segmentation":
        out = segment(card, card["tool_name"], probs, result, geo)
    else:
        out = classify(card, probs, result)
    return {"model": card["tool_name"], "version_hash": card["version_hash"], **out}


def list_models() -> dict:
    """The exported models: tool name, what it does, inputs, and its validation/test scores."""
    keys = ("tool_name", "task", "version", "description", "task_type", "sensor", "bands")
    models = [
        {k: c[k] for k in keys} | {"val": c["val"], "test": c["test"]} for c in list_bundles()
    ]
    return {"n_models": len(models), "models": models}


def make_tool(card: dict) -> Any:
    def tool_fn(path: str) -> dict:
        return run_model(card, path)

    tool_fn.__name__ = card["tool_name"]
    return tool_fn


def build_server() -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:  # mcp 2.x renamed it
        from mcp.server.mcpserver import MCPServer as FastMCP
    server = FastMCP("gotrain-models")
    server.add_tool(list_models, name="list_models", description=list_models.__doc__)
    for card in list_bundles():
        description = f"{card['description']} Pass the image file path (GeoTIFF or .npy)."
        server.add_tool(make_tool(card), name=card["tool_name"], description=description)
    return server


if __name__ == "__main__":
    build_server().run(transport="stdio")
