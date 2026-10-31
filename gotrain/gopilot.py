"""GoPilot-ready export: a registered version as a self-describing bundle with a predict().

    registry/<task>/<version>/gopilot/bundle.json    checkpoints, TTA/ensemble choice, card
    registry/<task>/<version>/gopilot/model_card.md  the same card for people

The model is exactly the one that won on VALIDATION (members + TTA as registered). Predictions
go through the engine's predict() (contract v2: raw N x C x H x W in the card's band order).
No TorchScript/ONNX and no tiling: the input is one image of the model's own size.
"""

import hashlib
import json
import re

import numpy as np

from gotrain.config import get_settings
from gotrain.engine import get_engine
from gotrain.tools._common import error, find_entry, now_utc, read_json, storage, tool, write_json
from gotrain.tools.final_test import REHEARSAL_LABEL, lock_key, rehearsal_key
from gotrain.tools.registry import list_versions, load_version, member_key

BUNDLE_VERSION = 1

# (tool name, what it finds) per known task; unknown tasks get a generic name
TOOL_NAMES = {
    "m-eurosat": ("classify_land_cover_eurosat", "land cover class of a Sentinel-2 tile"),
    "m-brick-kiln": ("detect_brick_kilns", "whether a Sentinel-2 tile contains a brick kiln"),
    "m-pv4ger-seg": ("segment_solar_panels", "solar panels in an aerial image"),
    "m-chesapeake": ("map_land_cover_chesapeake", "land cover map of an aerial image"),
}


def tool_name_for(task: str, task_type: str) -> str:
    if task in TOOL_NAMES:
        return TOOL_NAMES[task][0]
    verb = "segment" if task_type == "segmentation" else "classify"
    return f"{verb}_{re.sub(r'[^a-z0-9]+', '_', task.lower()).strip('_')}"


def bundle_key(task: str, version: str) -> str:
    return f"registry/{task}/{version}/gopilot"


def version_hash(parts: dict) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]


def test_score_for(task: str, version: str) -> dict | None:
    key = lock_key(task)
    if not storage().exists(key):
        return None
    lock = read_json(key)
    if lock.get("status") != "done" or lock.get("version") != version:
        return None
    keep = ("test_score", "metric_name", "ci_low", "ci_high", "ci_level")
    return {k: lock[k] for k in keep if k in lock}


def calibration_for(task: str, run_id: str) -> dict | None:
    """Calibration info the engine wrote into the run's result.json, if any."""
    try:
        result = read_json(f"runs/{task}/{run_id}/result.json")
    except (OSError, ValueError):
        return None
    return result.get("calibration")


def describe(card, hint: str) -> str:
    classes = ", ".join(card.class_names)
    return (
        f"{hint.capitalize()}. Input: a {card.sensor} image with {len(card.bands)} bands in this "
        f"order ({', '.join(card.bands)}), {card.image_size[0]}x{card.image_size[1]} px, as a "
        f"GeoTIFF path or a .npy file. Classes: {classes}."
    )


def checkpoint_keys(task: str, version: str, members: list[str]) -> list[str]:
    keys = []
    for rid in members:
        entry = find_entry(rid) or {}
        base = member_key(task, version, rid, len(members))
        keys.append(f"{base}/{entry['checkpoint']}")
    return keys


def build_card(task: str, version: str) -> dict:
    mv = load_version(task, version)
    card = get_engine().describe_dataset(task)  # corrected class names, never geobench labels
    hint = TOOL_NAMES.get(task, (None, f"{card.task_type} on {task}"))[1]
    entry = find_entry(mv.run_id) or {}
    identity = {
        "task": task,
        "members": mv.members,
        "tta": mv.tta,
        "commit": mv.git_commit,
        "val": mv.val_score,
    }
    return {
        "bundle_version": BUNDLE_VERSION,
        "task": task,
        "version": version,
        "tool_name": tool_name_for(task, card.task_type),
        "description": describe(card, hint),
        "task_type": card.task_type,
        "sensor": card.sensor,
        "bands": card.bands,
        "image_size": list(card.image_size),
        "n_classes": card.n_classes,
        "class_names": card.class_names,
        "members": mv.members,
        "tta": mv.tta,
        "checkpoints": checkpoint_keys(task, version, mv.members),
        "val": {
            "metric": entry.get("metric_name"),
            "score": mv.val_score,
            "std": mv.val_std,
            "n_seeds": mv.val_n,
        },
        "test": test_score_for(task, version),  # None until final_test ran
        "calibration": calibration_for(task, mv.run_id),
        "version_hash": version_hash(identity),
        "exported_at": now_utc().isoformat(),
    }


def card_markdown(c: dict) -> str:
    test = c["test"]
    test_line = (
        f"{test['test_score']:.4f} (95% CI {test.get('ci_low')}-{test.get('ci_high')})"
        if test
        else "not scored yet (final_test has not run)"
    )
    model = f"ensemble of {len(c['members'])}" if len(c["members"]) > 1 else "single model"
    calibration = json.dumps(c["calibration"]) if c["calibration"] else "none"
    classes = ", ".join(f"{i}={n}" for i, n in enumerate(c["class_names"]))
    rehearsal = c.get("rehearsal")
    marker = (
        f"**{REHEARSAL_LABEL}**: scored on validation ({rehearsal['test_score']:.4f}); "
        "this is not a test result.\n\n"
        if rehearsal
        else ""
    )
    return (
        f"# {c['tool_name']}  ({c['task']} {c['version']})\n\n{marker}"
        f"{c['description']}\n\n"
        f"- Sensor: {c['sensor']}; bands (in order): {', '.join(c['bands'])}\n"
        f"- Image size: {c['image_size'][0]}x{c['image_size'][1]} px; task type: {c['task_type']}\n"
        f"- Classes: {classes}\n"
        f"- Model: {model}{' with TTA' if c['tta'] else ''}, chosen on validation\n"
        f"- Validation {c['val']['metric']}: {c['val']['score']:.4f}\n"
        f"- Test: {test_line}\n"
        f"- Calibration: {calibration}\n"
        f"- Version hash: `{c['version_hash']}`\n"
    )


def export_bundle(task: str, version: str) -> dict:
    """Write the bundle for a registered version (again after finalise: adds the test score)."""
    if load_version(task, version) is None:
        raise ValueError(f"no registered version {version!r} for {task}")
    card = build_card(task, version)
    if get_settings().rehearsal:  # scored on validation: marked everywhere, never a test result
        card["rehearsal"] = {"label": REHEARSAL_LABEL, **read_json(rehearsal_key(task))}
    key = bundle_key(task, version)
    write_json(f"{key}/bundle.json", card)
    storage().write_text(f"{key}/model_card.md", card_markdown(card))
    return card


@tool
def export_model(task: str, version: str) -> dict:
    """Export a registered version as a GoPilot-ready bundle (card + predict()).

    Args:
        task: GEO-Bench task name.
        version: Registered version, e.g. "v1".
    """
    if version not in list_versions(task):
        return error(
            f"no registered version {version!r} for {task}",
            f"Registered versions: {list_versions(task) or 'none'}.",
        )
    card = export_bundle(task, version)
    return {
        "task": task,
        "version": version,
        "tool_name": card["tool_name"],
        "bundle": bundle_key(task, version),
        "version_hash": card["version_hash"],
    }


def list_bundles() -> list[dict]:
    """Every exported bundle card in the registry, sorted by tool name."""
    keys = [k for k in storage().list("registry") if k.endswith("/gopilot/bundle.json")]
    return sorted((read_json(k) for k in keys), key=lambda c: c["tool_name"])


def local_checkpoint(key: str):
    """A checkpoint key as a local file (local_path would create a directory at the key)."""
    parent, _, name = key.rpartition("/")
    return storage().local_path(parent) / name


def predict_bundle(card: dict, images: np.ndarray) -> dict:
    """The bundle's model on raw N x C x H x W images (card band order): engine predict()."""
    images = np.asarray(images, dtype=np.float32)
    if images.ndim != 4 or images.shape[1] != len(card["bands"]):
        raise ValueError(
            f"{card['tool_name']} needs N x {len(card['bands'])} x H x W images with bands "
            f"{card['bands']}, got shape {images.shape}"
        )
    paths = [local_checkpoint(k) for k in card["checkpoints"]]
    checkpoints = paths[0] if len(paths) == 1 else paths
    extra = {"tta": True} if card["tta"] else {}
    return get_engine().predict(card["task"], checkpoints, images, **extra)
