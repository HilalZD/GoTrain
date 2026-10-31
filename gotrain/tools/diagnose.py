"""Tool: explain how a finished run fails, from the engine's diagnose() output.

`diagnose` is a plain tool returning a JSON-serializable dict. If the engine drew an
error grid, the tool downscales it into the run folder and reports its metadata.
`diagnose_for_model` is the one place an image reaches the agent: it wraps the same
result as a Strands tool result with a text block plus a PNG image block.
"""

import functools
import json
from pathlib import Path

from gotrain.engine import get_engine
from gotrain.images import ImageError, prepare_image
from gotrain.tools._common import error, find_entry, run_key, storage, tool, unknown_run

N_CLASSES_SHOWN = 3
# The per-class metric to rank classes by, chosen by name (engines may add more keys).
PER_CLASS_METRICS = ("iou", "recall", "f1")
N_CONFUSIONS_SHOWN = 3
VERDICT_HINTS = {
    "overfitting": "Val peaked early then fell: add augmentation, weight decay, or fewer epochs.",
    "underfitting": "Val was still rising: give it more epochs/budget (promote) or a higher lr.",
    "fine": "Curve looks healthy: gains now come from weights, augmentation or loss choice.",
}
MODEL_GRID_NAME = "diagnostics/error_grid_for_model.png"
# After a diagnose, the tune agent must write both markers before any other tool call
# (enforced by gotrain.agent.ObservationGate).
OBSERVATION_MARKERS = ("VISUAL OBSERVATION", "HYPOTHESIS")
OBSERVATION_REMINDER = "Next: write VISUAL OBSERVATION and HYPOTHESIS before any other tool call."


def top_confusions(class_names: list[str], matrix: list[list[float]]) -> list[dict]:
    pairs = [
        {"true": class_names[i], "predicted": class_names[j], "rate": rate}
        for i, row in enumerate(matrix)
        for j, rate in enumerate(row)
        if i != j and rate > 0
    ]
    return sorted(pairs, key=lambda p: -p["rate"])[:N_CONFUSIONS_SHOWN]


def summarize_diagnosis(raw: dict) -> dict:
    """Keep the worst/best classes and biggest confusions, not the full matrix."""
    first_class_metrics = next(iter(raw["per_class"].values()))  # e.g. {"recall": 0.91, ...}
    metric = next((m for m in PER_CLASS_METRICS if m in first_class_metrics), None)
    metric = metric or next(iter(first_class_metrics))  # unknown engine: its first metric
    per_class = {name: m[metric] for name, m in raw["per_class"].items()}
    ordered = sorted(per_class.items(), key=lambda kv: kv[1])
    return {
        "curve_verdict": raw["curve_verdict"],
        "advice": VERDICT_HINTS.get(raw["curve_verdict"], ""),
        "per_class_metric": metric,
        "weakest_classes": dict(ordered[:N_CLASSES_SHOWN]),
        "strongest_classes": dict(ordered[::-1][:N_CLASSES_SHOWN]),
        "top_confusions": top_confusions(raw["class_names"], raw["confusion_matrix"]),
    }


def prepare_error_grid(grid_path: str | None, run: str) -> dict:
    """Downscale the engine's error grid into the run folder; metadata only, never bytes."""
    if not grid_path:
        return {"error_grid": None}
    try:
        prepared = prepare_image(Path(grid_path))
    except ImageError as exc:
        return {"error_grid": None, "image_note": f"error grid not shown, text only: {exc}"}
    key = f"{run}/{MODEL_GRID_NAME}"
    storage().write_bytes(key, prepared.png)
    return {"error_grid": {"key": key, "source_path": str(grid_path), **prepared.info()}}


@tool
def diagnose(run_id: str) -> dict:
    """Curve verdict (over/underfitting), weakest classes and biggest confusions of a run.

    If the engine drew an error grid (worst validation predictions: image | truth |
    prediction), it is attached as an image.

    Args:
        run_id: A successful run, copied exactly from run_trials or leaderboard.
    """
    entry = find_entry(run_id)
    if entry is None:
        return unknown_run(run_id)
    if entry["status"] != "ok":
        return error(
            f"run {run_id} has status {entry['status']!r} and no model to diagnose",
            f"Its error was: {entry.get('error')}. Fix the config and run it again.",
        )
    task = entry["task"]
    run = run_key(task, run_id)
    raw = get_engine().diagnose(task, storage().local_path(run))
    return {
        "run_id": run_id,
        "val_score": entry["val_score"],
        **summarize_diagnosis(raw),
        **prepare_error_grid(raw.get("error_grid_path"), run),
    }


@functools.wraps(diagnose)
def diagnose_for_model(run_id: str) -> dict:
    """`diagnose` as a Strands tool result: the JSON summary, plus the error grid as a PNG."""
    out = diagnose(run_id)
    if "error" not in out:
        out["next_step"] = OBSERVATION_REMINDER  # last key: the end of the text result
    grid = out.get("error_grid")
    png = None
    if grid:
        try:
            png = storage().read_bytes(grid["key"])
        except OSError as exc:
            out["error_grid"] = None
            out["image_note"] = f"error grid not shown, text only: {exc}"
    content: list[dict] = [{"text": json.dumps(out, default=str)}]
    if png is not None:
        content.append({"image": {"format": "png", "source": {"bytes": png}}})
    return {"status": "success", "content": content}
