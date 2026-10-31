"""Tool: explain how a finished run fails, from the engine's diagnose() output."""

from gotrain.engine import get_engine
from gotrain.tools._common import error, find_entry, run_key, storage, tool, unknown_run

N_CLASSES_SHOWN = 3
N_CONFUSIONS_SHOWN = 3
VERDICT_HINTS = {
    "overfitting": "Val peaked early then fell: add augmentation, weight decay, or fewer epochs.",
    "underfitting": "Val was still rising: give it more epochs/budget (promote) or a higher lr.",
    "fine": "Curve looks healthy: gains now come from weights, augmentation or loss choice.",
}


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
    first_class_metrics = next(iter(raw["per_class"].values()))  # e.g. {"recall": 0.91}
    metric = next(iter(first_class_metrics))
    per_class = {name: m[metric] for name, m in raw["per_class"].items()}
    ordered = sorted(per_class.items(), key=lambda kv: kv[1])
    return {
        "curve_verdict": raw["curve_verdict"],
        "advice": VERDICT_HINTS.get(raw["curve_verdict"], ""),
        "per_class_metric": metric,
        "weakest_classes": dict(ordered[:N_CLASSES_SHOWN]),
        "strongest_classes": dict(ordered[::-1][:N_CLASSES_SHOWN]),
        "top_confusions": top_confusions(raw["class_names"], raw["confusion_matrix"]),
        "error_grid_path": raw.get("error_grid_path"),
    }


@tool
def diagnose(run_id: str) -> dict:
    """Curve verdict (over/underfitting), weakest classes and biggest confusions of a run."""
    entry = find_entry(run_id)
    if entry is None:
        return unknown_run(run_id)
    if entry["status"] != "ok":
        return error(
            f"run {run_id} has status {entry['status']!r} and no model to diagnose",
            f"Its error was: {entry.get('error')}. Fix the config and run it again.",
        )
    task = entry["task"]
    run_dir = storage().local_path(run_key(task, run_id))
    raw = get_engine().diagnose(task, run_dir)
    return {"run_id": run_id, "val_score": entry["val_score"], **summarize_diagnosis(raw)}
