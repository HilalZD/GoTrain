"""Tool: summarise a task's dataset (from train/val only) plus relevant past lessons."""

from gotrain.contracts import DatasetCard
from gotrain.engine import get_engine
from gotrain.engine.combos import valid_combos_for
from gotrain.engine.rules import rules_for
from gotrain.engine.throughput import screening_advice
from gotrain.tools._common import error, tool
from gotrain.tools.memory import matching_lessons

MAX_LISTED_CLASSES = 20


def summarize_card(card: DatasetCard) -> dict:
    """The DatasetCard without per-band stats, plus derived flags the agent should notice."""
    fractions = card.class_distribution
    summary = {
        "task": card.task,
        "task_type": card.task_type,
        "sensor": card.sensor,
        "n_bands": len(card.bands),
        "bands": card.bands,
        "n_classes": card.n_classes,
        "image_size": list(card.image_size),
        "split_sizes": card.split_sizes,
        "notes": card.notes,
    }
    if card.n_classes <= MAX_LISTED_CLASSES:
        summary["class_distribution"] = {k: round(v, 3) for k, v in fractions.items()}
    if fractions and min(fractions.values()) > 0:
        summary["imbalance_ratio"] = round(max(fractions.values()) / min(fractions.values()), 1)
    optional = ("dtype", "value_range", "label_format", "nodata_fraction", "constant_bands")
    summary.update({k: getattr(card, k) for k in optional if getattr(card, k) is not None})
    summary["valid_backbones_by_weights"] = valid_combos_for(card)
    summary["training_rules"] = rules_for(card)
    if hint := screening_advice(card):
        summary["screening_hint"] = hint
    if card.sensor == "sentinel-2":
        summary["weights_hint"] = "Sentinel-2 task: default to weights='ssl4eo_s2'."
    return summary


@tool
def inspect_dataset(task: str) -> dict:
    """DatasetCard summary for `task` and the lessons tagged with its sensor or task type.

    Args:
        task: GEO-Bench task name, e.g. "m-eurosat".
    """
    try:
        card = get_engine().describe_dataset(task)
    except (ValueError, FileNotFoundError) as exc:
        return error(str(exc), "Use the exact GEO-Bench task name, e.g. 'm-eurosat'.")
    lessons = matching_lessons([card.sensor, card.task_type, card.task])
    return {"dataset": summarize_card(card), "lessons": lessons}
