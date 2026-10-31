"""Tools: run training trials, promote a run, read the leaderboard. Plus the budget ledger.

Every trial is scored on VALIDATION. Trials run one at a time and each one is
charged to the task's GPU budget (registry/<task>/budget.json).
"""

import time
from typing import Any

from pydantic import ValidationError

from gotrain.contracts import DatasetCard, TrainConfig, TrainResult, format_errors
from gotrain.engine import get_engine
from gotrain.engine.combos import check_combo
from gotrain.tools._common import (
    append_entry,
    error,
    find_entry,
    git_commit,
    group_of,
    group_stats,
    load_entries,
    load_seed_groups,
    mean_std_text,
    new_run_id,
    now_utc,
    rank,
    read_json,
    run_key,
    short_reason,
    storage,
    summarize_config,
    tool,
    unknown_run,
    write_json,
)

EPSILON_MIN = 1e-9


# ---------------------------------------------------------------- budget ledger


def budget_key(task: str) -> str:
    return f"registry/{task}/budget.json"


def budget_status(task: str) -> dict:
    """{"used_min", "remaining_min"} for a task; zeros if no budget was set."""
    if not storage().exists(budget_key(task)):
        return {"used_min": 0.0, "remaining_min": 0.0}
    ledger = read_json(budget_key(task))
    used = ledger["used_min"]
    return {"used_min": round(used, 3), "remaining_min": round(ledger["total_min"] - used, 3)}


@tool
def init_task_budget(task: str, total_minutes: float) -> dict:
    """Set a task's total GPU budget. For humans and scripts, not exposed to the agent."""
    if not total_minutes > 0:
        return error(f"total_minutes must be > 0, got {total_minutes!r}", "Pass e.g. 120.")
    key = budget_key(task)
    if storage().exists(key):
        ledger = read_json(key)
        if ledger["total_min"] == total_minutes:
            return {"task": task, "total_min": total_minutes, "created": False}
        return error(
            f"budget for {task} is already {ledger['total_min']} min "
            f"({ledger['used_min']:.1f} used); it can't be changed mid-task",
            f"A human can delete {key} to reset it on purpose.",
        )
    write_json(key, {"task": task, "total_min": total_minutes, "used_min": 0.0})
    return {"task": task, "total_min": total_minutes, "created": True}


def _budget_problem(task: str, minutes: float) -> dict | None:
    if not storage().exists(budget_key(task)):
        return error(
            f"no GPU budget set for task {task!r}",
            "A human must call init_task_budget(task, total_minutes) first. "
            "Check the task name is spelled exactly right.",
        )
    remaining = budget_status(task)["remaining_min"]
    if minutes > remaining + EPSILON_MIN:
        return error(
            f"trial needs {minutes:g} min but only {remaining:g} min of budget remain for {task}",
            "Use a smaller per-trial budget (or train_subset_frac), or stop exploring and "
            "register the best run so far.",
        )
    return None


def _charge(task: str, minutes: float) -> None:
    ledger = read_json(budget_key(task))
    ledger["used_min"] = ledger["used_min"] + minutes
    write_json(budget_key(task), ledger)


# ---------------------------------------------------------------- running one trial


def _train(config: TrainConfig, run_id: str, budget_minutes: float) -> TrainResult:
    run_dir = storage().local_path(run_key(config.task, run_id))
    started = time.monotonic()
    try:
        return get_engine().train(config, budget_minutes, run_dir)
    except Exception as exc:  # an engine crash is a failed trial, not a tool failure
        return TrainResult(
            run_id=run_id,
            task=config.task,
            status="error",
            metric_name="unknown",
            minutes_used=(time.monotonic() - started) / 60,
            error=f"engine crashed: {type(exc).__name__}: {exc}",
        )


def _execute_trial(
    config: TrainConfig, budget_minutes: float, parent: str | None, **extra: Any
) -> dict:
    """Train once, save everything under runs/<task>/<run_id>/, charge budget, log entry.

    `extra` adds fields to the leaderboard entry (e.g. rung, halving_id, seed_group).
    """
    run_id = new_run_id(config)
    key = run_key(config.task, run_id)
    storage().write_text(f"{key}/config.json", config.model_dump_json(indent=2))
    storage().write_text(f"{key}/git_commit", git_commit() + "\n")

    result = _train(config, run_id, budget_minutes)
    storage().write_text(f"{key}/result.json", result.model_dump_json(indent=2))
    _charge(config.task, result.minutes_used)

    checkpoint = None
    if result.checkpoint_path is not None:
        run_dir = storage().local_path(key)
        checkpoint = result.checkpoint_path.relative_to(run_dir).as_posix()
    entry = {
        "run_id": run_id,
        "task": config.task,
        "status": result.status,
        "metric_name": result.metric_name,
        "val_score": result.val_score,
        "minutes": round(result.minutes_used, 3),
        "budget_minutes": budget_minutes,
        "epochs": len(result.history),
        "max_epochs": config.max_epochs,
        "config": summarize_config(config),
        "reason": short_reason(config.reason),
        "parent": parent,
        "checkpoint": checkpoint,
        "error": result.error,
        "created_at": now_utc().isoformat(),
        **extra,
    }
    append_entry(config.task, entry)
    return entry


def _row(entry: dict, *extra_keys: str) -> dict:
    """The compact view of one trial that tools return (and the audit log records)."""
    keys = ("run_id", *extra_keys, "status", "val_score", "minutes", "reason")
    row = {k: entry.get(k) for k in keys}
    if entry.get("error"):
        row["error"] = entry["error"]
    return row


def _parse_config(task: str, index: int, raw: Any, card: DatasetCard) -> TrainConfig | dict:
    """A valid TrainConfig, or {"index", "error"} explaining what is wrong with it."""
    if not isinstance(raw, dict):
        return {"index": index, "error": f"config must be an object, got {type(raw).__name__}"}
    raw = {"task": task, **raw}
    if raw["task"] != task:
        return {"index": index, "error": f"config is for {raw['task']!r}, but task is {task!r}"}
    try:
        config = TrainConfig(**raw)
    except ValidationError as exc:
        return {"index": index, "error": format_errors(exc, TrainConfig)}
    if problem := check_combo(card, config.backbone, config.weights):
        return {"index": index, "error": problem}
    return config


def _dataset_card(task: str) -> DatasetCard | dict:
    """The task's DatasetCard, or an error dict if the engine doesn't know the task."""
    try:
        return get_engine().describe_dataset(task)
    except (ValueError, FileNotFoundError) as exc:
        return error(str(exc), "Use the exact GEO-Bench task name, e.g. 'm-eurosat'.")


# ---------------------------------------------------------------- tools


@tool
def run_trials(task: str, configs: list[dict], budget_minutes_per_trial: float) -> dict:
    """Validate configs, run the valid ones one by one, return them ranked by val score.

    Invalid configs are reported with the reason and skipped; the rest still run.
    A trial that would exceed the remaining budget is not started.

    Args:
        task: GEO-Bench task name, e.g. "m-eurosat".
        configs: List of training configs (see the field list below). "task" may be omitted.
        budget_minutes_per_trial: GPU minutes each trial may use; charged to the task budget.
    """
    if not isinstance(configs, list) or not configs:
        return error("configs must be a non-empty list of config objects", "Pass 1-6 configs.")
    if not budget_minutes_per_trial > 0:
        return error(
            f"budget_minutes_per_trial must be > 0, got {budget_minutes_per_trial!r}",
            "Typical scouting trials use 2-10 minutes.",
        )
    if problem := _budget_problem(task, budget_minutes_per_trial):
        return problem
    card = _dataset_card(task)
    if isinstance(card, dict):
        return card

    parsed = [_parse_config(task, i, raw, card) for i, raw in enumerate(configs)]
    invalid = [p for p in parsed if isinstance(p, dict)]
    ran, skipped = [], []
    for index, config in enumerate(parsed):
        if not isinstance(config, TrainConfig):
            continue
        if problem := _budget_problem(task, budget_minutes_per_trial):
            skipped.append({"index": index, "reason": problem["error"]})
            continue
        ran.append(_execute_trial(config, budget_minutes_per_trial, parent=None))

    best = rank(load_entries(task))[0] if load_entries(task) else None
    out = {
        "task": task,
        "trials": [_row(e) for e in rank(ran)],
        "invalid": invalid,
        "skipped": skipped,
        "best_overall": _row(best) if best else None,
    }
    if invalid:
        out["hint"] = "Fix the invalid configs using the messages above and resubmit only those."
    return out


@tool
def promote(run_id: str, extra_budget_minutes: float) -> dict:
    """Re-train a run's exact config with its budget + extra_budget_minutes, as a child run.

    Args:
        run_id: A successful run that stopped before max_epochs (from run_trials/leaderboard).
        extra_budget_minutes: Minutes added to the parent's per-trial budget.
    """
    parent = find_entry(run_id)
    if parent is None:
        return unknown_run(run_id)
    if parent["status"] != "ok":
        return error(
            f"run {run_id} has status {parent['status']!r}; only successful runs can be promoted",
            "Promote one of the top 'ok' runs from leaderboard().",
        )
    if parent["epochs"] >= parent["max_epochs"]:
        return error(
            f"run {run_id} already finished all {parent['max_epochs']} epochs; "
            "more time would reproduce the same result",
            "Submit the config via run_trials with a higher max_epochs instead.",
        )
    if not extra_budget_minutes > 0:
        return error(
            f"extra_budget_minutes must be > 0, got {extra_budget_minutes!r}", "Pass e.g. 10."
        )

    task = parent["task"]
    budget = parent["budget_minutes"] + extra_budget_minutes
    if problem := _budget_problem(task, budget):
        return problem
    config = TrainConfig.model_validate_json(
        storage().read_text(f"{run_key(task, run_id)}/config.json")
    )
    card = _dataset_card(task)
    if isinstance(card, dict):
        return card
    if problem := check_combo(card, config.backbone, config.weights):
        return error(problem, "Submit a config with a valid combination via run_trials instead.")
    original = config.reason.split("Original reason: ")[-1]
    config = config.model_copy(
        update={"reason": f"Promoted {run_id} to {budget:g} min. Original reason: {original}"}
    )
    child = _execute_trial(config, budget, parent=run_id)
    delta = None
    if child["val_score"] is not None:
        delta = round(child["val_score"] - parent["val_score"], 4)
    return {
        "task": task,
        "parent": {k: parent[k] for k in ("run_id", "val_score", "minutes", "epochs")},
        "child": {k: child[k] for k in ("run_id", "status", "val_score", "minutes", "epochs")},
        "delta_val_score": delta,
    }


@tool
def leaderboard(task: str, top_k: int = 10) -> dict:
    """The task's runs ranked by val score: config summary, score, minutes, status, parent.

    Runs in a seed group also show the group's mean ± std; "seed_groups" ranks the groups
    by mean score.

    Args:
        task: GEO-Bench task name.
        top_k: How many runs to return (1-100).
    """
    if not isinstance(top_k, int) or not 1 <= top_k <= 100:
        return error(f"top_k must be an integer from 1 to 100, got {top_k!r}", "Use top_k=10.")
    entries = rank(load_entries(task))
    groups = load_seed_groups(task)
    stats = {gid: group_stats(g["run_ids"], entries) for gid, g in groups.items()}
    top = []
    for i, e in enumerate(entries[:top_k], start=1):
        row = {"rank": i}
        row |= {k: e[k] for k in ("run_id", "status", "val_score", "minutes", "config", "parent")}
        if gid := group_of(e["run_id"], groups):
            row |= {"seed_group": gid, "mean_std": mean_std_text(stats[gid])}
        top.append(row)
    out = {"task": task, "n_runs": len(entries), "metric_name": None, "top": top}
    if groups:
        out["seed_groups"] = sorted(
            (
                {"seed_group": gid, "base_run_id": groups[gid]["base_run_id"]}
                | {"mean_std": mean_std_text(st), "mean": st["mean"], "std": st["std"]}
                for gid, st in stats.items()
            ),
            key=lambda g: -(g["mean"] or 0.0),
        )
    if entries:
        out["metric_name"] = entries[0]["metric_name"]
    else:
        out["hint"] = "No runs yet for this task. Start with run_halving or run_trials."
    return out
