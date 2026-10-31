"""Tools for smarter GPU use: successive halving and seed confirmation.

Both are built from the same pieces as run_trials/promote: every trial goes through
_execute_trial (run folder, budget charge, leaderboard entry), and every trial
appears as a row in the output's "trials" list, which the audit log records.
"""

import uuid

from gotrain.contracts import TrainConfig
from gotrain.engine.combos import check_combo
from gotrain.tools._common import (
    error,
    find_entry,
    group_of,
    group_stats,
    load_entries,
    load_seed_groups,
    run_key,
    save_seed_groups,
    storage,
    tool,
    unknown_run,
)
from gotrain.tools.trials import (
    EPSILON_MIN,
    SEED_CHECK_SEEDS,
    _budget_problem,
    _dataset_card,
    _execute_trial,
    _parse_config,
    _row,
    budget_status,
    seed_check_status,
)

MAX_SEEDS = 5
MIN_SCREENING_SPREAD = 0.005  # below this, rung 1 can't tell configs apart from seed noise
INCONCLUSIVE_WARNING = (
    "rung 1 was inconclusive; consider a longer short_minutes or train_subset_frac"
)
MIN_SCREENING_EPOCHS = 3  # fewer, and rung-1 scores mostly reflect how far training got
HALVING_ROW_KEYS = ("rung", "parent")


def _original_reason(reason: str) -> str:
    return reason.split("Original reason: ")[-1]


def screening_summary(rung1: list[dict]) -> dict:
    """Spread of rung-1 scores (best minus worst) plus warnings if the screen is unreliable.

    Warns when the spread is within seed noise, and when trials finished too few epochs
    to rank configs (errored configs never trained, so they are not counted).
    """
    scores = [e["val_score"] for e in rung1 if e["status"] == "ok"]
    spread = round(max(scores) - min(scores), 4) if len(scores) >= 2 else None
    warnings = []
    if spread is not None and spread < MIN_SCREENING_SPREAD:
        warnings.append(INCONCLUSIVE_WARNING)
    if short := [e for e in rung1 if e["status"] != "error" and e["epochs"] < MIN_SCREENING_EPOCHS]:
        warnings.append(few_epochs_warning(short))
    return {"screening_spread": spread, "warnings": warnings}


def few_epochs_warning(short: list[dict]) -> str:
    runs = ", ".join(f"{e['run_id']} ({e['epochs']})" for e in short)
    fewest = min(e["epochs"] for e in short)
    return (
        f"rung 1 trials finished only {fewest} epochs; screening is unreliable. "
        f"Use train_subset_frac 0.2–0.3 or a longer short_minutes. "
        f"Runs (epochs): {runs}"
    )


def _affordability_hint(remaining: float, n: int, short: float, long: float, keep: int) -> str:
    """Concrete ways to make a halving round fit the remaining budget."""
    options = []
    max_configs = int((remaining - keep * long) // short) if short > 0 else 0
    if keep < max_configs < n:
        options.append(f"run only {max_configs} configs")
    max_long = (remaining - n * short) / keep
    if short < max_long < long:
        options.append(f"lower long_minutes to {max_long:.1f}")
    if keep > 1 and n * short + long <= remaining:
        options.append("use keep_top=1")
    if not options:
        return "Not enough budget for a halving round. Use run_trials or promote, or register now."
    return "To fit the remaining budget: " + ", or ".join(options) + "."


# ---------------------------------------------------------------- successive halving


@tool
def run_halving(
    task: str, configs: list[dict], short_minutes: float, long_minutes: float, keep_top: int
) -> dict:
    """Successive halving: train every config briefly, then re-train only the best ones longer.

    Rung 1 trains each valid config with short_minutes. The keep_top best by val_score are
    re-trained with long_minutes as child runs (parent = their rung-1 run). The whole round's
    cost (n * short + keep_top * long) is checked against the budget before anything starts.
    "screening_spread" is best minus worst rung-1 score; "warnings" flags an unreliable screen
    (spread under 0.005, or rung-1 trials that finished fewer than 3 epochs).

    Args:
        task: GEO-Bench task name, e.g. "m-eurosat".
        configs: 4-8 diverse training configs (see the field list below). "task" may be omitted.
        short_minutes: GPU minutes per config in rung 1 (cheap screening).
        long_minutes: GPU minutes per survivor in rung 2; must be more than short_minutes.
        keep_top: How many rung-1 winners to re-train; fewer than the number of valid configs.
    """
    if not isinstance(configs, list) or not configs:
        return error("configs must be a non-empty list of config objects", "Pass 4-8 configs.")
    if not 0 < short_minutes < long_minutes:
        return error(
            f"need 0 < short_minutes < long_minutes, got {short_minutes!r} and {long_minutes!r}",
            "e.g. short_minutes=3, long_minutes=10.",
        )
    if problem := _budget_problem(task, short_minutes):
        return problem
    card = _dataset_card(task)
    if isinstance(card, dict):
        return card

    parsed = [_parse_config(task, i, raw, card) for i, raw in enumerate(configs)]
    invalid = [p for p in parsed if isinstance(p, dict)]
    valid = [p for p in parsed if isinstance(p, TrainConfig)]
    if not isinstance(keep_top, int) or not 1 <= keep_top < len(valid):
        return error(
            f"keep_top must be at least 1 and fewer than the {len(valid)} valid configs, "
            f"got {keep_top!r}",
            "Halving needs more configs than survivors; with 1-2 configs use run_trials.",
            invalid=invalid,
        )

    cost = len(valid) * short_minutes + keep_top * long_minutes
    remaining = budget_status(task)["remaining_min"]
    if cost > remaining + EPSILON_MIN:
        hint = _affordability_hint(remaining, len(valid), short_minutes, long_minutes, keep_top)
        return error(
            f"this halving round could cost up to {cost:g} min ({len(valid)} x {short_minutes:g} "
            f"+ {keep_top} x {long_minutes:g}) but only {remaining:g} min remain",
            hint,
            invalid=invalid,
        )

    halving_id = f"h-{uuid.uuid4().hex[:6]}"
    rung1 = [_execute_trial(c, short_minutes, None, rung=1, halving_id=halving_id) for c in valid]
    survivors = sorted((e for e in rung1 if e["status"] == "ok"), key=lambda e: -e["val_score"])[
        :keep_top
    ]
    rung2, carried, skipped = [], [], []
    for parent in survivors:
        if parent["epochs"] >= parent["max_epochs"]:
            carried.append(parent)  # more time would reproduce the same result
            skipped.append(
                {"run_id": parent["run_id"], "reason": "finished all epochs in rung 1; kept as is"}
            )
            continue
        if problem := _budget_problem(task, long_minutes):
            skipped.append({"run_id": parent["run_id"], "reason": problem["error"]})
            continue
        rung2.append(_promote_in_halving(task, parent, long_minutes, halving_id))

    finalists = [e for e in rung2 if e["status"] == "ok"] + carried or survivors
    winner = max(finalists, key=lambda e: e["val_score"]) if finalists else None
    return {
        "task": task,
        "halving_id": halving_id,
        "estimated_cost_min": round(cost, 3),
        "trials": [_row(e, *HALVING_ROW_KEYS) for e in rung1 + rung2],
        "winner": _row(winner, *HALVING_ROW_KEYS) if winner else None,
        **screening_summary(rung1),
        "invalid": invalid,
        "skipped": skipped,
        **seed_check_status(task),
    }


def _promote_in_halving(task: str, parent: dict, minutes: float, halving_id: str) -> dict:
    config = TrainConfig.model_validate_json(
        storage().read_text(f"{run_key(task, parent['run_id'])}/config.json")
    )
    reason = (
        f"Halving rung 2 of {parent['run_id']} at {minutes:g} min. "
        f"Original reason: {_original_reason(config.reason)}"
    )
    child = config.model_copy(update={"reason": reason})
    return _execute_trial(child, minutes, parent["run_id"], rung=2, halving_id=halving_id)


# ---------------------------------------------------------------- seed confirmation


@tool
def confirm_seeds(
    run_id: str, n_seeds: int = SEED_CHECK_SEEDS, budget_minutes: float | None = None
) -> dict:
    """Re-train a run's exact config with new seeds to measure how much of its score is luck.

    The new runs join the original in a seed group. Returns every score, the mean and the
    std. Repeat calls on any member add more seeds to the same group.

    Args:
        run_id: A successful run (from leaderboard or run_halving).
        n_seeds: How many new seeds to train (1-5).
        budget_minutes: Minutes per seed run. Leave empty to reuse the original run's budget,
            which keeps the scores comparable.
    """
    base = find_entry(run_id)
    if base is None:
        return unknown_run(run_id)
    if base["status"] != "ok":
        return error(
            f"run {run_id} has status {base['status']!r}; only successful runs can be seed-checked",
            "Pick one of the top 'ok' runs from leaderboard().",
        )
    if not isinstance(n_seeds, int) or not 1 <= n_seeds <= MAX_SEEDS:
        return error(f"n_seeds must be 1-{MAX_SEEDS}, got {n_seeds!r}", "Use n_seeds=2.")
    task = base["task"]
    minutes = base["budget_minutes"] if budget_minutes is None else budget_minutes
    if abs(minutes - base["budget_minutes"]) > EPSILON_MIN:
        return error(
            f"budget_minutes={minutes:g} differs from the original run's "
            f"{base['budget_minutes']:g}",
            "Seed runs must use the same budget as the original, or their scores aren't "
            "comparable. Leave budget_minutes empty.",
        )
    if problem := _budget_problem(task, n_seeds * minutes):
        problem["hint"] = f"Each seed costs {minutes:g} min. Try n_seeds=1, or skip the check."
        return problem

    config = TrainConfig.model_validate_json(
        storage().read_text(f"{run_key(task, run_id)}/config.json")
    )
    card = _dataset_card(task)
    if isinstance(card, dict):
        return card
    if problem := check_combo(card, config.backbone, config.weights):
        return error(problem, "This config is no longer valid; it can't be seed-checked.")

    groups = load_seed_groups(task)
    group_id = group_of(run_id, groups) or f"sg-{run_id}"
    group = groups.setdefault(group_id, {"base_run_id": run_id, "run_ids": [run_id]})
    new_runs = []
    for seed in _unused_seeds(task, group["run_ids"], n_seeds):
        reason = (
            f"Seed check of {group['base_run_id']} (seed {seed}). "
            f"Original reason: {_original_reason(config.reason)}"
        )
        seeded = config.model_copy(update={"seed": seed, "reason": reason})
        new_runs.append(_execute_trial(seeded, minutes, None, seed_group=group_id))
        group["run_ids"].append(new_runs[-1]["run_id"])
        save_seed_groups(task, groups)  # after each run, so a crash keeps the finished ones

    stats = group_stats(group["run_ids"], load_entries(task))
    return {
        "task": task,
        "seed_group": group_id,
        "trials": [_row(e, "seed_group") for e in new_runs],
        "scores": stats["scores"],
        "mean": stats["mean"],
        "std": stats["std"],
        "n": stats["n"],
        "hint": "A difference between candidates only counts if it is larger than the "
        "bigger of their stds.",
    }


def _unused_seeds(task: str, run_ids: list[str], n: int) -> list[int]:
    """The n smallest seeds not yet used by any run in the group."""
    used = {
        TrainConfig.model_validate_json(storage().read_text(f"{run_key(task, r)}/config.json")).seed
        for r in run_ids
    }
    seeds, candidate = [], 0
    while len(seeds) < n:
        if candidate not in used:
            seeds.append(candidate)
        candidate += 1
    return seeds
