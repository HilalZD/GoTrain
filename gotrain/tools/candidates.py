"""Tool: compare ensembles and test-time augmentation (TTA) on VALIDATION with leftover budget.

A candidate is 1-4 successful runs of one task whose probabilities are averaged, with or
without TTA (8 flip/rot90 views). Each is scored on the full validation split by
engine.evaluate; the test split is never requested here. Every result is stored in
registry/<task>/candidates.jsonl, so register_model can only register an ensemble or a TTA
model whose validation score was actually measured.
"""

import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gotrain.config import get_settings
from gotrain.contracts import DatasetCard, TrainConfig, format_errors
from gotrain.engine import get_engine
from gotrain.engine.throughput import active_timing
from gotrain.tools._common import (
    error,
    find_entry,
    group_stats,
    load_entries,
    load_seed_groups,
    now_utc,
    run_key,
    storage,
    tool,
)
from gotrain.tools.trials import _budget_problem, _charge, _dataset_card, best_run

MAX_CANDIDATES = 6
MAX_MEMBERS = 4
MAX_CALLS_PER_SESSION = 2  # enforced by the agent harness (gotrain/agent.py, SelectionQuota)
TTA_VIEWS = 8

SELECTION_RULES = (
    "Chosen on VALIDATION only. If candidates tie within the measured seed noise, choose the "
    "SIMPLER one: fewer members first, then no TTA. At most "
    f"{MAX_CALLS_PER_SESSION} evaluate_candidates calls per session (more comparisons on the "
    "same validation set would overfit it)."
)
NO_NOISE_NOTE = (
    "No seed check yet, so the seed noise is unknown and only exact ties count as ties. "
    "confirm_seeds measures it."
)


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_ids: list[str] = Field(min_length=1, max_length=MAX_MEMBERS)
    tta: bool = False

    @property
    def key(self) -> tuple[tuple[str, ...], bool]:
        return tuple(sorted(self.run_ids)), self.tta


def candidates_key(task: str) -> str:
    return f"registry/{task}/candidates.jsonl"


def load_candidate_results(task: str) -> list[dict]:
    key = candidates_key(task)
    if not storage().exists(key):
        return []
    return [json.loads(line) for line in storage().read_text(key).splitlines() if line.strip()]


def find_candidate_result(task: str, run_ids: list[str], tta: bool) -> dict | None:
    """The latest successful validation score of exactly these members with this TTA flag."""
    wanted = sorted(run_ids)
    for record in reversed(load_candidate_results(task)):
        if sorted(record["run_ids"]) == wanted and record["tta"] == tta and record["ok"]:
            return record
    return None


def member_problem(task: str, run_id: str) -> str | None:
    """Why this run can't be an ensemble member of `task` (None if it can)."""
    entry = find_entry(run_id)
    if entry is None:
        return f"unknown run {run_id!r}"
    if entry["task"] != task:
        return f"run {run_id} belongs to {entry['task']}, not {task}"
    if entry["status"] != "ok" or not entry.get("checkpoint"):
        return f"run {run_id} has status {entry['status']!r}; only 'ok' runs can be members"
    return None


def member_checkpoint(task: str, run_id: str) -> Path:
    return storage().local_path(run_key(task, run_id)) / find_entry(run_id)["checkpoint"]


def member_img_size(task: str, run_id: str) -> int:
    raw = storage().read_text(f"{run_key(task, run_id)}/config.json")
    return TrainConfig.model_validate_json(raw).img_size


def estimate_minutes(task: str, card: DatasetCard, candidate: Candidate) -> float:
    """Inference on the full val split: every member, every view, at the member's img_size.
    Priced at training speed per image (the timing model has no inference-only rate), so the
    estimate is on the safe side."""
    timing = active_timing()
    seg = card.task_type == "segmentation"
    views = TTA_VIEWS if candidate.tta else 1
    per_view = sum(
        timing.minutes_per_image(side, side, seg)
        for side in (member_img_size(task, r) for r in candidate.run_ids)
    )
    return card.split_sizes.get("val", 0) * views * per_view


def seed_noise(task: str) -> float | None:
    """The largest seed-to-seed std measured by confirm_seeds for this task (None: none yet)."""
    entries = load_entries(task)
    stds = [group_stats(g["run_ids"], entries)["std"] for g in load_seed_groups(task).values()]
    stds = [s for s in stds if s is not None]
    return max(stds) if stds else None


def simplicity(row: dict) -> tuple[int, bool]:
    return row["n_members"], row["tta"]


def parse_candidates(task: str, candidates: Any) -> list[Candidate] | dict:
    """Validated candidates, or an error dict (nothing is run if any candidate is invalid)."""
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= MAX_CANDIDATES:
        return error(
            f"candidates must be a list of 1-{MAX_CANDIDATES} objects, got {candidates!r}",
            'Pass e.g. [{"run_ids": ["<best run>"], "tta": true}].',
        )
    parsed, seen = [], set()
    for i, raw in enumerate(candidates):
        try:
            cand = Candidate.model_validate(raw)
        except ValidationError as exc:
            return error(
                f"candidate {i}: {format_errors(exc)}",
                f'Each candidate is {{"run_ids": [1-{MAX_MEMBERS} run ids], "tta": bool}}.',
            )
        if len(set(cand.run_ids)) != len(cand.run_ids):
            return error(f"candidate {i} lists a run twice", "List each member once.")
        problems = [p for r in cand.run_ids if (p := member_problem(task, r))]
        if problems:
            return error(
                f"candidate {i}: {'; '.join(problems)}",
                "Members must be 'ok' runs of this task, from leaderboard().",
            )
        if cand.key in seen:
            return error(f"candidate {i} repeats an earlier candidate", "Remove the duplicate.")
        seen.add(cand.key)
        parsed.append(cand)
    return parsed


def _evaluate(task: str, candidate: Candidate, estimate: float) -> dict:
    """Score one candidate on val, charge its time, record the result."""
    checkpoints = [member_checkpoint(task, r) for r in candidate.run_ids]
    started = time.monotonic()
    try:
        out = get_engine().evaluate(task, checkpoints, split="val", tta=candidate.tta)
        result = {
            "ok": True,
            "val_score": round(out["score"], 5),
            "metric_name": out["metric_name"],
        }
    except Exception as exc:  # one failed candidate doesn't stop the others
        result = {"ok": False, "val_score": None, "error": f"{type(exc).__name__}: {exc}"}
    elapsed = (time.monotonic() - started) / 60
    # The fake engine scores instantly, so it is charged its simulated inference time, as fake
    # trials are charged simulated training time; real runs are charged the measured time.
    minutes = elapsed if get_settings().engine == "real" else estimate
    _charge(task, minutes)
    record = {
        "task": task,
        "run_ids": candidate.run_ids,
        "tta": candidate.tta,
        **result,
        "minutes": round(minutes, 4),
        "created_at": now_utc().isoformat(),
    }
    storage().append_text(candidates_key(task), json.dumps(record) + "\n")
    return record


@tool
def evaluate_candidates(task: str, candidates: list[dict]) -> dict:
    """Score ensembles and/or test-time augmentation of finished runs on VALIDATION, ranked.

    Use it only once the best score is in the target zone or exploration is done, with
    budget left over. Compare the single best run with and without TTA, and the top-2 and
    top-3 ensembles with and without TTA. If candidates tie within the measured seed noise,
    choose the SIMPLER one: fewer members first, then no TTA. At most 2 calls per session.
    Inference time is charged to the budget. The single best run (no TTA) is always scored
    too, as the baseline for `delta_vs_single_best`.

    Args:
        task: GEO-Bench task name.
        candidates: 1-6 objects {"run_ids": [1-4 run ids], "tta": true or false}. Members
            must be 'ok' runs of this task; their probabilities are averaged.
    """
    parsed = parse_candidates(task, candidates)
    if isinstance(parsed, dict):
        return parsed
    card = _dataset_card(task)
    if isinstance(card, dict):
        return card
    single = Candidate(run_ids=[best_run(task)["run_id"]], tta=False)
    if single.key not in {c.key for c in parsed}:
        parsed.insert(0, single)
    estimates = [estimate_minutes(task, card, c) for c in parsed]
    if problem := _budget_problem(task, sum(estimates)):
        problem["error"] = f"evaluating these candidates is estimated at {sum(estimates):.2f} min"
        problem["hint"] = "Drop some candidates (TTA ones cost 8x), or register the best run."
        return problem

    records = [_evaluate(task, c, est) for c, est in zip(parsed, estimates, strict=True)]
    baseline = next(r for r, c in zip(records, parsed, strict=True) if c.key == single.key)
    noise = seed_noise(task)
    rows = [
        {
            "run_ids": r["run_ids"],
            "n_members": len(r["run_ids"]),
            "tta": r["tta"],
            "val_score": r["val_score"],
            "delta_vs_single_best": (
                round(r["val_score"] - baseline["val_score"], 5)
                if r["ok"] and baseline["ok"]
                else None
            ),
            **({"error": r["error"]} if not r["ok"] else {}),
        }
        for r in records
    ]
    scored = [r for r in rows if r["val_score"] is not None]
    scored.sort(key=lambda r: (-r["val_score"], simplicity(r)))
    failed = [r for r in rows if r["val_score"] is None]
    out: dict[str, Any] = {
        "task": task,
        "single_best_run": single.run_ids[0],
        "seed_noise_std": noise,
        "minutes_charged": round(sum(r["minutes"] for r in records), 4),
    }
    if scored:
        top = scored[0]["val_score"]
        for row in scored:
            row["ties_with_top"] = top - row["val_score"] <= (noise or 0.0) + 1e-12
        pick = min((r for r in scored if r["ties_with_top"]), key=simplicity)
        out["recommended"] = {
            "run_ids": pick["run_ids"],
            "tta": pick["tta"],
            "val_score": pick["val_score"],
            "why": "the simplest candidate tied with the top score within the seed noise",
        }
        out["next_step"] = (
            "Register your choice: register_model(run_ids=[...], tta=...) with the exact members "
            "and TTA flag of a candidate scored here."
        )
    out["ranked"] = [{"rank": i, **r} for i, r in enumerate(scored, start=1)] + failed
    out["rules"] = SELECTION_RULES
    if noise is None:
        out["note"] = NO_NOISE_NOTE
    return out
