"""Run one agent session to the end: continuations and the registration safety net.

A Strands agent stops as soon as it replies with text only. In the tune phase that can
end a session early, with budget left and nothing registered. The driver:

1. sends up to MAX_CONTINUATIONS "Continue your plan" messages while the agent ended its
   turn, budget remains above one short trial, and it has not registered a model;
2. if the session still ends with nothing registered but a successful run exists,
   registers the best run by validation itself, and says so in the audit log and journal.
"""

from typing import Any

from gotrain.agent import Phase, ToolCallLimit
from gotrain.audit import AuditLog
from gotrain.engine.throughput import SCREEN_MINUTES
from gotrain.tools._common import group_stats, load_entries, load_seed_groups, mean_std_text
from gotrain.tools.registry import list_versions, register_model
from gotrain.tools.trials import budget_status

MAX_CONTINUATIONS = 3
AUTO_REGISTER_NOTE = "auto-registered by harness: agent did not register"


def short_trial_cost(task: str) -> float:
    """The cheapest per-trial budget the agent has used (a 2-min screen if none yet)."""
    budgets = [e["budget_minutes"] for e in load_entries(task)]
    return min(budgets) if budgets else SCREEN_MINUTES


def registered_since(task: str, before: set[str]) -> list[str]:
    return [v for v in list_versions(task) if v not in before]


def best_for_registration(task: str) -> tuple[dict, str] | None:
    """The best run by validation: a seed group counts by its mean (via its base run).

    Returns (leaderboard entry, why it was chosen), or None if no run succeeded.
    """
    entries = load_entries(task)
    by_id = {e["run_id"]: e for e in entries if e["status"] == "ok"}
    candidates: list[tuple[float, dict, str]] = []
    grouped: set[str] = set()
    for gid, group in load_seed_groups(task).items():
        stats = group_stats(group["run_ids"], entries)
        members = [by_id[r] for r in group["run_ids"] if r in by_id]
        grouped.update(group["run_ids"])
        if stats["mean"] is None or not members:
            continue
        base = by_id.get(group["base_run_id"], members[0])
        candidates.append((stats["mean"], base, f"seed group {gid}, {mean_std_text(stats)}"))
    for entry in by_id.values():
        if entry["run_id"] not in grouped:
            basis = f"single-seed val_score {entry['val_score']:.4f}, not seed-checked"
            candidates.append((entry["val_score"], entry, basis))
    if not candidates:
        return None
    _, entry, basis = max(candidates, key=lambda c: c[0])
    return entry, basis


def _should_continue(task: str, result: Any, limit: ToolCallLimit | None, before: set[str]) -> bool:
    if str(result.stop_reason) != "end_turn" or (limit is not None and limit.exhausted):
        return False
    if registered_since(task, before):
        return False
    return budget_status(task)["remaining_min"] > short_trial_cost(task)


def _continue(agent: Any, audit: AuditLog, task: str, n: int) -> Any:
    remaining = budget_status(task)["remaining_min"]
    message = (
        f"Continue your plan. Remaining budget: {remaining:g} min. "
        "Remember to finish with register_model."
    )
    audit.log("continuation", n=n, max=MAX_CONTINUATIONS, remaining_min=remaining, message=message)
    audit.journal(
        f"**HARNESS: continuation {n}/{MAX_CONTINUATIONS}.** The agent ended its turn with "
        f'{remaining:g} min left and no model registered. Sent: "{message}"'
    )
    return agent(message)


def auto_register(task: str, audit: AuditLog) -> str | None:
    """Safety net: register the best run if the agent didn't. Returns the version, if any."""
    choice = best_for_registration(task)
    if choice is None:
        audit.log("auto_register_skipped", reason="no successful run")
        audit.journal("**HARNESS:** the agent registered nothing, and no run succeeded.")
        return None
    entry, basis = choice
    notes = f"{AUTO_REGISTER_NOTE}. Chosen as the best run by validation ({basis})."
    out = register_model(entry["run_id"], notes)
    if "error" in out:
        audit.log("auto_register_failed", run_id=entry["run_id"], error=out["error"])
        audit.journal(f"**HARNESS:** could not auto-register `{entry['run_id']}`: {out['error']}")
        return None
    audit.log(
        "auto_registered", run_id=entry["run_id"], version=out["version"], basis=basis, notes=notes
    )
    audit.journal(
        f"**HARNESS: AUTO-REGISTERED {out['version']}** = `{entry['run_id']}` ({basis}). "
        "The agent did not call register_model; this registration was made by the harness, "
        "not chosen by the agent."
    )
    return out["version"]


def run_session(
    agent: Any,
    audit: AuditLog,
    task: str,
    phase: Phase,
    kickoff: str,
    limit: ToolCallLimit | None = None,
) -> dict:
    """Run the agent on `kickoff`, then continuations and the safety net (tune only)."""
    before = set(list_versions(task))
    result = agent(kickoff)
    continuations = 0
    while phase == "tune" and _should_continue(task, result, limit, before):
        if continuations == MAX_CONTINUATIONS:
            audit.log("continuations_exhausted", max=MAX_CONTINUATIONS)
            audit.journal(f"**HARNESS:** stopped after {MAX_CONTINUATIONS} continuations.")
            break
        continuations += 1
        result = _continue(agent, audit, task, continuations)

    new_versions = registered_since(task, before)
    registered_by = "agent" if new_versions else None
    version = new_versions[-1] if new_versions else None
    if phase == "tune" and not new_versions:
        version = auto_register(task, audit)
        registered_by = "harness" if version else None
    return {
        "stop_reason": str(result.stop_reason),
        "continuations": continuations,
        "registered_by": registered_by,
        "version": version,
        "usage": getattr(getattr(result, "metrics", None), "accumulated_usage", None),
    }
