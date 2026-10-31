"""Run one agent session to the end: continuations and the registration safety net.

A Strands agent stops as soon as it replies with text only. In the tune phase that can
end a session early, with budget left and nothing registered. The driver:

1. sends up to MAX_CONTINUATIONS "Continue your plan" messages while the agent ended its
   turn, budget remains above one short trial, and it has not registered a model;
2. at the end of a tune session, asks the agent for 2-5 lessons (validation results only) and
   writes them with the tag "source:tune"; if that fails, writes one lesson from the leaderboard;
3. if the session still ends with nothing registered but a successful run exists,
   registers the best run by validation itself, and says so in the audit log and journal.
"""

import json
import re
from typing import Any

from gotrain.agent import Phase, ToolCallLimit
from gotrain.audit import AuditLog
from gotrain.config import get_settings
from gotrain.engine import get_engine
from gotrain.engine.throughput import SCREEN_MINUTES
from gotrain.gopilot import export_model
from gotrain.tools._common import (
    group_stats,
    load_entries,
    load_seed_groups,
    mean_std_text,
    read_json,
)
from gotrain.tools.final_test import lock_key
from gotrain.tools.memory import write_lesson
from gotrain.tools.registry import list_versions, register_model
from gotrain.tools.trials import budget_status

MAX_CONTINUATIONS = 3
TUNE_TAG = "source:tune"
MIN_LESSONS, MAX_LESSONS = 2, 5
LESSON_REQUEST = (
    "The session is over. Reply with ONLY a JSON list of 2-5 lessons for future tasks, e.g. "
    '[{"lesson": "one concrete finding with its validation scores", "tags": ["sentinel-2"]}]. '
    "Base every lesson on VALIDATION results from this session only; never mention the test set. "
    "Do not call any tool."
)
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


def lesson_tags(task: str) -> list[str]:
    """Sensor and task type from the dataset card, so lessons match across tasks."""
    try:
        card = get_engine().describe_dataset(task)
        return [str(card.sensor), str(card.task_type)]
    except Exception:
        return [task]


def parse_lessons(text: str) -> list[dict]:
    """The JSON list of {"lesson", "tags"} in the agent's reply (tolerates prose around it)."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    items = json.loads(match.group(0)) if match else []
    good = [i for i in items if isinstance(i, dict) and str(i.get("lesson", "")).strip()]
    return good[:MAX_LESSONS]


def automatic_lesson(task: str) -> str | None:
    choice = best_for_registration(task)
    if choice is None:
        return None
    entry, basis = choice
    return (
        f"Best validation result on {task} (auto-written by harness): {entry.get('config', '?')}"
        f" with val {entry['metric_name']} {entry['val_score']:.4f}; {basis}."
    )


def write_session_lessons(agent: Any, audit: AuditLog, task: str) -> int:
    """Lessons from a tune session, tagged source:tune. Returns how many were written."""
    base_tags = [TUNE_TAG, *lesson_tags(task)]
    lessons: list[dict] = []
    try:
        lessons = parse_lessons(str(agent(LESSON_REQUEST)))
    except Exception as exc:
        audit.log("lessons_agent_failed", error=f"{type(exc).__name__}: {exc}")
    if len(lessons) < MIN_LESSONS:
        auto = automatic_lesson(task)
        lessons = [{"lesson": auto, "tags": []}] if auto else []
        source = "automatic"
    else:
        source = "agent"
    written = 0
    for item in lessons:
        tags = [*base_tags, *[t for t in item.get("tags") or [] if isinstance(t, str)]]
        out = write_lesson(task, str(item["lesson"]), tags)
        written += "error" not in out
    audit.log("session_lessons", source=source, written=written)
    audit.journal(f"**HARNESS:** wrote {written} {source} lesson(s) tagged `{TUNE_TAG}`.")
    return written


def export_for_gopilot(audit: AuditLog, task: str, version: str | None, when: str) -> None:
    """Export the bundle (best effort: a failed export never fails the session)."""
    if version is None:
        return
    out = export_model(task, version)
    if "error" in out:
        audit.log("gopilot_export_failed", version=version, when=when, error=out["error"])
        audit.journal(f"**HARNESS:** GoPilot export of {version} failed: {out['error']}")
        return
    audit.log("gopilot_export", version=version, when=when, tool_name=out["tool_name"])
    audit.journal(
        f"**HARNESS: GoPilot export ({when}).** {task} {version} is now the MCP tool "
        f"`{out['tool_name']}` (bundle `{out['bundle']}`, hash {out['version_hash']})."
    )


def finalised_version(task: str) -> str | None:
    """The version final_test scored, if the test ran to the end."""
    try:
        lock = read_json(lock_key(task))
    except (OSError, ValueError):
        return None
    return lock["version"] if lock.get("status") == "done" else None


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
    lessons = write_session_lessons(agent, audit, task) if phase == "tune" else 0
    if phase == "tune":
        export_for_gopilot(audit, task, version, "after tune registration")
    elif not get_settings().rehearsal:
        export_for_gopilot(audit, task, finalised_version(task), "after finalise, with test score")
    return {
        "stop_reason": str(result.stop_reason),
        "continuations": continuations,
        "registered_by": registered_by,
        "version": version,
        "lessons_written": lessons,
        "usage": getattr(getattr(result, "metrics", None), "accumulated_usage", None),
    }
