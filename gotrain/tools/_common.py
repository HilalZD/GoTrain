"""Helpers shared by the tools: the @tool wrapper, error shape, run ids, leaderboard I/O.

Storage layout (keys under GOTRAIN_HOME):
    runs/<task>/<run_id>/            config.json, result.json, git_commit, checkpoints/
    registry/<task>/leaderboard.jsonl
    registry/<task>/budget.json
    registry/<task>/v1/, v2/ ...
    registry/<task>/TEST_LOCK.json
    memory/lessons.md
"""

import functools
import hashlib
import inspect
import json
import re
import subprocess
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from gotrain.config import REPO_ROOT
from gotrain.contracts import TrainConfig
from gotrain.storage import Storage, get_storage

RUN_ID_PATTERN = re.compile(r"^(?P<task>.+)-\d{8}-\d{6}-[0-9a-f]{4}$")
REASON_CHARS = 120


def error(message: str, hint: str, **extra: Any) -> dict:
    """The one error shape every tool returns instead of raising."""
    return {"error": message, "hint": hint, **extra}


def tool(fn: Callable[..., dict]) -> Callable[..., dict]:
    """Make `fn` agent-safe: never raises, and every output carries the task budget."""
    signature = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> dict:
        try:
            out = fn(*args, **kwargs)
        except Exception as exc:  # tools must never raise to the agent
            out = error(
                f"{fn.__name__} failed unexpectedly: {type(exc).__name__}: {exc}",
                "Check the arguments against the tool description. If they look right, "
                "this is a tool bug: try a different action and leave it for a human.",
            )
        out["budget"] = _budget_for_call(signature, args, kwargs)
        return out

    return wrapper


def _budget_for_call(signature: inspect.Signature, args: tuple, kwargs: dict) -> dict | None:
    from gotrain.tools.trials import budget_status  # late import: trials uses this module

    try:
        bound = signature.bind_partial(*args, **kwargs).arguments
    except TypeError:
        return None
    task = bound.get("task") or task_from_run_id(bound.get("run_id"))
    return budget_status(task) if isinstance(task, str) else None


def storage() -> Storage:
    return get_storage()


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def task_from_run_id(run_id: Any) -> str | None:
    match = RUN_ID_PATTERN.match(run_id) if isinstance(run_id, str) else None
    return match.group("task") if match else None


def run_key(task: str, run_id: str) -> str:
    return f"runs/{task}/{run_id}"


def leaderboard_key(task: str) -> str:
    return f"registry/{task}/leaderboard.jsonl"


def new_run_id(config: TrainConfig) -> str:
    """'<task>-<YYYYmmdd-HHMMSS>-<4 hex>', unique on this storage."""
    store = storage()
    while True:
        salt = f"{config.model_dump_json()}{uuid.uuid4()}"
        short = hashlib.sha1(salt.encode()).hexdigest()[:4]
        run_id = f"{config.task}-{now_utc():%Y%m%d-%H%M%S}-{short}"
        if not store.exists(run_key(config.task, run_id)):
            return run_id


def git_commit() -> str:
    """Current commit hash, with '-dirty' if tracked files have uncommitted changes."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"{commit}-dirty" if dirty else commit


def read_json(key: str) -> dict:
    return json.loads(storage().read_text(key))


def write_json(key: str, data: dict) -> None:
    storage().write_text(key, json.dumps(data, indent=2, default=str))


def load_entries(task: str) -> list[dict]:
    key = leaderboard_key(task)
    if not storage().exists(key):
        return []
    return [json.loads(line) for line in storage().read_text(key).splitlines() if line.strip()]


def append_entry(task: str, entry: dict) -> None:
    storage().append_text(leaderboard_key(task), json.dumps(entry, default=str) + "\n")


def find_entry(run_id: str) -> dict | None:
    task = task_from_run_id(run_id)
    if task is None:
        return None
    matches = [e for e in load_entries(task) if e["run_id"] == run_id]
    return matches[-1] if matches else None


def unknown_run(run_id: Any) -> dict:
    return error(
        f"no run with run_id {run_id!r}",
        "Copy a run_id exactly from run_trials or leaderboard output "
        "(format '<task>-<YYYYmmdd-HHMMSS>-<4 hex>').",
    )


def rank(entries: list[dict]) -> list[dict]:
    """Successful runs by val_score (best first), then the rest; ties go to cheaper runs."""
    return sorted(
        entries,
        key=lambda e: (e["status"] != "ok", -(e.get("val_score") or 0.0), e["minutes"]),
    )


def short_reason(reason: str) -> str:
    return reason if len(reason) <= REASON_CHARS else reason[: REASON_CHARS - 3] + "..."


def summarize_config(config: TrainConfig) -> str:
    """One line an LLM can scan, e.g. 'resnet50/ssl4eo_s2 lr=0.0003 bs=64 ...'."""
    bands = "all" if config.bands == "all" else f"{len(config.bands)}bands"
    parts = [
        f"{config.backbone}/{config.weights}",
        f"lr={config.lr:g}",
        f"wd={config.weight_decay:g}",
        f"bs={config.batch_size}",
        f"img={config.img_size}",
        f"ep={config.max_epochs}",
        f"loss={config.loss}",
        f"aug={'+'.join(config.augment) or 'none'}",
        f"bands={bands}",
    ]
    if config.train_subset_frac < 1.0:
        parts.append(f"frac={config.train_subset_frac:g}")
    if config.seed:
        parts.append(f"seed={config.seed}")
    return " ".join(parts)
