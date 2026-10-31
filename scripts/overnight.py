"""Run a queue of tune sessions from a YAML plan, one after another. TUNE PHASE ONLY.

    python scripts/overnight.py configs/overnight_a.yaml

    queue:
      - {task: m-eurosat, budget_min: 45, home: ~/gotrain_home, memory: shared}
      - {task: m-eurosat, budget_min: 5, home: ~/gotrain_home, label_fracs: [0.05, 0.25]}

Never finalises: the test set is not touched here. Per entry: a log file in <home>/logs/queue/
and a row in <home>/queue_status_<plan>.json. A crash is logged and the queue goes on. If
~/GOTRAIN_STOP exists (override: GOTRAIN_STOP_FILE) the queue stops after the current entry.
A label_fracs entry skips the tune if a model is already registered in its home.
memory "empty" gives the entry an isolated lessons file (memory ablation). label_fracs
retrains the registered config on fractions of TRAIN and writes VALIDATION scores to
<home>/label_efficiency/<task>.json.
"""

import argparse
import contextlib
import json
import os
import sys
import traceback
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gotrain.cli import run as cli_run  # noqa: E402
from gotrain.config import get_settings  # noqa: E402
from gotrain.engine import get_engine  # noqa: E402
from gotrain.storage import get_storage  # noqa: E402
from gotrain.tools._common import summarize_config  # noqa: E402
from gotrain.tools.memory import ISOLATED_LESSONS_KEY  # noqa: E402
from gotrain.tools.registry import list_versions, load_version  # noqa: E402

ENV_KEYS = ("GOTRAIN_HOME", "GOTRAIN_MEMORY")


def stop_file() -> Path:
    return Path(os.environ.get("GOTRAIN_STOP_FILE", "~/GOTRAIN_STOP")).expanduser()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_plan(path: Path) -> list[dict]:
    queue = yaml.safe_load(path.read_text(encoding="utf-8"))["queue"]
    for entry in queue:
        missing = {"task", "budget_min", "home"} - set(entry)
        if missing:
            raise ValueError(f"queue entry {entry} is missing {sorted(missing)}")
        if entry.get("memory", "shared") not in ("shared", "empty"):
            raise ValueError(f"memory must be 'shared' or 'empty': {entry}")
    return queue


@contextlib.contextmanager
def entry_env(entry: dict) -> Iterator[Path]:
    """GOTRAIN_HOME / GOTRAIN_MEMORY for one entry, restored afterwards."""
    saved = {k: os.environ.get(k) for k in ENV_KEYS}
    home = Path(entry["home"]).expanduser()
    os.environ["GOTRAIN_HOME"] = str(home)
    os.environ["GOTRAIN_MEMORY"] = entry.get("memory", "shared")
    try:
        yield home
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def write_status(home: Path, plan: str, index: int, row: dict) -> None:
    path = home / f"queue_status_{plan}.json"
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    while len(rows) <= index:
        rows.append({})
    rows[index] = row
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def registered_version(task: str) -> tuple[str | None, float | None]:
    versions = list_versions(task)
    if not versions:
        return None, None
    mv = load_version(task, versions[-1])
    return versions[-1], mv.val_score


def label_efficiency(task: str, fracs: list[float], home: Path) -> dict:
    """Retrain the registered (single-run) config at each fraction; VALIDATION scores only."""
    version, _ = registered_version(task)
    if version is None:
        return {"error": "nothing registered"}
    mv = load_version(task, version)
    minutes = max(2.0, _run_minutes(task, mv.run_id))
    rows = []
    for frac in fracs:
        config = mv.config.model_copy(update={"train_subset_frac": frac})
        run_dir = get_storage().local_path(f"runs/{task}/labelfrac-{frac:g}-{mv.run_id[-4:]}")
        result = get_engine().train(config, minutes, run_dir)
        rows.append({"frac": frac, "status": result.status, "val_score": result.val_score})
    out = {
        "task": task,
        "version": version,
        "split": "val",
        "config": summarize_config(mv.config),
        "rows": rows,
    }
    path = home / "label_efficiency" / f"{task}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def _run_minutes(task: str, run_id: str) -> float:
    try:
        result = json.loads(get_storage().read_text(f"runs/{task}/{run_id}/result.json"))
        return float(result.get("minutes_used") or 2.0)
    except (OSError, ValueError):
        return 2.0


def run_entry(entry: dict, index: int, plan: str = "plan") -> dict:
    """One queue entry: tune, then optional label curve. Never raises."""
    task = entry["task"]
    row = {
        "task": task,
        "start": utc_now(),
        "end": None,
        "status": "running",
        "best_val": None,
        "registered_version": None,
    }
    with entry_env(entry) as home:
        write_status(home, plan, index, row)
        log_path = home / "logs" / "queue" / plan / f"{index:02d}-{task}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with (
            log_path.open("a", encoding="utf-8") as log,
            contextlib.redirect_stdout(log),
            contextlib.redirect_stderr(log),
        ):
            try:
                if entry.get("memory") == "empty":
                    get_storage().write_text(ISOLATED_LESSONS_KEY, "")
                get_settings()  # fail early on bad settings
                if entry.get("label_fracs") and registered_version(task)[0]:
                    code = 0  # curve entry: use the model an earlier entry registered
                else:
                    args = argparse.Namespace(
                        task=task,
                        phase="tune",
                        budget_min=float(entry["budget_min"]),
                        max_tool_calls=None,
                    )
                    code = cli_run(args)
                row["status"] = "ok" if code == 0 else f"failed (exit {code})"
                row["registered_version"], row["best_val"] = registered_version(task)
                if code == 0 and entry.get("label_fracs"):
                    label_efficiency(task, list(entry["label_fracs"]), home)
            except BaseException as exc:  # noqa: BLE001 - the queue must go on
                traceback.print_exc()
                row["status"] = f"crashed: {type(exc).__name__}: {exc}"
                if isinstance(exc, KeyboardInterrupt):
                    row["end"] = utc_now()
                    write_status(home, plan, index, row)
                    raise
        row["end"] = utc_now()
        row["log"] = str(log_path)
        write_status(home, plan, index, row)
    return row


def run_queue(queue: list[dict], plan: str = "plan") -> list[dict]:
    rows = []
    for index, entry in enumerate(queue):
        if stop_file().exists():
            print(f"{stop_file()} exists: stopping before entry {index} ({entry['task']})")
            break
        rows.append(run_entry(entry, index, plan))
        print(f"[{index}] {entry['task']}: {rows[-1]['status']}", flush=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("plan", type=Path)
    plan = parser.parse_args(argv).plan
    run_queue(load_plan(plan), plan.stem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
