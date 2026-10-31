"""The test vault: with the real engine, a task's test set is scored at most once per machine.

The per-home TEST_LOCK.json alone had a loophole: a second home has no lock, so a real
final test could run once per home. Two machine-level guards close it:

- official home: with GOTRAIN_ENGINE=real, the finalise phase and final_test run only when
  GOTRAIN_HOME is GOTRAIN_OFFICIAL_HOME (default ~/gotrain_home);
- ledger: every real test is appended (start, then outcome) to GOTRAIN_TEST_LEDGER
  (default ~/.gotrain_test_ledger.jsonl), outside every home. A task already in the ledger
  is refused, whichever home asks and whatever its status ("started" may have read data).

The ledger is a machine-wide file, so it is deliberately not behind get_storage().
Fake-engine tests stay per-home: they never read or write the ledger.
"""

import json
from typing import Any

from gotrain.config import get_settings
from gotrain.tools._common import now_utc


def ledger_entries(task: str | None = None) -> list[dict]:
    """Ledger records, oldest first; only `task`'s if given."""
    path = get_settings().test_ledger
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [r for r in records if task is None or r.get("task") == task]


def append_ledger(task: str, status: str, **fields: Any) -> dict:
    """Append one record (append-only: earlier lines are never rewritten)."""
    settings = get_settings()
    record = {
        "ts": now_utc().isoformat(),
        "task": task,
        "status": status,
        "home": str(settings.home),
        **fields,
    }
    settings.test_ledger.parent.mkdir(parents=True, exist_ok=True)
    with settings.test_ledger.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return record


def real_test_problem(task: str) -> str | None:
    """Why a real test evaluation of `task` must not run here, or None (also None if fake)."""
    settings = get_settings()
    if settings.engine != "real":
        return None
    if not settings.is_official_home:
        return (
            f"real test evaluations run only in the official home {settings.official_home} "
            f"(GOTRAIN_OFFICIAL_HOME), but GOTRAIN_HOME is {settings.home}"
        )
    if previous := ledger_entries(task):
        last = previous[-1]
        return (
            f"the machine-wide test ledger {settings.test_ledger} already has {task} "
            f"(status {last['status']!r}, home {last['home']}, {last['ts']}); its test set "
            "has been used"
        )
    return None
