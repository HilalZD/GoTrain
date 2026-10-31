"""Tool: score a registered model version on the TEST split, exactly once per task.

This is the ONLY module allowed to request the test split from engine.evaluate.
The first call writes registry/<task>/TEST_LOCK.json *before* evaluating; every
later call for that task is refused and gets the lock contents back. If the
evaluation crashes, the lock stays (status "failed") and only a human may remove it.

With the real engine, gotrain/vault.py adds machine-wide guards: only the official home
may run it, and every real test is recorded in the global test ledger, which refuses a
task already tested from any home.
"""

from gotrain.config import get_settings
from gotrain.contracts import TestRecord
from gotrain.engine import get_engine
from gotrain.tools._common import (
    error,
    git_commit,
    now_utc,
    read_json,
    storage,
    tool,
    write_json,
)
from gotrain.tools.registry import list_versions, load_version, version_checkpoints
from gotrain.vault import append_ledger, ledger_entries, real_test_problem


def lock_key(task: str) -> str:
    return f"registry/{task}/TEST_LOCK.json"


@tool
def final_test(task: str, version: str) -> dict:
    """One-time TEST score of registry/<task>/<version>. Refuses if the task was tested.

    Args:
        task: GEO-Bench task name.
        version: Registered version to score, e.g. "v2".
    """
    if storage().exists(lock_key(task)):
        return error(
            f"the test set for {task} has already been used; it can only be scored once",
            "Report the locked result below. Do not try to improve on it.",
            lock=read_json(lock_key(task)),
        )
    if problem := real_test_problem(task):
        return error(
            f"test vault: {problem}",
            "Report the existing result instead. Do not retry from another home.",
            ledger=ledger_entries(task),
        )
    mv = load_version(task, version)
    if mv is None:
        return error(
            f"no registered version {version!r} for {task}",
            f"Registered versions: {list_versions(task) or 'none - call register_model first'}.",
        )
    checkpoints = version_checkpoints(mv)
    if checkpoints is None:
        return error(f"cannot locate the checkpoints of {version}", "This is a registry bug.")
    # Exactly the model chosen on VALIDATION: one checkpoint or the ensemble, TTA as registered.
    checkpoint = checkpoints[0] if len(checkpoints) == 1 else checkpoints
    tta = {"tta": True} if mv.tta else {}  # only passed when set, so v1 engines keep working

    real = get_settings().engine == "real"
    lock = {"task": task, "version": version, "timestamp": now_utc().isoformat()}
    write_json(lock_key(task), {"status": "started", **lock})
    if real:
        append_ledger(task, "started", version=version)
    try:
        out = get_engine().evaluate(task, checkpoint, split="test", **tta)
    except Exception as exc:  # keep the lock: the test set may already have been read
        failed = {"status": "failed", **lock, "error": f"{type(exc).__name__}: {exc}"}
        write_json(lock_key(task), failed)
        if real:
            append_ledger(task, "failed", version=version, error=failed["error"])
        return error(
            f"test evaluation crashed: {failed['error']}",
            "Stop here. The test lock stays in place; a human must inspect and remove it.",
            lock=failed,
        )
    record = TestRecord(
        task=task,
        version=version,
        test_score=out["score"],
        metric_name=out["metric_name"],
        git_commit=git_commit(),
        timestamp=now_utc(),
    )
    write_json(lock_key(task), {"status": "done", **record.model_dump(mode="json")})
    if real:
        append_ledger(task, "done", **record.model_dump(mode="json", exclude={"task"}))
    model = {"members": mv.members, "tta": mv.tta}
    return {**record.model_dump(mode="json"), **model, "val_score": mv.val_score, "locked": True}
