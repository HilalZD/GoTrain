"""Tool: score a registered model version on the TEST split, exactly once per task.

This is the ONLY module allowed to request the test split from engine.evaluate.
The first call writes registry/<task>/TEST_LOCK.json *before* evaluating; every
later call for that task is refused and gets the lock contents back. If the
evaluation crashes, the lock stays (status "failed") and only a human may remove it.
"""

from gotrain.contracts import TestRecord
from gotrain.engine import get_engine
from gotrain.tools._common import (
    error,
    find_entry,
    git_commit,
    now_utc,
    read_json,
    storage,
    tool,
    write_json,
)
from gotrain.tools.registry import list_versions, load_version, version_key


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
    mv = load_version(task, version)
    if mv is None:
        return error(
            f"no registered version {version!r} for {task}",
            f"Registered versions: {list_versions(task) or 'none - call register_model first'}.",
        )
    entry = find_entry(mv.run_id)
    if entry is None or not entry.get("checkpoint"):
        return error(f"cannot locate the checkpoint of {version}", "This is a registry bug.")
    checkpoint = storage().local_path(version_key(task, version)) / entry["checkpoint"]

    lock = {"task": task, "version": version, "timestamp": now_utc().isoformat()}
    write_json(lock_key(task), {"status": "started", **lock})
    try:
        out = get_engine().evaluate(task, checkpoint, split="test")
    except Exception as exc:  # keep the lock: the test set may already have been read
        failed = {"status": "failed", **lock, "error": f"{type(exc).__name__}: {exc}"}
        write_json(lock_key(task), failed)
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
    return {**record.model_dump(mode="json"), "val_score": mv.val_score, "locked": True}
