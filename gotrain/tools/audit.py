"""Tool: the data auditor (gotrain/engine/audit.py, Serena's) as an agent tool.

    audit_dataset(task, apply_fixes=True) -> findings dict

The result is cached per task (registry/<task>/data_audit.json), so later sessions and
repeat calls are free. It runs on CPU and is never charged to the GPU budget.
"""

import importlib
import importlib.util

from gotrain.tools._common import error, now_utc, read_json, storage, tool, write_json

AUDIT_MODULE = "gotrain.engine.audit"


def cache_key(task: str) -> str:
    return f"registry/{task}/data_audit.json"


def audit_available() -> bool:
    return importlib.util.find_spec(AUDIT_MODULE) is not None


@tool
def audit_dataset(task: str, apply_fixes: bool = True) -> dict:
    """Audit the task's data (labels, class names, flat tiles, duplicates, ...) before training.

    Read every finding; they change what you should train and how you judge scores. Cached per
    task and free: it is not charged to the GPU budget.

    Args:
        task: GEO-Bench task name.
        apply_fixes: Apply the auditor's safe corrections (e.g. corrected class names).
    """
    if storage().exists(cache_key(task)):
        return {**read_json(cache_key(task)), "cached": True}
    if not audit_available():
        return error(
            f"the data auditor ({AUDIT_MODULE}) is not available in this checkout",
            "Continue with inspect_dataset only and say in your plan that no audit was run.",
        )
    findings = importlib.import_module(AUDIT_MODULE).audit_dataset(task, apply_fixes=apply_fixes)
    out = {
        "task": task,
        "apply_fixes": apply_fixes,
        "audited_at": now_utc().isoformat(),
        "findings": findings,
    }
    write_json(cache_key(task), out)
    return {**out, "cached": False}
