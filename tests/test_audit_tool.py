"""audit_dataset tool: cached, free, and a gate before the first trial."""

import json
import sys
import types

import pytest
from conftest import make_config
from test_agent import ScriptedModel

from gotrain.agent import build_agent
from gotrain.audit import AuditLog
from gotrain.tools import audit_dataset, init_task_budget
from gotrain.tools.trials import budget_status

TASK = "m-eurosat"
TRIAL = {"task": TASK, "configs": [make_config()], "budget_minutes_per_trial": 2.0}


@pytest.fixture
def stub_audit(monkeypatch):
    calls = []
    module = types.ModuleType("gotrain.engine.audit")

    def audit_dataset_stub(task, apply_fixes=True):
        calls.append((task, apply_fixes))
        return {"flat_tiles": 3, "class_names_corrected": True}

    module.audit_dataset = audit_dataset_stub
    monkeypatch.setitem(sys.modules, "gotrain.engine.audit", module)
    monkeypatch.setattr("gotrain.tools.audit.audit_available", lambda: True)
    return calls


def test_audit_is_cached_per_task_and_free(stub_audit):
    init_task_budget(TASK, 30)
    first = audit_dataset(TASK)
    second = audit_dataset(TASK)
    assert first["cached"] is False and second["cached"] is True
    assert (
        first["findings"]
        == second["findings"]
        == {
            "flat_tiles": 3,
            "class_names_corrected": True,
        }
    )
    assert stub_audit == [(TASK, True)]  # the auditor ran once, with fixes applied
    assert budget_status(TASK)["used_min"] == 0


def test_missing_auditor_is_a_clear_error_not_a_crash(monkeypatch):
    monkeypatch.setattr("gotrain.tools.audit.audit_available", lambda: False)
    out = audit_dataset(TASK)
    assert "not available" in out["error"] and "hint" in out


def run(script):
    init_task_budget(TASK, 30)
    audit = AuditLog(TASK, "tune", session_id="s")
    agent = build_agent(TASK, "tune", audit=audit, model=ScriptedModel(script), require_audit=True)
    agent("go")
    lines = audit._storage.read_text(audit.jsonl_key).splitlines()
    return [json.loads(line) for line in lines]


def test_gate_blocks_trials_until_the_audit_was_read(stub_audit):
    events = run(
        [
            [("tool", "run_trials", TRIAL)],
            [("tool", "audit_dataset", {"task": TASK})],
            [("tool", "run_trials", TRIAL)],
            [("text", "Done.")],
        ]
    )
    calls = [(e["tool"], e["cancelled"]) for e in events if e["event"] == "tool_call"]
    assert calls == [("run_trials", True), ("audit_dataset", False), ("run_trials", False)]
    assert any(e["event"] == "audit_gate_enforced" for e in events)
    assert any(e["event"] == "audit_read" for e in events)
