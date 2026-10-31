"""finalise --rehearsal: the whole finalise path on validation, with no lock and no ledger."""

import json

import pytest
from test_agent import ScriptedModel
from test_tools_models import TASK, budget, one_run  # noqa: F401, F811

from gotrain import cli
from gotrain.agent import build_agent
from gotrain.engine import get_engine
from gotrain.tools import final_test, register_model
from gotrain.tools.memory import all_lessons


@pytest.fixture
def splits(monkeypatch):
    seen = []
    real = get_engine().__class__.evaluate

    def spy(self, task, checkpoint, split, **kw):
        seen.append(split)
        return real(self, task, checkpoint, split, **kw)

    monkeypatch.setattr(get_engine().__class__, "evaluate", spy)
    return seen


def scripted(task, phase, **kwargs):
    script = [
        [("tool", "final_test", {"task": task, "version": "v1"})],
        [("tool", "write_lesson", {"task": task, "lesson": "ensembles help", "tags": ["x"]})],
        [("text", "Done.")],
    ]
    return build_agent(task, phase, model=ScriptedModel(script), **kwargs)


def test_rehearsal_scores_val_and_leaves_no_lock_or_ledger(budget, monkeypatch, splits):  # noqa: F811
    home = budget
    register_model(one_run())
    monkeypatch.setenv("GOTRAIN_REHEARSAL", "1")
    out = final_test(TASK, "v1")
    assert splits == ["val"] and out["rehearsal"] == "REHEARSAL (val)" and out["locked"] is False
    assert out["ci_low"] <= out["test_score"] <= out["ci_high"]
    assert not (home / "registry" / TASK / "TEST_LOCK.json").exists()
    assert not (home.parent / "test_ledger.jsonl").exists()
    saved = json.loads((home / "registry" / TASK / "REHEARSAL.json").read_text())
    assert saved["rehearsal"] == "REHEARSAL (val)" and saved["split"] == "val"
    card = (home / "registry" / TASK / "v1" / "model_card.md").read_text(encoding="utf-8")
    assert "## REHEARSAL (val)" in card and "## Test result" not in card
    monkeypatch.delenv("GOTRAIN_REHEARSAL")
    real = final_test(TASK, "v1")  # the real test is still available afterwards
    assert "error" not in real and real["locked"] and splits == ["val", "test"]


def test_cli_rehearsal_runs_finalise_without_confirmation(budget, monkeypatch, splits):  # noqa: F811
    register_model(one_run())
    monkeypatch.setattr(cli, "build_agent", scripted)
    monkeypatch.delenv("GOTRAIN_REHEARSAL", raising=False)
    args = cli.parse_args(["run", "--task", TASK, "--phase", "finalise", "--rehearsal"])
    try:
        assert cli.run(args, ask=lambda _: pytest.fail("must not ask for confirmation")) == 0
    finally:
        monkeypatch.delenv("GOTRAIN_REHEARSAL", raising=False)
    assert splits == ["val"]
    (lesson,) = all_lessons()
    assert lesson["lesson"].startswith("REHEARSAL (val):") and "rehearsal" in lesson["tags"]
    assert not (budget / "registry" / TASK / "TEST_LOCK.json").exists()


def test_rehearsal_flag_is_finalise_only():
    args = cli.parse_args(["run", "--task", TASK, "--phase", "tune", "--rehearsal"])
    assert cli.run(args) == 2
