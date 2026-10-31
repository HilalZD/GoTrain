"""Dev homes and the test vault: official home + machine-wide test ledger (real engine)."""

import importlib
import json
from pathlib import Path

import pytest
from conftest import make_config
from pydantic import ValidationError

from gotrain import cli
from gotrain.config import get_settings, is_dev_home
from gotrain.engine.fake import FakeEngine
from gotrain.tools import final_test, init_task_budget, register_model, run_trials
from gotrain.vault import ledger_entries

TASK = "m-eurosat"


# ---------------------------------------------------------------- dev homes


@pytest.mark.parametrize(
    "name, dev",
    [
        ("gotrain_dev", True),
        ("gotrain_dev_serena", True),
        ("my_dev", True),
        ("gotrain_home", False),
        ("gotrain_home_serena", False),
        ("gotrain_devx", False),
        ("gotrain-dev", False),
    ],
)
def test_dev_home_names(name, dev):
    assert is_dev_home(Path("/x") / name) is dev


def test_fake_engine_works_in_serenas_dev_home(monkeypatch, tmp_path):
    monkeypatch.setenv("GOTRAIN_HOME", str(tmp_path / "gotrain_dev_serena"))
    assert get_settings().home.name == "gotrain_dev_serena"
    monkeypatch.setenv("GOTRAIN_HOME", str(tmp_path / "gotrain_home_serena"))
    with pytest.raises(ValidationError, match="is not a dev home"):
        get_settings()


# ---------------------------------------------------------------- vault helpers


def build_registered_home(monkeypatch, path: Path) -> Path:
    """A home with v1 registered, built with the fake engine (so its name ends in _dev)."""
    monkeypatch.setenv("GOTRAIN_ENGINE", "fake")
    monkeypatch.setenv("GOTRAIN_HOME", str(path))
    init_task_budget(TASK, 20)
    register_model(run_trials(TASK, [make_config()], 2.0)["trials"][0]["run_id"])
    return path


def go_real(monkeypatch, home: Path, official: Path) -> None:
    """Switch to the real engine, with the fake engine standing in for evaluation."""
    monkeypatch.setenv("GOTRAIN_ENGINE", "real")
    monkeypatch.setenv("GOTRAIN_HOME", str(home))
    monkeypatch.setenv("GOTRAIN_OFFICIAL_HOME", str(official))
    module = importlib.import_module("gotrain.tools.final_test")  # not the same-named function
    monkeypatch.setattr(module, "get_engine", lambda: FakeEngine(sleep_s=0))


@pytest.fixture
def official(monkeypatch, tmp_path) -> Path:
    return build_registered_home(monkeypatch, tmp_path / "official_dev")


# ---------------------------------------------------------------- final_test


def test_real_final_test_refused_outside_the_official_home(monkeypatch, tmp_path, official):
    other = build_registered_home(monkeypatch, tmp_path / "other_dev")
    go_real(monkeypatch, other, official)
    out = final_test(TASK, "v1")
    assert "test vault: real test evaluations run only in the official home" in out["error"]
    assert not (other / "registry" / TASK / "TEST_LOCK.json").exists()
    assert ledger_entries() == []


def test_real_final_test_runs_once_and_is_recorded_in_the_ledger(monkeypatch, official):
    go_real(monkeypatch, official, official)
    first = final_test(TASK, "v1")
    assert first["locked"] is True
    started, done = ledger_entries(TASK)
    assert (started["status"], done["status"]) == ("started", "done")
    assert done["test_score"] == first["test_score"] and done["home"] == str(official)


def test_ledger_blocks_a_second_home_even_without_its_lock(monkeypatch, tmp_path, official):
    go_real(monkeypatch, official, official)
    assert final_test(TASK, "v1")["locked"] is True

    # The loophole: a fresh home has no TEST_LOCK.json. Even made official, it is refused.
    second = build_registered_home(monkeypatch, tmp_path / "second_dev")
    go_real(monkeypatch, second, second)
    out = final_test(TASK, "v1")
    assert "machine-wide test ledger" in out["error"] and "has been used" in out["error"]
    assert [r["status"] for r in out["ledger"]] == ["started", "done"]
    assert not (second / "registry" / TASK / "TEST_LOCK.json").exists()


def test_a_crashed_real_test_also_blocks_retries(monkeypatch, tmp_path, official):
    go_real(monkeypatch, official, official)

    class Broken:
        def evaluate(self, *args, **kwargs):
            raise RuntimeError("GPU fell over")

    module = importlib.import_module("gotrain.tools.final_test")
    monkeypatch.setattr(module, "get_engine", lambda: Broken())
    assert "crashed" in final_test(TASK, "v1")["error"]
    assert [r["status"] for r in ledger_entries(TASK)] == ["started", "failed"]

    second = build_registered_home(monkeypatch, tmp_path / "second_dev")
    go_real(monkeypatch, second, second)
    assert "status 'failed'" in final_test(TASK, "v1")["error"]


def test_fake_engine_tests_never_touch_the_ledger(monkeypatch, tmp_path):
    build_registered_home(monkeypatch, tmp_path / "a_dev")
    assert final_test(TASK, "v1")["locked"] is True
    build_registered_home(monkeypatch, tmp_path / "b_dev")
    assert final_test(TASK, "v1")["locked"] is True  # per-home lock only, for dev runs
    assert ledger_entries() == []
    assert not Path(get_settings().test_ledger).exists()


# ---------------------------------------------------------------- CLI finalise


def run_finalise(monkeypatch) -> tuple[int, list]:
    built = []
    monkeypatch.setattr(cli, "build_agent", lambda *a, **k: built.append(a) or None)
    args = cli.parse_args(["run", "--task", TASK, "--phase", "finalise"])
    monkeypatch.setattr(cli, "get_engine", lambda: FakeEngine(sleep_s=0))  # no torch needed
    return cli.run(args, ask=lambda prompt: TASK), built


def test_cli_finalise_refused_outside_the_official_home(monkeypatch, tmp_path, official):
    other = build_registered_home(monkeypatch, tmp_path / "other_dev")
    go_real(monkeypatch, other, official)
    code, built = run_finalise(monkeypatch)
    assert code == 1 and built == []


def test_cli_finalise_refused_when_the_ledger_has_the_task(monkeypatch, official):
    go_real(monkeypatch, official, official)
    ledger = get_settings().test_ledger
    ledger.write_text(json.dumps({"ts": "t", "task": TASK, "status": "done", "home": "h"}) + "\n")
    code, built = run_finalise(monkeypatch)
    assert code == 1 and built == []
