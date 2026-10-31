"""The overnight queue, with the fake engine."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import overnight  # noqa: E402
from conftest import make_config  # noqa: E402
from test_agent import ScriptedModel  # noqa: E402

from gotrain import cli  # noqa: E402
from gotrain.agent import build_agent  # noqa: E402
from gotrain.tools import leaderboard  # noqa: E402
from gotrain.tools.memory import all_lessons, write_lesson  # noqa: E402

TASK = "m-eurosat"


def scripted_agent(task, phase, **kwargs):
    """The real agent wiring with a scripted model: one trial, register the best, 2 lessons."""
    config = make_config()
    lessons = json.dumps([{"lesson": "a", "tags": []}, {"lesson": "b", "tags": []}])
    script = [
        [
            (
                "tool",
                "run_trials",
                {"task": task, "configs": [config], "budget_minutes_per_trial": 2},
            )
        ],
        lambda: [("tool", "register_model", {"run_id": leaderboard(task)["top"][0]["run_id"]})],
        [("text", "Done.")],
        [("text", lessons)],
    ]
    return build_agent(task, phase, model=ScriptedModel(script), **kwargs)


@pytest.fixture(autouse=True)
def fake_agent(monkeypatch):
    monkeypatch.setattr(cli, "build_agent", scripted_agent)


@pytest.fixture(autouse=True)
def stop_file(monkeypatch, tmp_path):
    path = tmp_path / "GOTRAIN_STOP"
    monkeypatch.setenv("GOTRAIN_STOP_FILE", str(path))
    return path


def entry(home, **extra):
    return {"task": TASK, "budget_min": 8, "home": str(home), "memory": "shared", **extra}


def status(home) -> list[dict]:
    return json.loads((home / "queue_status.json").read_text(encoding="utf-8"))


def test_plans_load_and_never_finalise():
    root = Path(__file__).resolve().parents[1] / "configs"
    a, b = (overnight.load_plan(root / f"overnight_{x}.yaml") for x in "ab")
    assert [(e["task"], e["budget_min"]) for e in a[:3]] == [
        ("m-eurosat", 45),
        ("m-brick-kiln", 90),
        ("m-brick-kiln", 90),
    ]
    assert a[2]["memory"] == "empty" and a[3]["label_fracs"] == [0.05, 0.1, 0.25, 0.5]
    assert [e["task"] for e in b] == ["m-pv4ger-seg", "m-chesapeake"]
    assert "finalise" not in (root / "overnight_a.yaml").read_text()


def test_entry_runs_tune_only_and_writes_status_and_log(tmp_path):
    home = tmp_path / "q_dev"
    (row,) = overnight.run_queue([entry(home)])
    assert row["status"] == "ok" and row["registered_version"] == "v1"
    (s,) = status(home)
    assert s["task"] == TASK and s["best_val"] is not None and s["end"]
    assert Path(s["log"]).exists()
    assert not (home / "registry" / TASK / "TEST_LOCK.json").exists()


def test_crash_in_one_entry_does_not_stop_the_queue(tmp_path, monkeypatch):
    real_run = overnight.cli_run
    calls = []

    def flaky(args):
        calls.append(args.task)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return real_run(args)

    monkeypatch.setattr(overnight, "cli_run", flaky)
    rows = overnight.run_queue([entry(tmp_path / "a_dev"), entry(tmp_path / "b_dev")])
    assert rows[0]["status"].startswith("crashed: RuntimeError: boom")
    assert rows[1]["status"] == "ok"
    assert "boom" in Path(rows[0]["log"]).read_text(encoding="utf-8")


def test_kill_switch_stops_after_the_current_entry(tmp_path, monkeypatch, stop_file):
    real_run = overnight.cli_run

    def touch_then_run(args):
        stop_file.write_text("")  # asked to stop while the first entry is running
        return real_run(args)

    monkeypatch.setattr(overnight, "cli_run", touch_then_run)
    rows = overnight.run_queue([entry(tmp_path / "a_dev"), entry(tmp_path / "b_dev")])
    assert len(rows) == 1 and rows[0]["status"] == "ok"
    assert not (tmp_path / "b_dev").exists()


def test_empty_memory_ignores_shared_lessons_and_keeps_them_apart(tmp_path):
    home = tmp_path / "m_dev"
    shared = entry(home)
    empty = entry(home, memory="empty")
    overnight.run_queue([shared])
    shared_before = overnight.os.environ.get("GOTRAIN_HOME")
    assert shared_before != str(home)  # env restored after the entry
    # inspect the lessons through the same settings the entries used
    with overnight.entry_env(shared):
        shared_lessons = all_lessons()
        write_lesson(TASK, "old shared lesson", ["sentinel-2"])
    assert shared_lessons  # the tune session wrote source:tune lessons
    with overnight.entry_env(empty):
        assert all_lessons() == []
        write_lesson(TASK, "isolated", ["x"])
        assert [x["lesson"] for x in all_lessons()] == ["isolated"]
    with overnight.entry_env(shared):
        assert "isolated" not in [x["lesson"] for x in all_lessons()]


def test_label_fracs_write_validation_curve(tmp_path):
    home = tmp_path / "l_dev"
    (row,) = overnight.run_queue([entry(home, label_fracs=[0.25, 0.5])])
    assert row["status"] == "ok"
    curve = json.loads((home / "label_efficiency" / f"{TASK}.json").read_text(encoding="utf-8"))
    assert curve["split"] == "val" and [r["frac"] for r in curve["rows"]] == [0.25, 0.5]
    assert all(r["val_score"] is not None for r in curve["rows"])


def test_label_entry_reuses_an_already_registered_model(tmp_path, monkeypatch):
    home = tmp_path / "r_dev"
    overnight.run_queue([entry(home)])
    monkeypatch.setattr(overnight, "cli_run", lambda args: pytest.fail("must not tune again"))
    (row,) = overnight.run_queue([entry(home, label_fracs=[0.5])])
    assert row["status"] == "ok" and (home / "label_efficiency" / f"{TASK}.json").exists()
