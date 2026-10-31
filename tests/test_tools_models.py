"""inspect_dataset, memory, diagnose, register_model, final_test and the never-raise rule."""

import importlib

import pytest
from conftest import make_config

from gotrain.contracts import ModelVersion
from gotrain.engine.fake import FakeEngine
from gotrain.tools import (
    diagnose,
    final_test,
    init_task_budget,
    inspect_dataset,
    leaderboard,
    promote,
    read_lessons,
    register_model,
    run_trials,
    write_lesson,
)

TASK = "m-eurosat"


@pytest.fixture
def budget(home):
    for task in (TASK, "m-pv4ger-seg"):
        init_task_budget(task, 100)
    return home


def one_run(**overrides) -> str:
    return run_trials(TASK, [make_config(**overrides)], 2.0)["results"][0]["run_id"]


# ---------------------------------------------------------------- memory + inspect


def test_lessons_roundtrip_and_tag_matching(home):
    write_lesson(TASK, "SSL4EO beats imagenet by ~3 points.", ["Sentinel-2", "classification"])
    write_lesson("m-chesapeake-landcover", "## not a header\nDice helps rare classes.", ["naip"])
    found = read_lessons(["sentinel-2"])
    assert found["n_found"] == 1
    assert found["lessons"][0]["tags"] == ["classification", "sentinel-2"]
    assert (
        read_lessons(["naip"])["lessons"][0]["lesson"] == "not a header\nDice helps rare classes."
    )
    assert read_lessons([])["n_found"] == 2


def test_write_lesson_rejects_bad_input(home):
    assert "empty" in write_lesson(TASK, "   ", ["x"])["error"]
    assert "at least one tag" in write_lesson(TASK, "ok", [])["error"]
    assert "list of strings" in write_lesson(TASK, "ok", "sentinel-2")["error"]
    assert "limit" in write_lesson(TASK, "x" * 5000, ["x"])["error"]


def test_inspect_dataset_returns_summary_and_matching_lessons(home):
    write_lesson(TASK, "rot90 helps on S2.", ["sentinel-2"])
    write_lesson("m-pv4ger-seg", "use dice.", ["aerial-rgb"])
    out = inspect_dataset("m-brick-kiln")
    assert out["dataset"]["sensor"] == "sentinel-2" and out["dataset"]["n_bands"] == 13
    assert "band_mean" not in out["dataset"]  # compact
    assert "ssl4eo_s2" in out["dataset"]["weights_hint"]
    assert [x["lesson"] for x in out["lessons"]] == ["rot90 helps on S2."]


def test_inspect_unknown_task(home):
    out = inspect_dataset("m-nope")
    assert "unknown task" in out["error"] and "m-eurosat" in out["hint"]


# ---------------------------------------------------------------- diagnose


def test_diagnose_overfitting_and_underfitting(budget):
    over = run_trials(TASK, [make_config(augment=[], max_epochs=60)], 7.0)["results"][0]
    out = diagnose(over["run_id"])
    assert out["curve_verdict"] == "overfitting" and "augmentation" in out["advice"]
    assert out["per_class_metric"] == "recall" and len(out["weakest_classes"]) == 3
    assert "confusion_matrix" not in out  # summarised

    assert diagnose(one_run(max_epochs=3))["curve_verdict"] == "underfitting"


def test_diagnose_finds_rare_segmentation_class(budget):
    cfg = make_config(weights="imagenet", img_size=320, loss="dice_bce")
    run_id = run_trials("m-pv4ger-seg", [cfg], 10.0)["results"][0]["run_id"]
    out = diagnose(run_id)
    assert out["per_class_metric"] == "iou"
    assert next(iter(out["weakest_classes"])) == "solar_panel"


def test_diagnose_refuses_failed_and_unknown_runs(budget):
    assert "no model to diagnose" in diagnose(one_run(bands=["B99"]))["error"]
    assert "no run with run_id" in diagnose("garbage")["error"]


# ---------------------------------------------------------------- registry


def test_register_model_writes_version_folder(budget):
    run_id = one_run()
    out = register_model(run_id, notes="first baseline")
    assert out["version"] == "v1"

    vdir = budget / "registry" / TASK / "v1"
    mv = ModelVersion.model_validate_json((vdir / "model_version.json").read_text())
    assert mv.run_id == run_id and mv.config.task == TASK
    assert (vdir / "checkpoints" / "best.ckpt").exists()
    card = (vdir / "model_card.md").read_text()
    assert run_id in card and "first baseline" in card and "sentinel-2" in card

    assert "already registered as v1" in register_model(run_id)["error"]
    assert register_model(one_run(seed=1))["version"] == "v2"


def test_register_refuses_error_runs(budget):
    out = register_model(one_run(bands=["B99"]))
    assert "only successful runs can be registered" in out["error"]
    assert not (budget / "registry" / TASK / "v1").exists()


def test_model_card_shows_lineage(budget):
    parent = run_trials(TASK, [make_config()], 1.0)["results"][0]["run_id"]
    child = promote(parent, 2.0)["child"]["run_id"]
    register_model(child)
    card = (budget / "registry" / TASK / "v1" / "model_card.md").read_text()
    assert f"`{child}` <- `{parent}`" in card


# ---------------------------------------------------------------- final test


def test_final_test_runs_once_then_refuses(budget):
    register_model(one_run())
    first = final_test(TASK, "v1")
    assert first["locked"] is True and 0 < first["test_score"] <= first["val_score"]

    second = final_test(TASK, "v1")
    assert "already been used" in second["error"]
    assert second["lock"]["status"] == "done"
    assert second["lock"]["test_score"] == first["test_score"]


def test_final_test_unknown_version_does_not_lock(budget):
    register_model(one_run())
    out = final_test(TASK, "v9")
    assert "no registered version 'v9'" in out["error"] and "v1" in out["hint"]
    assert not (budget / "registry" / TASK / "TEST_LOCK.json").exists()


def test_final_test_crash_keeps_lock(budget, monkeypatch):
    register_model(one_run())

    class Broken:
        def evaluate(self, *args, **kwargs):
            raise RuntimeError("GPU fell over")

    module = importlib.import_module("gotrain.tools.final_test")  # not the same-named function
    monkeypatch.setattr(module, "get_engine", lambda: Broken())
    assert "crashed" in final_test(TASK, "v1")["error"]
    assert final_test(TASK, "v1")["lock"]["status"] == "failed"


# ---------------------------------------------------------------- shared rules


def test_tools_never_raise(budget, monkeypatch):
    assert "non-empty list" in run_trials(TASK, "not a list", 1.0)["error"]
    assert "unexpectedly" in run_trials(TASK)["error"]  # missing argument

    def boom(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr("gotrain.tools.inspect.get_engine", boom)
    out = inspect_dataset(TASK)
    assert "RuntimeError: disk on fire" in out["error"] and "budget" in out


def test_engine_crash_is_a_failed_trial_not_a_tool_error(budget, monkeypatch):
    class Crashing(FakeEngine):
        def train(self, *args, **kwargs):
            raise MemoryError("CUDA out of memory")

    monkeypatch.setattr("gotrain.tools.trials.get_engine", lambda: Crashing())
    (row,) = run_trials(TASK, [make_config()], 1.0)["results"]
    assert row["status"] == "error" and "CUDA out of memory" in row["error"]


@pytest.mark.parametrize(
    "call",
    [
        lambda: inspect_dataset(TASK),
        lambda: run_trials(TASK, [make_config()], 1.0),
        lambda: run_trials(TASK, [], 1.0),
        lambda: leaderboard(TASK),
        lambda: diagnose(one_run()),
        lambda: register_model(one_run()),
        lambda: final_test(TASK, "v1"),
        lambda: write_lesson(TASK, "x", ["y"]),
        lambda: read_lessons(["y"], task=TASK),
    ],
)
def test_every_output_reports_task_budget(budget, call):
    out = call()
    assert set(out["budget"]) == {"used_min", "remaining_min"}
