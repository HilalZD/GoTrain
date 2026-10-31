"""Engine-only training rules: checked before training, mirrored by the fake engine,
shown in inspect_dataset and the prompt, and kept in sync with the real engine."""

import json

import pytest
from conftest import make_config

from gotrain.agent import load_prompt
from gotrain.contracts import TrainConfig
from gotrain.engine.fake import CARDS, FakeEngine
from gotrain.engine.rules import (
    BINARY_SEG_LOSSES,
    LOSSES_BY_TASK_TYPE,
    SEG_IMG_MULTIPLE,
    check_task_rules,
    rules_for,
)
from gotrain.tools import init_task_budget, inspect_dataset, promote, run_trials

SEG, MULTI_SEG = "m-pv4ger-seg", "m-chesapeake-landcover"  # 2 classes, 7 classes


def cfg(task: str, **overrides) -> TrainConfig:
    return TrainConfig(task=task, **make_config(weights="imagenet", **overrides))


# ---------------------------------------------------------------- the rules


def test_rules_match_the_real_engine():
    """The real engine's own constants (Serena's files); importing them needs no torch."""
    from gotrain.engine import models, train

    assert LOSSES_BY_TASK_TYPE == train.LOSSES
    assert BINARY_SEG_LOSSES == train.BINARY_SEG_LOSSES
    assert SEG_IMG_MULTIPLE == models.UNET_STRIDE


@pytest.mark.parametrize(
    "task, overrides, expected",
    [
        ("m-eurosat", {"loss": "dice_ce"}, "does not fit a classification task"),
        ("m-eurosat", {"loss": "dice_bce"}, "Allowed for m-eurosat: ['ce', 'focal', 'bce']"),
        (MULTI_SEG, {"loss": "dice_bce", "img_size": 256}, "for 2-class masks and"),
        (MULTI_SEG, {"loss": "bce", "img_size": 256}, "has 7 classes"),
        (SEG, {"loss": "dice_bce", "img_size": 300}, "Use e.g. 288 or 320"),
        (SEG, {"loss": "ce", "img_size": 40}, "Use e.g. 32 or 64"),
    ],
)
def test_bad_configs_are_rejected_with_a_fix(task, overrides, expected):
    problem = check_task_rules(CARDS[task], cfg(task, **overrides))
    assert problem is not None and expected in problem


@pytest.mark.parametrize(
    "task, overrides",
    [
        ("m-eurosat", {"loss": "focal", "img_size": 64}),
        ("m-eurosat", {"loss": "ce", "img_size": 100}),  # classification: any size
        (SEG, {"loss": "dice_bce", "img_size": 320}),
        (MULTI_SEG, {"loss": "dice_ce", "img_size": 256}),
    ],
)
def test_good_configs_pass(task, overrides):
    assert check_task_rules(CARDS[task], cfg(task, **overrides)) is None


# ---------------------------------------------------------------- tools, engine, inspect, prompt


def test_run_trials_rejects_before_training_and_runs_the_rest(home):
    init_task_budget(SEG, 100)
    configs = [
        make_config(weights="imagenet", img_size=300, loss="dice_bce"),
        make_config(weights="imagenet", img_size=320, loss="dice_bce"),
    ]
    out = run_trials(SEG, configs, 10.0)
    assert [t["status"] for t in out["trials"]] == ["ok"]
    (bad,) = out["invalid"]
    assert bad["index"] == 0 and "not a multiple of 32" in bad["error"]
    assert len(list((home / "runs" / SEG).iterdir())) == 1  # the bad one never trained


def test_promote_rechecks_the_rules(home):
    init_task_budget(SEG, 100)
    good = make_config(weights="imagenet", img_size=320, loss="dice_bce", max_epochs=40)
    run_id = run_trials(SEG, [good], 10.0)["trials"][0]["run_id"]
    path = home / "runs" / SEG / run_id / "config.json"
    path.write_text(json.dumps(json.loads(path.read_text()) | {"img_size": 300}))
    assert "not a multiple of 32" in promote(run_id, 5.0)["error"]


def test_fake_engine_fails_like_the_real_one(tmp_path):
    result = FakeEngine(sleep_s=0).train(cfg("m-eurosat", loss="dice_ce"), 5.0, tmp_path / "r")
    assert result.status == "error" and "does not fit a classification task" in result.error


def test_inspect_dataset_shows_the_rules():
    seg = inspect_dataset(SEG)["dataset"]["training_rules"]
    assert seg == rules_for(CARDS[SEG])
    assert seg["img_size_multiple_of"] == 32 and seg["binary_only_losses"] == ["bce", "dice_bce"]
    multi = inspect_dataset(MULTI_SEG)["dataset"]["training_rules"]
    assert "dice_bce" not in multi["allowed_losses"]
    assert inspect_dataset("m-eurosat")["dataset"]["training_rules"] == {
        "allowed_losses": ["ce", "focal", "bce"]
    }


def test_tune_prompt_states_the_rules():
    prompt = load_prompt("tune")
    assert "multiple of 32" in prompt
    for task_type in ("classification", "segmentation"):
        losses = LOSSES_BY_TASK_TYPE[task_type]
        line = next(ln for ln in prompt.splitlines() if f"{task_type}:" in ln.lower())
        assert all(f"`{loss}`" in line for loss in losses), (task_type, line)
