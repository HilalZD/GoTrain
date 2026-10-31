"""Valid backbone + weights combinations: helper, tools, fake engine and prompt agree."""

import json

import pytest
from conftest import make_config

from gotrain.agent import load_prompt
from gotrain.contracts import TrainConfig
from gotrain.engine.combos import VALID_COMBOS, check_combo, is_sentinel2, valid_combos_for
from gotrain.engine.fake import CARDS, FakeEngine
from gotrain.tools import init_task_budget, inspect_dataset, promote, run_trials

EUROSAT, PV4GER = CARDS["m-eurosat"], CARDS["m-pv4ger-seg"]


# ---------------------------------------------------------------- helper


@pytest.mark.parametrize(
    "card, backbone, weights",
    [
        (EUROSAT, "resnet50", "ssl4eo_s2"),
        (EUROSAT, "resnet18", "ssl4eo_s2"),
        (EUROSAT, "convnext_tiny", "imagenet"),
        (PV4GER, "resnet34", "imagenet"),
        (PV4GER, "convnext_tiny", "none"),
    ],
)
def test_valid_combo_passes(card, backbone, weights):
    assert check_combo(card, backbone, weights) is None


def test_ssl4eo_with_convnext_tiny_rejected():
    msg = check_combo(EUROSAT, "convnext_tiny", "ssl4eo_s2")
    assert "backbone='convnext_tiny' is not available with weights='ssl4eo_s2'" in msg
    assert "weights='ssl4eo_s2': resnet18, resnet50" in msg  # lists the valid options
    assert "weights='imagenet': resnet18, resnet34, resnet50, convnext_tiny" in msg


def test_ssl4eo_on_aerial_task_rejected():
    msg = check_combo(PV4GER, "resnet50", "ssl4eo_s2")
    assert "only works on Sentinel-2 tasks, not aerial-rgb" in msg
    assert "ssl4eo_s2" not in msg.split("Valid backbone")[1]  # not offered for this task


def test_unknown_backbone_rejected():
    assert "backbone='vit_large'" in check_combo(EUROSAT, "vit_large", "imagenet")


@pytest.mark.parametrize("sensor", ["sentinel-2", "Sentinel-2", "sentinel2", "SENTINEL_2"])
def test_sentinel2_spellings(sensor):
    assert is_sentinel2(EUROSAT.model_copy(update={"sensor": sensor}))


def test_valid_combos_for_task():
    assert set(valid_combos_for(EUROSAT)) == {"imagenet", "none", "ssl4eo_s2"}
    assert set(valid_combos_for(PV4GER)) == {"imagenet", "none"}


# ---------------------------------------------------------------- tools


@pytest.fixture
def budgets():
    for task in ("m-eurosat", "m-pv4ger-seg"):
        init_task_budget(task, 100)


def test_run_trials_rejects_invalid_combos_before_training(budgets, home):
    configs = [
        make_config(backbone="convnext_tiny", weights="ssl4eo_s2"),
        make_config(backbone="resnet18", weights="ssl4eo_s2"),
    ]
    out = run_trials("m-eurosat", configs, 2.0)
    assert [r["status"] for r in out["results"]] == ["ok"]
    (bad,) = out["invalid"]
    assert bad["index"] == 0 and "resnet18, resnet50" in bad["error"]
    assert len(list((home / "runs" / "m-eurosat").iterdir())) == 1  # nothing trained for it
    assert out["budget"]["used_min"] == out["results"][0]["minutes"]


def test_run_trials_rejects_ssl4eo_on_pv4ger(budgets):
    seg = {"img_size": 320, "loss": "dice_bce"}
    configs = [make_config(weights="ssl4eo_s2", **seg), make_config(weights="imagenet", **seg)]
    out = run_trials("m-pv4ger-seg", configs, 10.0)
    assert len(out["results"]) == 1 and out["invalid"][0]["index"] == 0
    assert "only works on Sentinel-2 tasks" in out["invalid"][0]["error"]


def test_run_trials_unknown_task(home):
    init_task_budget("m-nope", 10)
    assert "unknown task" in run_trials("m-nope", [make_config()], 1.0)["error"]


def test_promote_rechecks_combo(budgets, home):
    run_id = run_trials("m-eurosat", [make_config()], 1.0)["results"][0]["run_id"]
    config_path = home / "runs" / "m-eurosat" / run_id / "config.json"
    tampered = json.loads(config_path.read_text()) | {"backbone": "convnext_tiny"}
    config_path.write_text(json.dumps(tampered))  # e.g. the valid list changed since
    out = promote(run_id, 2.0)
    assert "not available with weights='ssl4eo_s2'" in out["error"]


def test_inspect_dataset_lists_valid_combos():
    combos = inspect_dataset("m-pv4ger-seg")["dataset"]["valid_backbones_by_weights"]
    assert combos == {
        "imagenet": list(VALID_COMBOS["imagenet"]),
        "none": list(VALID_COMBOS["none"]),
    }


# ---------------------------------------------------------------- fake engine + prompt


def test_fake_engine_errors_on_invalid_combo(tmp_path):
    cfg = TrainConfig(task="m-pv4ger-seg", **make_config(weights="ssl4eo_s2", img_size=320))
    result = FakeEngine(sleep_s=0).train(cfg, 10.0, tmp_path / "run")
    assert result.status == "error" and "Valid backbone + weights" in result.error


def test_tune_prompt_matches_valid_combos():
    prompt = load_prompt("tune")
    for weights, backbones in VALID_COMBOS.items():
        line = next(ln for ln in prompt.splitlines() if f'"{weights}"' in ln and ":" in ln)
        listed = line.split(":", 1)[1]
        for backbone in VALID_COMBOS["imagenet"]:
            assert (backbone in listed) == (backbone in backbones), (weights, backbone)
