"""GoPilot export bundle: same predictions as engine.predict, a card, and a clear band error."""

import numpy as np
import pytest
from test_candidates import TASK, cand, runs  # noqa: F401, F811

from gotrain.engine import get_engine
from gotrain.gopilot import export_bundle, export_model, list_bundles, predict_bundle
from gotrain.tools import evaluate_candidates, final_test, register_model
from gotrain.tools.registry import load_version, version_checkpoints


def images(n=3, bands=13, side=64):
    return np.random.default_rng(0).uniform(0, 3000, (n, bands, side, side)).astype(np.float32)


def test_ensemble_tta_bundle_predicts_exactly_like_the_engine(runs):  # noqa: F811
    evaluate_candidates(TASK, [cand(runs, True)])
    version = register_model(run_ids=runs, tta=True)["version"]
    out = export_model(TASK, version)
    assert out["tool_name"] == "classify_land_cover_eurosat"

    (card,) = list_bundles()
    assert card["members"] == runs and card["tta"] is True and len(card["checkpoints"]) == len(runs)
    x = images(bands=len(card["bands"]))
    got = predict_bundle(card, x)
    mv = load_version(TASK, version)
    want = get_engine().predict(TASK, version_checkpoints(mv), x, tta=True)
    assert np.array_equal(got["probabilities"], want["probabilities"])
    assert np.array_equal(got["labels"], want["labels"]) and got["n_checkpoints"] == len(runs)


def test_single_run_bundle_and_card_contents(runs):  # noqa: F811
    version = register_model(runs[0])["version"]
    card = export_bundle(TASK, version)
    engine_card = get_engine().describe_dataset(TASK)
    assert card["class_names"] == engine_card.class_names  # the dataset card's names
    assert card["bands"] == engine_card.bands and card["sensor"] == engine_card.sensor
    assert card["test"] is None and len(card["version_hash"]) == 12
    x = images(bands=len(card["bands"]))
    mv = load_version(TASK, version)
    want = get_engine().predict(TASK, version_checkpoints(mv)[0], x)
    assert np.array_equal(predict_bundle(card, x)["probabilities"], want["probabilities"])


def test_export_after_finalise_adds_the_test_score(runs, home):  # noqa: F811
    version = register_model(runs[0])["version"]
    before = export_bundle(TASK, version)
    final_test(TASK, version)
    after = export_bundle(TASK, version)
    assert before["test"] is None and 0 < after["test"]["test_score"] <= 1
    assert after["version_hash"] == before["version_hash"]  # same model, same hash
    card_md = (home / "registry" / TASK / version / "gopilot" / "model_card.md").read_text(
        encoding="utf-8"
    )
    assert "Test: 0." in card_md and "classify_land_cover_eurosat" in card_md


def test_wrong_band_count_is_a_clear_error(runs):  # noqa: F811
    version = register_model(runs[0])["version"]
    card = export_bundle(TASK, version)
    with pytest.raises(ValueError, match=r"needs N x 13 x H x W.*got shape \(2, 3, 64, 64\)"):
        predict_bundle(card, images(n=2, bands=3))


def test_export_unknown_version_is_a_tool_error(runs):  # noqa: F811
    assert "no registered version" in export_model(TASK, "v9")["error"]
