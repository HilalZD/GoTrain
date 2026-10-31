"""The GoPilot demo: validation tiles only, and a real agent call through the MCP server."""

import sys
from pathlib import Path

import numpy as np
from conftest import make_config
from test_agent import ScriptedModel
from test_tools_models import budget  # noqa: F401, F811

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import demo_gopilot  # noqa: E402

from gotrain.gopilot import export_bundle  # noqa: E402
from gotrain.tools import register_model, run_trials  # noqa: E402


def test_val_tile_reads_only_validation_files(tmp_path, monkeypatch):
    cache = tmp_path / "cache" / "m-brick-kiln"
    cache.mkdir(parents=True)
    x = np.arange(4 * 8 * 8 * 3, dtype=np.float32).reshape(4, 8, 8, 3)
    np.save(cache / "val_x.npy", x)
    np.save(cache / "val_y.npy", np.array([0, 0, 1, 0]))
    monkeypatch.setenv("GOTRAIN_DATA_CACHE", str(tmp_path / "cache"))
    path, label = demo_gopilot.val_tile("m-brick-kiln", tmp_path)
    assert label == 1 and np.array_equal(np.load(path), x[2].transpose(2, 0, 1))
    assert not list(cache.glob("test_*"))  # nothing but val files were needed


def test_agent_calls_the_model_tool_over_mcp_stdio(budget, tmp_path):  # noqa: F811
    run_id = run_trials("m-eurosat", [make_config()], 2.0)["trials"][0]["run_id"]
    card = export_bundle("m-eurosat", register_model(run_id)["version"])
    tile, _ = demo_gopilot.fake_tile("m-eurosat", tmp_path)
    script = [
        [("tool", card["tool_name"], {"path": str(tile)})],
        [("text", "It looks like sea and lake.")],
    ]
    out = demo_gopilot.run_demo("m-eurosat", tile, model=ScriptedModel(script))
    assert card["tool_name"] in out["tools"] and "list_models" in out["tools"]
    assert out["called"] == [card["tool_name"]] and "sea and lake" in out["answer"]
