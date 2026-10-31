"""Error-grid images: downscaling, the diagnose image block, fallback, and audit hygiene."""

import json
import random
import re
from pathlib import Path

import numpy as np
import pytest
from conftest import make_config
from PIL import Image
from test_agent import ScriptedModel

from gotrain.agent import IMAGE_ADAPTERS, build_agent
from gotrain.audit import AuditLog
from gotrain.contracts import TrainConfig
from gotrain.engine.fake import CARDS, FakeEngine, draw_error_grid, fake_confusion
from gotrain.images import MAX_BYTES, ImageError, prepare_image
from gotrain.tools import diagnose, init_task_budget, run_halving, run_trials

TASK = "m-pv4ger-seg"
PNG_MAGIC = b"\x89PNG"
REPO = Path(__file__).resolve().parents[1]


def seg_run() -> str:
    init_task_budget(TASK, 100)
    cfg = make_config(weights="imagenet", img_size=320, loss="dice_bce")
    return run_trials(TASK, [cfg], 10.0)["trials"][0]["run_id"]


def tune_agent(audit: AuditLog | None = None):
    """Keep the result in a variable: agent.tool holds only a weak reference to the agent."""
    return build_agent(TASK, "tune", audit=audit, model=ScriptedModel([]))


# ---------------------------------------------------------------- downscaling


def test_downscales_long_side_and_bytes(tmp_path):
    noise = np.random.default_rng(0).integers(0, 256, (1500, 3000, 3), dtype=np.uint8)
    path = tmp_path / "big.png"
    Image.fromarray(noise).save(path)
    assert path.stat().st_size > MAX_BYTES  # random noise barely compresses

    prepared = prepare_image(path)
    assert max(prepared.width, prepared.height) <= 1024
    assert len(prepared.png) < MAX_BYTES
    assert prepared.png.startswith(PNG_MAGIC)
    assert prepared.width / prepared.height == pytest.approx(2.0, rel=0.02)  # aspect kept
    assert prepared.info()["original_size"] == [3000, 1500]
    assert "png" not in prepared.info()  # metadata only


def test_small_image_is_not_upscaled(tmp_path):
    Image.new("RGB", (300, 200), "white").save(tmp_path / "small.png")
    prepared = prepare_image(tmp_path / "small.png")
    assert (prepared.width, prepared.height) == (300, 200)


@pytest.mark.parametrize("content", [None, b"not a png at all"])
def test_missing_or_unreadable_image_raises(tmp_path, content):
    path = tmp_path / "grid.png"
    if content is not None:
        path.write_bytes(content)
    with pytest.raises(ImageError, match="cannot read image"):
        prepare_image(path)


# ---------------------------------------------------------------- fake engine grid


@pytest.mark.parametrize("task", ["m-pv4ger-seg", "m-eurosat"])
def test_fake_engine_draws_error_grid(tmp_path, task):
    card = CARDS[task]
    _, matrix = fake_confusion(card, 0.85, random.Random(1))
    path = draw_error_grid(card, matrix, 7, tmp_path / "grid.png")
    with Image.open(path) as image:
        assert image.size == (3 * 128 + 2 * 4, 16 + 4 * (128 + 16))
    again = draw_error_grid(card, matrix, 7, tmp_path / "again.png")
    assert again.read_bytes() == path.read_bytes()  # deterministic


def test_fake_diagnose_returns_grid_path(tmp_path):
    engine = FakeEngine(sleep_s=0)
    cfg = make_config(weights="imagenet", img_size=320, loss="dice_bce")
    engine.train(TrainConfig(task=TASK, **cfg), 10.0, tmp_path / "run")
    out = engine.diagnose(TASK, tmp_path / "run")
    assert Path(out["error_grid_path"]).is_file()


# ---------------------------------------------------------------- diagnose tool


def test_diagnose_attaches_image_block_when_png_exists():
    run_id = seg_run()
    agent = tune_agent()
    result = agent.tool.diagnose(run_id=run_id)
    assert result["status"] == "success"
    text_blocks = [b for b in result["content"] if "text" in b]
    images = [b["image"] for b in result["content"] if "image" in b]
    assert len(text_blocks) == 1 and len(images) == 1
    assert images[0]["format"] == "png" and images[0]["source"]["bytes"].startswith(PNG_MAGIC)

    summary = json.loads(text_blocks[0]["text"])
    assert summary["curve_verdict"] and summary["budget"]  # budget lives in the text block
    grid = summary["error_grid"]
    assert grid["size_bytes"] == len(images[0]["source"]["bytes"])
    assert max(grid["width"], grid["height"]) <= 1024


def test_plain_diagnose_returns_json_only():
    out = diagnose(seg_run())
    json.dumps(out)  # no bytes anywhere
    assert out["error_grid"]["key"].endswith("error_grid_for_model.png")


def test_diagnose_falls_back_to_text_when_png_missing(monkeypatch, tmp_path):
    run_id = seg_run()
    real = FakeEngine.diagnose

    def without_png(self, task, run_dir):
        return real(self, task, run_dir) | {"error_grid_path": str(tmp_path / "gone.png")}

    monkeypatch.setattr(FakeEngine, "diagnose", without_png)
    agent = tune_agent()
    result = agent.tool.diagnose(run_id=run_id)
    assert [list(b) for b in result["content"]] == [["text"]]
    summary = json.loads(result["content"][0]["text"])
    assert summary["error_grid"] is None
    assert "error grid not shown, text only" in summary["image_note"]


def test_diagnose_without_engine_grid_is_text_only(monkeypatch):
    run_id = seg_run()
    real = FakeEngine.diagnose
    monkeypatch.setattr(
        FakeEngine, "diagnose", lambda self, t, d: real(self, t, d) | {"error_grid_path": None}
    )
    agent = tune_agent()
    result = agent.tool.diagnose(run_id=run_id)
    assert [list(b) for b in result["content"]] == [["text"]]
    assert "image_note" not in json.loads(result["content"][0]["text"])


def test_only_diagnose_can_send_images():
    assert [fn.__name__ for fn in IMAGE_ADAPTERS] == ["diagnose"]
    builds_image = re.compile(r"""\{\s*["']image["']\s*:""")
    offenders = [
        str(p.relative_to(REPO))
        for p in (REPO / "gotrain").rglob("*.py")
        if builds_image.search(p.read_text(encoding="utf-8"))
    ]
    assert offenders == [str(Path("gotrain/tools/diagnose.py"))]


# ---------------------------------------------------------------- audit


def test_audit_logs_image_metadata_but_never_bytes(home):
    run_id = seg_run()
    audit = AuditLog(TASK, "tune", session_id="img")
    agent = tune_agent(audit)
    agent.tool.diagnose(run_id=run_id)

    raw = (home / audit.jsonl_key).read_bytes()
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    (shown,) = [r for r in records if r["event"] == "image_shown"]
    assert shown["tool"] == "diagnose" and shown["format"] == "png"
    assert shown["path"].endswith("error_grid_for_model.png")
    assert shown["width"] and shown["height"] and shown["size_bytes"] > 1000

    for marker in (PNG_MAGIC, b"\\x89PNG", b"iVBORw0KGgo", b"b'\\x"):  # raw, repr, base64
        assert marker not in raw
    assert len(raw) < 20_000  # the ~100 KB image is not in the log
    journal = (home / audit.journal_key).read_text(encoding="utf-8")
    assert "image shown:" in journal


# ---------------------------------------------------------------- halving rows


def test_halving_rows_include_minutes_per_epoch():
    init_task_budget("m-eurosat", 100)
    configs = [make_config(seed=s) for s in range(3)]
    out = run_halving("m-eurosat", configs, 1.0, 3.0, 1)
    assert all(t["minutes_per_epoch"] == pytest.approx(0.1, abs=0.03) for t in out["trials"])
