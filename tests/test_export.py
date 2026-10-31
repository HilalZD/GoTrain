"""gotrain export: files copied, no checkpoints, summary table."""

from test_tools_models import TASK, budget, one_run  # noqa: F401, F811

from gotrain.cli import main
from gotrain.tools import final_test, register_model


def test_export_copies_small_files_and_writes_summary(budget, tmp_path):  # noqa: F811
    home = budget
    register_model(one_run())
    final_test(TASK, "v1")
    (home / "queue_status.json").write_text("[]", encoding="utf-8")
    (home / "label_efficiency").mkdir()
    (home / "label_efficiency" / f"{TASK}.json").write_text("{}", encoding="utf-8")
    out = tmp_path / "results"

    assert main(["export", "--out", str(out), "--home", str(home)]) == 0

    copied = {
        p.relative_to(out / home.name).as_posix()
        for p in (out / home.name).rglob("*")
        if p.is_file()
    }
    assert f"registry/{TASK}/leaderboard.jsonl" in copied
    assert f"registry/{TASK}/v1/model_card.md" in copied
    assert f"registry/{TASK}/TEST_LOCK.json" in copied
    assert "queue_status.json" in copied and f"label_efficiency/{TASK}.json" in copied
    assert not any("checkpoint" in p or p.endswith((".pt", ".pth", ".ckpt")) for p in copied)

    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert f"## {TASK}" in summary and "Published (test): 0.9816" in summary
    v1_row = next(line for line in summary.splitlines() if "| v1 |" in line)
    assert "[" in v1_row  # test score with its CI


def test_export_refuses_a_missing_home(tmp_path):
    assert main(["export", "--out", str(tmp_path / "r"), "--home", str(tmp_path / "nope")]) == 1
