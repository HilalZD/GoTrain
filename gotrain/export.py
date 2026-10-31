"""Export results from one or more homes into the repo: `gotrain export --out results/`.

Copies a fixed list of small files per home (never checkpoints) to <out>/<home name>/, and
writes <out>/summary.md with one table per task. Reads files directly, never through the tools.
"""

import json
import re
import shutil
import statistics
from pathlib import Path

from gotrain.knowledge.targets import load_targets

VERSION_DIR = re.compile(r"^v[1-9][0-9]*$")
JOURNAL_SUFFIX = "_journal.md"
REGISTRY_FILES = (
    "leaderboard.jsonl",
    "seed_groups.json",
    "budget.json",
    "TEST_LOCK.json",
    "REHEARSAL.json",
)


def read_json(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_jsonl(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]
    except (OSError, ValueError):
        return []


def copy_file(src: Path, home: Path, dest_root: Path) -> None:
    dest = dest_root / src.relative_to(home)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)


def task_dirs(home: Path) -> list[Path]:
    return sorted(p for p in (home / "registry").glob("*") if p.is_dir())


def versions(task_dir: Path) -> list[Path]:
    return sorted(p for p in task_dir.glob("v*") if p.is_dir() and VERSION_DIR.match(p.name))


def files_to_copy(home: Path) -> list[Path]:
    """Leaderboards, model cards/versions, journals, queue status, label curves, test records."""
    found: list[Path] = []
    for task_dir in task_dirs(home):
        found += [task_dir / name for name in REGISTRY_FILES]
        for vdir in versions(task_dir):
            found += [vdir / "model_card.md", vdir / "model_version.json"]
    found += sorted((home / "logs").glob(f"*/*{JOURNAL_SUFFIX}"))
    found += sorted((home / "label_efficiency").glob("*.json"))
    found += sorted(home.glob("queue_status_*.json"))
    found += [home / "memory" / "lessons.md"]
    return [p for p in found if p.is_file()]


def tokens_used(log_dir: Path) -> int:
    """Bedrock tokens summed over a task's sessions (session_end events)."""
    total = 0
    for log in log_dir.glob("*.jsonl"):
        for event in read_jsonl(log):
            usage = event.get("usage") if event.get("event") == "session_end" else None
            if isinstance(usage, dict):
                total += int(
                    usage.get("totalTokens")
                    or usage.get("inputTokens", 0) + usage.get("outputTokens", 0)
                )
    return total


def val_text(task_dir: Path, version: dict) -> str:
    """The version's validation score: its seed group's mean ± std if it has one."""
    entries = {e["run_id"]: e for e in read_jsonl(task_dir / "leaderboard.jsonl")}
    groups = read_json(task_dir / "seed_groups.json") or {}
    if len(version.get("members") or [0]) == 1:
        for group in groups.values():
            if version["run_id"] not in group["run_ids"]:
                continue
            scores = [
                entries[r]["val_score"]
                for r in group["run_ids"]
                if r in entries and entries[r].get("val_score") is not None
            ]
            if len(scores) >= 2:
                mean, std = statistics.mean(scores), statistics.stdev(scores)
                return f"{mean:.4f} ± {std:.4f} (n={len(scores)})"
    return f"{version['val_score']:.4f} (n=1)"


def test_text(lock: dict | None, version: str) -> str:
    if not lock or lock.get("status") != "done" or lock.get("version") != version:
        return "-"
    text = f"{lock['test_score']:.4f}"
    if lock.get("ci_low") is not None:
        text += f" [{lock['ci_low']:.4f}, {lock['ci_high']:.4f}]"
    return text


def task_rows(home: Path, task_dir: Path) -> list[str]:
    lock = read_json(task_dir / "TEST_LOCK.json")
    budget = read_json(task_dir / "budget.json") or {}
    gpu = f"{budget['used_min']:.0f}" if "used_min" in budget else "-"
    tokens = tokens_used(home / "logs" / task_dir.name)
    rows = []
    for vdir in versions(task_dir):
        version = read_json(vdir / "model_version.json")
        if not version:
            continue
        cells = [home.name, vdir.name, val_text(task_dir, version), test_text(lock, vdir.name)]
        cells += [gpu, f"{tokens:,}" if tokens else "-"]
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def reference_lines(task: str) -> list[str]:
    targets = load_targets().tasks.get(task)
    if targets is None:
        return ["_No published reference for this task._"]
    lines = [
        f"- Published ({r.split}): {r.value:.4f}, {r.model} ({r.source})"
        for r in targets.references
    ]
    if targets.human_baseline:
        hb = targets.human_baseline
        lines.append(f"- Human baseline ({hb.split}): {hb.value:.4f}, {hb.model}")
    return lines


def build_summary(homes: list[Path]) -> str:
    tasks: dict[str, list[tuple[Path, Path]]] = {}
    for home in homes:
        for task_dir in task_dirs(home):
            tasks.setdefault(task_dir.name, []).append((home, task_dir))
    out = [
        "# Results",
        "",
        "Val = validation score (mean ± std over seeds when checked). Test is scored once; its",
        "95% bootstrap CI is in brackets.",
        "",
    ]
    for task, places in sorted(tasks.items()):
        rows = [r for home, task_dir in places for r in task_rows(home, task_dir)]
        if not rows:
            continue
        header = "| home | version | val | test [95% CI] | GPU min | Bedrock tokens |"
        out += [f"## {task}", "", *reference_lines(task), "", header, "|---|---|---|---|---|---|"]
        out += [*rows, ""]
    return "\n".join(out)


def export_results(homes: list[Path], out: Path) -> dict:
    """Copy the files and write summary.md. Returns {"files": n, "summary": path}."""
    out.mkdir(parents=True, exist_ok=True)
    n_files = 0
    for home in homes:
        for src in files_to_copy(home):
            copy_file(src, home, out / home.name)
            n_files += 1
    summary = out / "summary.md"
    summary.write_text(build_summary(homes), encoding="utf-8")
    return {"files": n_files, "summary": str(summary)}
