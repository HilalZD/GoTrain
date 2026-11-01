"""Data Auditor report: a short markdown file and one slide-ready PNG for a task. CPU only.

    python scripts/audit_report.py m-eurosat --raw      # audit GEO-Bench's raw metadata
    python scripts/audit_report.py m-eurosat            # with GoTrain's errata fixes

Writes <out>/<task>_audit[_raw].md and .png (default out: docs/figures/audit). Reads TRAIN
(and VAL labels/hashes) only; see gotrain/engine/audit.py for the checks.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import textwrap  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gotrain.engine.audit import audit_dataset  # noqa: E402

# Chart tokens (dataviz reference palette): ink, muted, hairline, neutral bars, status.
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
NEUTRAL, CRITICAL, SURFACE = "#b9b8b1", "#d03b3b", "#fcfcfb"
SEVERITY_ICON = {"error": "✗", "warning": "!", "info": "i"}


def plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def write_markdown(report: dict, path: Path) -> None:
    mode = "raw GEO-Bench metadata" if not report["apply_fixes"] else "GoTrain errata applied"
    counts = ", ".join(f"{n} {s}" for s, n in sorted(report["summary"].items()))
    lines = [
        f"# Data audit: {report['task']} ({mode})",
        "",
        f"- Task type: {report['task_type']}; {report['n_sampled_train_tiles']} sampled TRAIN "
        f"tiles; duplicates checked on {report['n_hashed']['train']} train / "
        f"{report['n_hashed']['val']} val tiles. The test split is never read.",
        f"- Bands: {', '.join(report['bands'])}",
        f"- Classes: {', '.join(report['class_names'])}",
        f"- Findings: {counts}",
        "",
        "| severity | check | finding | evidence |",
        "|---|---|---|---|",
    ]
    for f in report["findings"]:
        evidence = json.dumps(f["evidence"], ensure_ascii=False).replace("|", "/")
        lines.append(
            f"| {f['severity']} | {f['check']} | {f['suggestion'].replace('|', '/')} | "
            f"`{evidence}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_png(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": INK,
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
        }
    )
    fig = plt.figure(figsize=(16, 9), dpi=100)
    mode = "raw GEO-Bench metadata" if not report["apply_fixes"] else "with GoTrain's fixes"
    fig.text(
        0.04,
        0.93,
        f"Data audit · {report['task']} · {mode}",
        fontsize=26,
        fontweight="bold",
        va="center",
    )
    counts = report["summary"]
    fig.text(
        0.04,
        0.875,
        f"{plural(counts.get('error', 0), 'error')} · "
        f"{plural(counts.get('warning', 0), 'warning')} · {plural(counts.get('info', 0), 'note')}"
        "  —  TRAIN tiles only, test split never read",
        fontsize=15,
        color=INK2,
        va="center",
    )

    ax = fig.add_axes([0.20, 0.10, 0.32, 0.70])
    flagged = {
        f["evidence"].get("class")
        for f in report["findings"]
        if f["check"] == "classes" and f["severity"] == "error"
    }
    ndvi = report["ndvi_by_class"]
    if ndvi:
        names = list(ndvi)[::-1]  # label 0 at the top
        vals = [ndvi[n] for n in names]
        colors = [CRITICAL if n in flagged else NEUTRAL for n in names]
        ax.barh(range(len(names)), vals, height=0.6, color=colors)
        ax.set_yticks(range(len(names)), names, fontsize=13, color=INK2)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_xlim(-0.6, 1.25)
        for y, (n, v) in enumerate(zip(names, vals, strict=True)):
            if n in flagged:  # selective direct labels: only the contradictions
                x = v + 0.03 if v >= 0 else 0.03
                ax.text(x, y, f"✗ {v:+.2f} contradicts name", va="center", fontsize=12, color=INK)
        ax.set_title(
            "Mean NDVI per class (plants high, water below 0)",
            fontsize=15,
            loc="left",
            color=INK,
            pad=12,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No NIR band:\nNDVI class check skipped",
            ha="center",
            va="center",
            fontsize=15,
            color=INK2,
            transform=ax.transAxes,
        )
        ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, colors=MUTED)
    ax.xaxis.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)

    y = 0.80
    fig.text(0.58, y, "Findings", fontsize=17, fontweight="bold", va="top")
    y -= 0.05
    for f in report["findings"][:7]:
        icon = SEVERITY_ICON[f["severity"]]
        color = CRITICAL if f["severity"] == "error" else INK2
        wrapped = textwrap.fill(f["suggestion"], 62)
        fig.text(
            0.58,
            y,
            f"{icon} {f['severity']} · {f['check']}",
            fontsize=12,
            color=color,
            fontweight="bold",
            va="top",
        )
        fig.text(0.60, y - 0.03, wrapped, fontsize=11, color=INK2, va="top", linespacing=1.3)
        y -= 0.05 + 0.025 * (wrapped.count("\n") + 1)
        if y < 0.08:
            break
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("task")
    ap.add_argument("--raw", action="store_true", help="ignore the errata tables")
    ap.add_argument("--out", type=Path, default=Path("docs/figures/audit"))
    ap.add_argument("--samples", type=int, default=300, help="train tiles for the pixel checks")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    report = audit_dataset(
        args.task, apply_fixes=not args.raw, data_dir=args.data_dir, n_samples=args.samples
    )
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.task}_audit{'_raw' if args.raw else ''}"
    write_markdown(report, args.out / f"{stem}.md")
    write_png(report, args.out / f"{stem}.png")
    print(f"{args.task}: {report['summary']} -> {args.out / stem}.md / .png")


if __name__ == "__main__":
    main()
