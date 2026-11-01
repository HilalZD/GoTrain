"""Confidence-flags report for a calibrated run: a short markdown file and one 1600x900 PNG.

    python scripts/confidence_report.py <run_dir>            # calibrates first if needed
    python scripts/confidence_report.py <run_dir> --out docs/figures/confidence

Left panel: reliability diagram before/after temperature scaling. Right panel: accuracy of the
most-confident predictions vs coverage, with the 99% target and the chosen threshold. All numbers
are measured on VALIDATION, where the temperature and the threshold were fitted. CPU only.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BEFORE, AFTER, SURFACE = "#b9b8b1", "#2a78d6", "#fcfcfb"


def write_markdown(c: dict, path: Path) -> None:
    th = c["threshold"]
    unit = c.get("unit", "images")
    lines = [
        f"# Confidence flags: {c['task']}",
        "",
        f"> {c['note']}",
        "",
        f"- Temperature T = **{c['temperature']}** (fitted on {c['n_val']} validation "
        f"{'tiles' if c['task_type'] == 'segmentation' else 'images'}; unit: {unit})",
        f"- ECE (15 bins): {c['ece_before']} -> **{c['ece_after']}**;  "
        f"NLL: {c['nll_before']} -> **{c['nll_after']}**",
        f"- Accuracy of all predictions: {c['accuracy']}",
        f"- Threshold: confidence >= {th['confidence']:.4f} keeps **{th['coverage']:.1%}** of the "
        f"predictions at accuracy {th['confident_accuracy']} (target: <= {th['max_error']:.0%} "
        f"error); **{th['flagged_pct']:.1f}%** are flagged needs_review.",
    ]
    if "tiles" in c:
        t = c["tiles"]
        lines.append(
            f"- Tiles: {t['flagged_pct']:.1f}% flagged needs_review ({t['rule']}); mean tile mIoU "
            f"flagged {t['mean_tile_miou_flagged']} vs unflagged {t['mean_tile_miou_unflagged']}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(length=0, colors=MUTED, labelsize=12)
    ax.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)


def write_png(c: dict, path: Path) -> None:
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
    th = c["threshold"]
    fig.text(
        0.05, 0.93, f"Confidence flags · {c['task']}", fontsize=26, fontweight="bold", va="center"
    )
    fig.text(
        0.05,
        0.875,
        "Measured on VALIDATION (where T and the threshold were fitted)",
        fontsize=15,
        color=INK2,
        va="center",
    )

    ax1 = fig.add_axes([0.06, 0.12, 0.36, 0.64])
    ax1.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    for key, color, name in (
        ("before", BEFORE, "before"),
        ("after", AFTER, f"T = {c['temperature']:.2f}"),
    ):
        rel = c["reliability"][key]
        pts = [
            (x, y) for x, y in zip(rel["bin_conf"], rel["bin_acc"], strict=True) if x is not None
        ]
        xs, ys = zip(*pts, strict=True)
        ax1.plot(xs, ys, color=color, lw=2.5, marker="o", ms=8, mec=SURFACE, mew=2, label=name)
    ax1.legend(loc="lower right", frameon=False, fontsize=13, labelcolor=INK2)
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 1.02)
    ax1.set_xlabel("confidence", fontsize=13, color=INK2)
    ax1.set_ylabel("accuracy", fontsize=13, color=INK2)
    ax1.set_title(
        f"Reliability · ECE {c['ece_before']:.3f} → {c['ece_after']:.3f}",
        fontsize=16,
        loc="left",
        pad=12,
    )
    _style(ax1)

    ax2 = fig.add_axes([0.55, 0.12, 0.40, 0.64])
    cov, acc = c["curve"]["coverage"], c["curve"]["accuracy"]
    ax2.plot(cov, acc, color=AFTER, lw=2.5)
    target = 1 - th["max_error"]
    ax2.axhline(target, color=MUTED, lw=1, ls="--")
    ax2.text(
        0.98, target + 0.004, f"{target:.0%} accuracy target", fontsize=12, color=INK2, ha="right"
    )
    if th["coverage"] > 0:
        ax2.plot(
            [th["coverage"]],
            [th["confident_accuracy"]],
            marker="o",
            ms=11,
            color=AFTER,
            mec=SURFACE,
            mew=2,
        )
        ax2.annotate(
            f"threshold {th['confidence']:.3f}\nkeeps {th['coverage']:.1%}, flags "
            f"{th['flagged_pct']:.1f}%",
            (th["coverage"], th["confident_accuracy"]),
            xytext=(10, -300),
            textcoords="offset points",
            fontsize=13,
            color=INK,
            arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 1},
        )
    low = min(min(acc), target) - 0.02
    ax2.set_ylim(max(0.0, low), 1.005)
    ax2.set_xlim(0, 1.0)
    ax2.set_xlabel(
        f"coverage (share of {c.get('unit', 'images').split(' ')[0]} kept)", fontsize=13, color=INK2
    )
    ax2.set_ylabel("accuracy of the kept predictions", fontsize=13, color=INK2)
    ax2.set_title("Coverage vs accuracy (most confident first)", fontsize=16, loc="left", pad=12)
    _style(ax2)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="default: the run dir")
    args = ap.parse_args()
    cal_path = args.run_dir / "calibration.json"
    if not cal_path.exists():
        import torch

        from gotrain.engine.calibration import calibrate

        task = torch.load(
            args.run_dir / "checkpoints" / "best.ckpt", map_location="cpu", weights_only=True
        )["task"]
        calibrate(task, args.run_dir)
    c = json.loads(cal_path.read_text(encoding="utf-8"))
    out = args.out or args.run_dir
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{c['task']}_confidence"
    write_markdown(c, out / f"{stem}.md")
    write_png(c, out / f"{stem}.png")
    th = c["threshold"]
    print(f"{c['task']}: T={c['temperature']} coverage={th['coverage']} -> {out / stem}.md/.png")


if __name__ == "__main__":
    main()
