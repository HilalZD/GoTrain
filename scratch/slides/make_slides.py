"""Three 1920x1080 slide figures. CPU only, TRAIN tiles only; scores are VAL numbers."""

import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).parent))
import slidekit as sk  # noqa: E402

OUT = Path.home() / "gotrain_dev_serena" / "slides"
OUT.mkdir(parents=True, exist_ok=True)
W, H = 1920, 1080


def ax_px(fig, x, y, w, h):
    """Axes from pixel box (x, y = top-left, measured from the top)."""
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])


def show_tile(fig, task, i, x, y, size, expect=None, overlay=None):
    img, lab = sk.tile(task, i)
    if expect is not None:
        assert int(lab) == expect, f"{task} tile {i}: label {int(lab)} != expected {expect}"
    ax = ax_px(fig, x, y, size, size)
    ax.imshow(img, interpolation="nearest")
    if overlay is not None:
        overlay(ax, np.asarray(lab).squeeze())
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(sk.GRID)
    return ax


def swatch(fig, x, y, color, text, size=18, box=18):
    fig.patches.append(Rectangle((x / W, 1 - (y + box) / H), box / W, box / H,
                                 transform=fig.transFigure, color=color, ec="none"))
    fig.text((x + box + 10) / W, 1 - (y + box / 2) / H, text, fontsize=size, color=sk.INK2,
             va="center")


def pv_overlay(ax, m):
    ax.imshow(np.ma.masked_where(m != 1, m), cmap=ListedColormap([sk.PANEL_COLOR]),
              alpha=0.55, interpolation="nearest")


CHESA_NAMES = list(sk.CHESA_COLORS)


def chesa_overlay(ax, m):
    cmap = ListedColormap([sk.CHESA_COLORS[n] for n in CHESA_NAMES])
    ax.imshow(m, cmap=cmap, vmin=0, vmax=6, alpha=0.5, interpolation="nearest")


# ------------------------------------------------------------------------------------------
def tasks_overview():
    fig = sk.slide("GoTrain's 4 tasks",
                   "Sample TRAIN tiles · true-colour RGB, 2–98% stretch · corrected band and class names")
    tasks = [
        dict(task="m-eurosat", tiles=[(1217, 8, "River"), (1595, 7, "Residential\nBuildings")],
             info=["Sentinel-2, 13 bands", "64×64 px · 10 m/pixel", "Classification, 10 classes",
                   "2,000 train tiles", "Metric: accuracy"]),
        dict(task="m-brick-kiln", tiles=[(12272, 1, "brick kiln"), (1133, 0, "not brick\nkiln")],
             info=["Sentinel-2, 10 bands + RGB", "64×64 px · 10 m/pixel", "Classification, 2 classes",
                   "15,063 train tiles", "Metric: accuracy"]),
        dict(task="m-pv4ger-seg", tiles=[(1081, None, None), (1402, None, None)], overlay=pv_overlay,
             legend=[(sk.PANEL_COLOR, "solar panel")],
             info=["Aerial RGB", "320×320 px · 0.1 m/pixel", "Segmentation, 2 classes",
                   "3,000 train tiles", "Metric: mIoU"]),
        dict(task="m-chesapeake", tiles=[(922, None, None), (225, None, None)], overlay=chesa_overlay,
             legend=[(sk.CHESA_COLORS[n], n) for n in CHESA_NAMES[1:]],
             info=["Aerial RGB + near-infrared", "256×256 px · 1 m/pixel", "Segmentation, 7 classes",
                   "3,000 train tiles", "Metric: mIoU"]),
    ]
    col_x, tile = [60, 495, 930, 1365], 255
    for k, t in enumerate(tasks):
        x = col_x[k]
        fig.text(x / W, 1 - 205 / H, t["task"], fontsize=28, fontweight="bold", va="center")
        for r, (i, exp, cap) in enumerate(t["tiles"]):
            y = 240 + r * (tile + 22)
            show_tile(fig, t["task"], i, x, y, tile, expect=exp, overlay=t.get("overlay"))
            if cap:
                fig.text((x + tile + 16) / W, 1 - (y + 16) / H, "label", fontsize=16,
                         color=sk.MUTED, va="top")
                fig.text((x + tile + 16) / W, 1 - (y + 42) / H, cap, fontsize=19,
                         color=sk.INK, va="top", linespacing=1.2)
        if t.get("legend"):
            for j, (c, name) in enumerate(t["legend"]):
                swatch(fig, x + tile + 16, 244 + j * 34, c, name, size=16, box=18)
        y_info = 240 + 2 * tile + 22 + 40
        for j, line in enumerate(t["info"]):
            fig.text(x / W, 1 - (y_info + j * 38) / H, line, fontsize=21, color=sk.INK,
                     fontweight="bold" if j == 4 else "normal", va="top")
    fig.savefig(OUT / "tasks_overview.png", dpi=100)
    sk.plt.close(fig)


# ------------------------------------------------------------------------------------------
def brickkiln_shortcut():
    fig = sk.slide("A lazy model could cheat; ours finds kilns",
                   "m-brick-kiln · tiles from TRAIN · scores on the 999 VALIDATION tiles")
    tile, gap, x0 = 250, 24, 60
    rows = [("Flat “not kiln” tiles: water, haze or cloud.\nThe stretch shows only noise.",
             [5682, 5664, 8545], 0, sk.BLUE),
            ("Real brick kilns: pale yards\nwith small orange spots",
             [12272, 7625, 12986], 1, sk.ORANGE)]
    for r, (text, idx, lab, color) in enumerate(rows):
        y = 200 + r * (tile + 120)
        fig.patches.append(Rectangle((x0 / W, 1 - (y + 52) / H), 8 / W, 52 / H,
                                     transform=fig.transFigure, color=color, ec="none"))
        fig.text((x0 + 22) / W, 1 - y / H, text, fontsize=21, va="top", linespacing=1.25)
        for c, i in enumerate(idx):
            show_tile(fig, "m-brick-kiln", i, x0 + c * (tile + gap), y + 70, tile, expect=lab)

    # right: two small bar panels (no shared axis between different measures)
    ax1 = ax_px(fig, 1180, 230, 660, 230)
    vals = [13.8, 0.0]
    ax1.barh([1, 0], vals, height=0.55, color=[sk.BLUE, sk.ORANGE])
    ax1.set_yticks([1, 0], ["not kiln", "brick kiln"], fontsize=21)
    ax1.set_xlim(0, 20)
    ax1.set_xticks([0, 5, 10, 15, 20], ["0%", "5%", "10%", "15%", "20%"], fontsize=17)
    for yv, v in zip([1, 0], vals, strict=True):
        ax1.text(v + 0.4, yv, f"{v:.1f}%", va="center", fontsize=22, fontweight="bold")
    ax1.set_title("Share of TRAIN tiles that are flat", fontsize=23, loc="left", pad=14)
    _clean(ax1)

    ax2 = ax_px(fig, 1180, 610, 660, 270)
    groups = ["Accuracy", "Kiln F1"]
    lazy, ours = [0.664, 0.0], [0.989, 0.984]
    ypos = np.array([1.0, 0.0])
    ax2.barh(ypos + 0.18, lazy, height=0.32, color=sk.NEUTRAL, label="Always “not kiln”")
    ax2.barh(ypos - 0.18, ours, height=0.32, color=sk.VIOLET, label="Our 5-min smoke model")
    for yv, a, b in zip(ypos, lazy, ours, strict=True):
        lazy_txt = f"{a:.3f}" if a > 0 else "0.000  (never predicts a kiln)"
        ax2.text(a + 0.015, yv + 0.18, lazy_txt, va="center", fontsize=20)
        ax2.text(b + 0.015, yv - 0.18, f"{b:.3f}", va="center", fontsize=20, fontweight="bold")
    ax2.set_yticks(ypos, groups, fontsize=21)
    ax2.set_xlim(0, 1.15)
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"], fontsize=17)
    ax2.set_title("VALIDATION score", fontsize=23, loc="left", pad=14)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=18,
               handlelength=1.0)
    _clean(ax2)

    fig.text(x0 / W, 0.06,
             "Flat = median band contrast (p98 − p2) below 0.25 × the band’s train std "
             "(2,000 sampled train tiles).", fontsize=16, color=sk.INK2)
    fig.text(x0 / W, 0.025,
             "Smoke model: ResNet50 with SSL4EO weights, 64 px, 5-minute budget, not tuned.",
             fontsize=16, color=sk.INK2)
    fig.savefig(OUT / "brickkiln_shortcut.png", dpi=100)
    sk.plt.close(fig)


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(sk.MUTED)
    ax.spines["bottom"].set_color(sk.GRID)
    ax.tick_params(length=0)
    ax.xaxis.grid(True, color=sk.GRID, lw=1)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------------------------------
def eurosat_label_bug():
    fig = sk.slide("We found a label bug in GEO-Bench",
                   "m-eurosat: geobench’s class-name list doesn’t match the label numbers")
    cards = [  # (tile, label, geobench name, true name, mean NDVI, reading)
        (1892, 9, "Forest", "Sea and Lake", -0.25, "below 0: open water"),
        (55, 5, "Sea and Lake", "Pasture", 0.65, "high: green grass, not water"),
        (217, 1, "Residential Buildings", "Forest", 0.71, "highest of all 10 labels:\ndense vegetation"),
        (1398, 4, "River", "Industrial Buildings", 0.22, "lowest land value:\nroofs and concrete"),
    ]
    card_w, x0, tile = 455, 60, 300
    for k, (i, lab, wrong, right, ndvi, reading) in enumerate(cards):
        x = x0 + k * card_w
        fig.text(x / W, 1 - 182 / H, f"label {lab}", fontsize=18, color=sk.MUTED, va="center")
        show_tile(fig, "m-eurosat", i, x, 205, tile, expect=lab)
        y = 535
        fig.text(x / W, 1 - y / H, "✗", fontsize=24, color=sk.RED, va="center", fontweight="bold")
        fig.text((x + 32) / W, 1 - y / H, "geobench says", fontsize=17, color=sk.MUTED,
                 va="center")
        fig.text((x + 32) / W, 1 - (y + 32) / H, wrong, fontsize=21, color=sk.INK2, va="center")
        fig.text(x / W, 1 - (y + 80) / H, "✓", fontsize=24, color=sk.GREEN, va="center",
                 fontweight="bold")
        fig.text((x + 32) / W, 1 - (y + 80) / H, "actually", fontsize=17, color=sk.MUTED,
                 va="center")
        fig.text((x + 32) / W, 1 - (y + 112) / H, right, fontsize=23, fontweight="bold",
                 va="center")
        _ndvi_gauge(fig, x, 715, 360, ndvi)
        fig.text(x / W, 1 - 832 / H, reading, fontsize=19, color=sk.INK2, va="top",
                 linespacing=1.25)
    fig.text(x0 / W, 0.105,
             "Mean NDVI over all 200 TRAIN tiles of that label.  NDVI = (near-infrared − red) / "
             "(near-infrared + red): plants score high, water below 0.", fontsize=18, color=sk.INK2)
    fig.text(x0 / W, 0.065,
             "Label numbers follow the original alphabetical EuroSAT order, but all 10 names in "
             "geobench’s list are in a different order.", fontsize=18, color=sk.INK2)
    fig.text(x0 / W, 0.025, "GoTrain corrects the names in its data loader.", fontsize=18,
             color=sk.INK2)
    fig.savefig(OUT / "eurosat_label_bug.png", dpi=100)
    sk.plt.close(fig)


def _ndvi_gauge(fig, x, y, w, v, lo=-0.4, hi=0.8):
    ax = ax_px(fig, x, y, w, 90)
    ax.set_xlim(lo, hi), ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((lo, 0.42), hi - lo, 0.16, boxstyle="round,pad=0,rounding_size=0.08",
                                color=sk.GRID, ec="none", mutation_aspect=0.3))
    ax.plot([0, 0], [0.3, 0.7], color=sk.MUTED, lw=2)
    def fmt(t, d):
        return f"{t:+.{d}f}".replace("-", "−")

    ax.text(0, 0.08, "0", ha="center", fontsize=16, color=sk.MUTED)
    ax.text(lo, 0.08, fmt(lo, 1), ha="left", fontsize=16, color=sk.MUTED)
    ax.text(hi, 0.08, fmt(hi, 1), ha="right", fontsize=16, color=sk.MUTED)
    ax.scatter([v], [0.5], s=280, color=sk.BLUE if v < 0 else sk.GREEN, zorder=3,
               edgecolor="white", linewidth=2)
    ax.text(lo, 0.92, f"mean NDVI  {fmt(v, 2)}", ha="left", fontsize=21, fontweight="bold")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3"]
    if "1" in which:
        tasks_overview()
    if "2" in which:
        brickkiln_shortcut()
    if "3" in which:
        eurosat_label_bug()
    print("saved to", OUT)
