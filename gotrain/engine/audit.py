"""Data Auditor: automatic checks that a GEO-Bench task's data matches its own metadata.

    audit_dataset(task, apply_fixes=True) -> {"task", "apply_fixes", ..., "findings": [...]}

Each finding is {check, severity, evidence, suggestion}:
    severity "error"   the data contradicts the metadata (a band or class name is wrong)
             "warning" something a model could exploit or that inflates scores
             "info"    a useful fact, or a check that was skipped and why
    evidence numbers measured on the data (correlations, NDVI, percentages, counts)

apply_fixes=False audits GEO-Bench's RAW metadata (ignores BAND_ORDER_FIXES and
CLASS_NAME_FIXES). On raw m-eurosat this re-discovers both upstream bugs from the pixels alone;
with the fixes applied those findings disappear (tests/test_engine_audit.py).

Checks (all data-driven; names are only used to know what to expect):
1. bands      Sentinel-2 relations: the NIR family (B06/B07/B08/B8A) moves together, the
              SWIR pair B11/B12 too, B10 (cirrus) is near zero over land, the visible bands
              are strongly correlated, NIR > red on most pixels, and 8-bit channels (0-255)
              mixed with reflectance. Aerial RGB(N): visible bands correlated, NIR > red.
2. classes    per-class mean NDVI / NDWI (pixel-level for segmentation) against what the class
              name implies (forest is green, water isn't, buildings aren't dense vegetation).
              Skipped without a NIR band.
3. shortcuts  majority-class baseline; per-class share of flat tiles and of tiles with nodata.
4. leaks      exact duplicate images within train and between val and train.
5. nodata     classes whose input pixels are (almost) all zero.

CPU only. Reads TRAIN and VAL only, never the test split. Writes nothing.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any

import numpy as np

from gotrain.engine import data

S2_RE = re.compile(r"^b?0?(\d{1,2})(a?)\b")
ROLE_WORDS = {
    "red": "red",
    "green": "green",
    "blue": "blue",
    "nir": "nir",
    "nearinfrared": "nir",
    "near-infrared": "nir",
}
S2_ROLES = {"B02": "blue", "B03": "green", "B04": "red", "B08": "nir"}
NIR_FAMILY = ("B06", "B07", "B08", "B8A")
NEAR_EMPTY = 0.05  # p99 below 5% of the brightest band's p99
WATER_WORDS = ("water", "sea", "lake", "river", "ocean", "pond", "reservoir")
FOREST_WORDS = ("forest", "tree", "canopy", "wood")
VEG_WORDS = ("vegetation", "pasture", "grass", "meadow", "crop", "herbaceous", "shrub")
BUILT_WORDS = ("building", "residential", "industrial", "urban", "impervious", "road", "highway")


def finding(check: str, severity: str, evidence: dict, suggestion: str) -> dict:
    return {"check": check, "severity": severity, "evidence": evidence, "suggestion": suggestion}


# ---------------------------------------------------------------------------------------------
# Names -> what to expect
# ---------------------------------------------------------------------------------------------
def s2_id(name: str) -> str | None:
    """'08A - Vegetation Red Edge' / 'B8A' -> 'B8A'; '10 - SWIR - Cirrus' -> 'B10'."""
    m = S2_RE.match(name.strip().lower())
    if not m:
        return None
    num, a = int(m.group(1)), m.group(2)
    if not 1 <= num <= 12 or (a and num != 8):
        return None
    return "B8A" if a else f"B{num:02d}"


def band_roles(names: list[str]) -> dict[str, int]:
    """Channel index for red/green/blue/nir found by name (Sentinel-2 ids or plain words)."""
    roles: dict[str, int] = {}
    for i, n in enumerate(names):
        role = S2_ROLES.get(s2_id(n) or "") or ROLE_WORDS.get(n.strip().lower())
        if role and role not in roles:
            roles[role] = i
    return roles


def name_kind(name: str) -> str | None:
    low = name.lower()
    if "no data" in low or "nodata" in low:
        return None
    for kind, words in (
        ("water", WATER_WORDS),
        ("forest", FOREST_WORDS),
        ("vegetation", VEG_WORDS),
        ("built", BUILT_WORDS),
    ):
        if any(w in low for w in words):
            return kind
    return None


# ---------------------------------------------------------------------------------------------
# 1. Band sanity
# ---------------------------------------------------------------------------------------------
def check_bands(pixels: np.ndarray, names: list[str]) -> list[dict]:
    """pixels: [N, C] raw values (nodata removed). Relational checks only."""
    out: list[dict] = []
    ids = [s2_id(n) for n in names]
    with np.errstate(invalid="ignore"):
        corr = np.nan_to_num(np.corrcoef(pixels.T))
    p99 = np.percentile(pixels, 99, axis=0)
    top = float(p99.max()) if len(p99) else 0.0
    idx = {b: i for i, b in enumerate(ids) if b}

    def label(i: int) -> str:
        return f"'{names[i]}'"

    if len(idx) >= 4:  # Sentinel-2-style channels
        if "B08" in idx and "B8A" in idx:
            i08, i8a = idx["B08"], idx["B8A"]
            r = float(corr[i08, i8a])
            if r < 0.9:
                others = [j for j in range(len(names)) if ids[j] not in NIR_FAMILY]
                best = max(others, key=lambda j: corr[i08, j]) if others else None
                hint = ""
                if best is not None and corr[i08, best] >= 0.9:
                    hint = (
                        f" Channel {label(best)} correlates r={corr[i08, best]:.2f} with B08: "
                        "it behaves like B8A."
                    )
                out.append(
                    finding(
                        "bands",
                        "error",
                        {"channel": names[i8a], "r_with_B08": round(r, 3), "expected": ">= 0.9"},
                        f"The channel named {label(i8a)} doesn't move with B08 (NIR) as B8A "
                        f"should.{hint} The band names are probably out of order.",
                    )
                )
        if "B11" in idx and "B12" in idx:
            r = float(corr[idx["B11"], idx["B12"]])
            if r < 0.8:
                out.append(
                    finding(
                        "bands",
                        "error",
                        {
                            "channels": [names[idx["B11"]], names[idx["B12"]]],
                            "r": round(r, 3),
                            "expected": ">= 0.8",
                        },
                        "The two channels named SWIR B11/B12 don't move together as SWIR "
                        "bands do. The band names are probably out of order.",
                    )
                )
        if "B10" in idx and top > 0:
            i10 = idx["B10"]
            empty = [j for j in range(len(names)) if p99[j] < NEAR_EMPTY * top]
            if i10 not in empty:
                hint = ""
                if empty:
                    hint = (
                        " Near-zero channel(s) "
                        + ", ".join(label(j) for j in empty)
                        + " look like B10 instead."
                    )
                out.append(
                    finding(
                        "bands",
                        "error",
                        {
                            "channel": names[i10],
                            "p99": round(float(p99[i10]), 1),
                            "brightest_p99": round(top, 1),
                        },
                        f"The channel named {label(i10)} should be B10 (cirrus), which is near "
                        f"zero over land, but it isn't.{hint}",
                    )
                )
    roles = band_roles(names)
    vis = [roles[c] for c in ("blue", "green", "red") if c in roles]
    weak = [
        (names[a], names[b], round(float(corr[a, b]), 3))
        for k, a in enumerate(vis)
        for b in vis[k + 1 :]
        if corr[a, b] < 0.7
    ]
    if weak:
        out.append(
            finding(
                "bands",
                "error",
                {"weak_pairs": weak, "expected": "r >= 0.7"},
                "Visible bands (blue/green/red) are normally strongly correlated; these aren't. "
                "Check that these channels really are visible light.",
            )
        )
    if "nir" in roles and "red" in roles:
        frac = float(np.mean(pixels[:, roles["nir"]] > pixels[:, roles["red"]]))
        if frac < 0.5:
            out.append(
                finding(
                    "bands",
                    "warning",
                    {
                        "nir": names[roles["nir"]],
                        "red": names[roles["red"]],
                        "share_nir_gt_red": round(frac, 3),
                        "expected": ">= 0.5 over land",
                    },
                    "NIR should exceed red on most land pixels (plants reflect NIR). Either the "
                    "scene is mostly water/urban or these two channels are swapped.",
                )
            )
    eight_bit = [
        j
        for j in range(len(names))
        if pixels[:, j].min() >= 0
        and pixels[:, j].max() <= 255
        and p99[j] >= 100  # actually uses the 0-255 range (B10 cirrus is small, not 8-bit)
        and np.mean(pixels[:, j] == np.round(pixels[:, j])) > 0.99
    ]
    if eight_bit and len(eight_bit) < len(names) and top > 1000:
        out.append(
            finding(
                "bands",
                "info",
                {
                    "eight_bit_channels": [names[j] for j in eight_bit],
                    "their_max_p99": round(float(p99[eight_bit].max()), 1),
                    "other_max_p99": round(top, 1),
                },
                "8-bit channels (0-255, e.g. rendered true colour) sit next to reflectance "
                "bands in the thousands: per-band normalisation is required.",
            )
        )
    return out


# ---------------------------------------------------------------------------------------------
# 2. Class names vs spectra
# ---------------------------------------------------------------------------------------------
def check_class_names(ndvi: dict[str, float], ndwi: dict[str, float]) -> list[dict]:
    """Per-class mean NDVI/NDWI vs what each class name implies."""
    out = []
    names = list(ndvi)

    def candidates(kinds: tuple[str, ...], exclude: str) -> list[str]:
        return [n for n in names if n != exclude and name_kind(n) in kinds]

    for name, v in ndvi.items():
        kind = name_kind(name)
        rule = {
            "forest": (v < 0.2, "forest/tree classes are dense vegetation (NDVI > 0.2)"),
            "vegetation": (v < 0.0, "vegetation classes have positive NDVI"),
            "water": (v > 0.5, "water has low NDVI; this class is strongly vegetated"),
            "built": (v > 0.6, "built-up classes are not dense vegetation (NDVI < 0.6)"),
        }.get(kind)
        if not rule or not rule[0]:
            continue
        if v < 0:
            likely = candidates(("water",), name)
        elif v > 0.6:
            likely = candidates(("forest",), name) or candidates(("vegetation",), name)
        else:
            likely = candidates(("built", "vegetation"), name)
        out.append(
            finding(
                "classes",
                "error",
                {
                    "class": name,
                    "kind_from_name": kind,
                    "mean_ndvi": round(v, 3),
                    "mean_ndwi": round(ndwi.get(name, float("nan")), 3),
                },
                f"'{name}' contradicts its spectra: {rule[1]}."
                + (f" The data look like: {', '.join(repr(c) for c in likely)}." if likely else ""),
            )
        )
    if len(out) >= 2:
        out.append(
            finding(
                "classes",
                "error",
                {"contradicting_classes": len(out)},
                "Several class names contradict their data: the class-name list is probably "
                "out of order relative to the label integers. Check a few tiles per label.",
            )
        )
    return out


# ---------------------------------------------------------------------------------------------
# 3-5. Shortcuts, leaks, no-data classes
# ---------------------------------------------------------------------------------------------
def check_shortcuts(
    majority: dict,
    flat: dict[str, float],
    nodata_share: dict[str, float],
    ndvi: dict[str, float] | None = None,
) -> list[dict]:
    out = [
        finding(
            "shortcuts",
            "info",
            {
                "majority_class_baseline": majority["score"],
                "metric": majority["metric"],
                "predict": majority["predict"],
            },
            "A model must clearly beat always predicting the majority class.",
        )
    ]
    for what, shares in (("flat (low-contrast) tiles", flat), ("tiles with nodata", nodata_share)):
        if len(shares) < 2:
            continue
        hi = max(shares, key=shares.get)
        lo = min(shares, key=shares.get)
        gap = shares[hi] - shares[lo]
        if gap < 0.05:
            continue
        expected = (
            what.startswith("flat") and name_kind(hi) == "water" and (ndvi or {}).get(hi, 1.0) < 0.2
        )
        out.append(
            finding(
                "shortcuts",
                "info" if expected else "warning",
                {
                    "measure": what,
                    "highest": [hi, round(shares[hi], 4)],
                    "lowest": [lo, round(shares[lo], 4)],
                    "per_class": shares,
                },
                (
                    f"Open water is flat, so '{hi}' having most flat tiles is expected."
                    if expected
                    else f"{what.capitalize()} are much more common in '{hi}' than in "
                    f"'{lo}': a model can score by spotting them instead of the content. "
                    f"Report per-class recall of '{lo}' and look at the error grid."
                ),
            )
        )
    return out


def check_leaks(n_train: int, dup_train: int, n_val: int, val_in_train: int) -> list[dict]:
    out = []
    if dup_train:
        out.append(
            finding(
                "leaks",
                "warning",
                {"duplicates_in_train": dup_train, "train_tiles": n_train},
                "Exact duplicate train images: harmless for training but they inflate the "
                "effective weight of those samples.",
            )
        )
    if val_in_train:
        out.append(
            finding(
                "leaks",
                "warning",
                {"val_images_in_train": val_in_train, "val_tiles": n_val},
                "Validation images identical to train images: val scores are optimistic.",
            )
        )
    return out


def check_nodata_classes(zero_share: dict[str, float]) -> list[dict]:
    return [
        finding(
            "nodata",
            "warning",
            {"class": name, "share_of_its_pixels_all_zero": round(s, 4)},
            f"'{name}' marks blank (all-zero) input, not content. Keep it in the official "
            "metric if the benchmark does, but report scores without it too.",
        )
        for name, s in zero_share.items()
        if s >= 0.95
    ]


# ---------------------------------------------------------------------------------------------
# Loading (TRAIN/VAL only) and the entry point
# ---------------------------------------------------------------------------------------------
def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (a - b) / (a + b)
    return np.where(np.isfinite(r), r, np.nan)


def _hashes(ds: Any, limit: int | None) -> list[str]:
    n = len(ds) if limit is None else min(limit, len(ds))
    return [hashlib.md5(np.ascontiguousarray(ds.raw(i)[0]).tobytes()).hexdigest() for i in range(n)]


def audit_dataset(
    task: str,
    apply_fixes: bool = True,
    data_dir: str | None = None,
    n_samples: int = 300,
    leak_limit: int | None = None,
    seed: int = 0,
) -> dict:
    """Audit one task on TRAIN (+ VAL labels and hashes). See the module docstring."""
    gtask = data.load_task(task, data_dir)
    kind = data.task_type(gtask)
    k = int(gtask.label_type.n_classes)
    train = data.GeoBenchDataset(task, "train", norm="none", data_dir=data_dir)
    val = data.GeoBenchDataset(task, "val", norm="none", data_dir=data_dir)
    raw_bands = data.sample_array(train.ds[0])[1]
    raw_classes = data.raw_class_names(gtask, k)
    bands = data.channel_names(task, raw_bands)[0] if apply_fixes else list(raw_bands)
    classes = data.corrected_class_names(task, raw_classes)[0] if apply_fixes else raw_classes
    roles = band_roles(bands)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train), size=min(n_samples, len(train)), replace=False)
    pix, spreads, tile_classes, has_nodata = [], [], [], []
    v_sum, w_sum, cnt = np.zeros(k), np.zeros(k), np.zeros(k)
    zero_px, all_px = np.zeros(k), np.zeros(k)
    for i in idx:
        arr, label = train.raw(int(i))
        x = arr.reshape(-1, arr.shape[-1]).astype(np.float64)
        zero = np.all(x == 0, axis=1) | ~np.isfinite(x).all(axis=1)
        valid = x[~zero]
        if len(valid):
            pix.append(valid[rng.choice(len(valid), size=min(200, len(valid)), replace=False)])
        spreads.append(data.tile_spread(arr))
        present = data.tile_classes(label, kind, k)
        tile_classes.append(present)
        has_nodata.append(bool(zero.any()))
        if kind == "segmentation":
            lab = np.asarray(label).reshape(-1).astype(np.int64)
        else:
            lab = np.full(
                len(x), int(np.asarray(label).reshape(-1)[0]) if kind != "multilabel" else -1
            )
        ok = (lab >= 0) & (lab < k)
        np.add.at(zero_px, lab[ok], zero[ok])
        np.add.at(all_px, lab[ok], 1)
        if "red" in roles and "nir" in roles and kind != "multilabel":
            ndvi = _ratio(x[:, roles["nir"]], x[:, roles["red"]])
            ndwi = (
                _ratio(x[:, roles["green"]], x[:, roles["nir"]])
                if "green" in roles
                else np.full(len(x), np.nan)
            )
            use = ok & ~zero & np.isfinite(ndvi)
            if kind == "segmentation":
                np.add.at(v_sum, lab[use], ndvi[use])
                np.add.at(w_sum, lab[use], np.nan_to_num(ndwi[use]))
                np.add.at(cnt, lab[use], 1)
            elif use.any():  # classification: per-tile mean, averaged per class
                c = int(lab[0])
                v_sum[c] += float(np.nanmean(ndvi[use]))
                w_sum[c] += float(np.nanmean(np.nan_to_num(ndwi[use])))
                cnt[c] += 1
    pixels = np.concatenate(pix) if pix else np.zeros((0, len(bands)))

    findings: list[dict] = []
    findings += check_bands(pixels, bands) if len(pixels) > 10 else []
    ndvi_by_class: dict[str, float] = {}
    if "red" in roles and "nir" in roles and kind != "multilabel":
        ndvi_by_class = {classes[c]: float(v_sum[c] / cnt[c]) for c in range(k) if cnt[c]}
        ndwi_by_class = {classes[c]: float(w_sum[c] / cnt[c]) for c in range(k) if cnt[c]}
        findings += check_class_names(ndvi_by_class, ndwi_by_class)
    else:
        findings.append(
            finding(
                "classes",
                "info",
                {"bands": bands},
                "No NIR band (or multilabel task): the NDVI/NDWI class-name check was skipped.",
            )
        )

    std = pixels.std(axis=0) if len(pixels) else np.ones(len(bands))
    flat = data.flat_tile_fraction(np.asarray(spreads), std, tile_classes, classes)["per_class"]
    flat = {c: v for c, v in flat.items() if v is not None}
    nod = {
        classes[c]: round(
            float(np.mean([h for h, t in zip(has_nodata, tile_classes, strict=True) if c in t])), 4
        )
        for c in range(k)
        if any(c in t for t in tile_classes)
    }
    train_counts = Counter(data._label_counts(train.ds, kind))
    train_counts["_n"] = len(train)
    majority = data.majority_class_baseline(
        kind, train_counts, data._label_counts(val.ds, kind), len(val), classes
    )
    findings += check_shortcuts(majority, flat, nod, ndvi_by_class)

    th, vh = _hashes(train, leak_limit), _hashes(val, leak_limit)
    dup_train = len(th) - len(set(th))
    in_train = sum(h in set(th) for h in vh)
    findings += check_leaks(len(th), dup_train, len(vh), in_train)
    zero_share = {classes[c]: float(zero_px[c] / all_px[c]) for c in range(k) if all_px[c]}
    findings += check_nodata_classes(zero_share)

    order = {"error": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda f: order[f["severity"]])
    return {
        "task": task,
        "apply_fixes": apply_fixes,
        "task_type": kind,
        "bands": bands,
        "class_names": classes,
        "n_sampled_train_tiles": int(len(idx)),
        "n_hashed": {"train": len(th), "val": len(vh)},
        "ndvi_by_class": {c: round(v, 4) for c, v in ndvi_by_class.items()},
        "findings": findings,
        "summary": dict(Counter(f["severity"] for f in findings)),
    }
