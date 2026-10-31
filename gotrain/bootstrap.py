"""95% bootstrap confidence interval of a test score, from the engine's per_sample results.

Images are resampled with replacement and the metric is recomputed from the resampled
per-image counts (the same formulas as the engine's score), with a fixed seed.
"""

import numpy as np

N_RESAMPLES = 1000
SEED = 0
LEVEL = 0.95


def _metric(task_type: str, arrays: dict[str, np.ndarray], idx: np.ndarray) -> float:
    if task_type == "segmentation":
        inter = arrays["intersection"][idx].sum(axis=0)
        union = arrays["union"][idx].sum(axis=0)
        present = union > 0
        return float(np.mean(inter[present] / union[present])) if present.any() else 0.0
    if task_type == "classification":
        return float(arrays["correct"][idx].mean())
    tp, fp, fn = (float(arrays[k][idx].sum()) for k in ("tp", "fp", "fn"))
    return 2 * tp / max(2 * tp + fp + fn, 1.0)


def bootstrap_ci(task_type: str, per_sample: dict) -> dict | None:
    """{"ci_low", "ci_high", "ci_level", "ci_resamples"}, or None if per_sample is unusable."""
    try:
        arrays = {k: np.asarray(v, dtype=np.float64) for k, v in per_sample.items()}
        n = len(next(iter(arrays.values())))
        if n == 0:
            return None
        rng = np.random.default_rng(SEED)
        scores = [_metric(task_type, arrays, rng.integers(0, n, n)) for _ in range(N_RESAMPLES)]
    except (KeyError, StopIteration, ValueError, TypeError):
        return None
    low, high = np.percentile(scores, [(1 - LEVEL) / 2 * 100, (1 + LEVEL) / 2 * 100])
    return {
        "ci_low": round(float(low), 4),
        "ci_high": round(float(high), 4),
        "ci_level": LEVEL,
        "ci_resamples": N_RESAMPLES,
    }
