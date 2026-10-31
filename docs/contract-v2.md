# Engine contract v2 (for the real engine)

The agent side calls the engine only through `gotrain/engine/interface.py`. The real engine is
`gotrain/engine/train.py` (`train`, `evaluate`, `diagnose`, `predict`) and
`gotrain/engine/data.py` (`describe_dataset`). `gotrain/engine/fake.py` implements all of
this and is the reference behaviour; `tests/test_engine_v2.py` is the executable spec.

**Backward compatible:** v1 calls still work. The adapter passes `tta` only when it is `True`,
so a v1 `evaluate(task, checkpoint_path, split)` keeps working until v2 lands.

## evaluate

```python
evaluate(task: str, checkpoint_path: Path | list[Path], split: str, tta: bool = False) -> dict
```

- `split` is `"val"` or `"test"` (anything else: `ValueError`). Only
  `gotrain/tools/final_test.py` ever asks for `"test"`.
- A **list of checkpoints is an ensemble**: run each model and **average their class
  probabilities** before taking the argmax. A one-element list is the same as a single path.
- `tta=True`: average probabilities over flips + rot90 (the 8 dihedral transforms, or at least
  identity, h-flip, v-flip, rot90), undoing the transform on segmentation outputs before
  averaging.
- Evaluate on the **full, fixed split** (no subsampling) so scores are comparable.

Returns:

| key | type | meaning |
|---|---|---|
| `task`, `split` | str | echoed |
| `metric_name` | str | `"accuracy"` (classification) or `"miou"` (segmentation) |
| `score` | float | the metric, **computed from `per_sample`** |
| `n_samples` | int | images evaluated |
| `per_sample` | dict | see below; used for bootstrap confidence intervals |
| `n_checkpoints` | int | 1, or the ensemble size |
| `tta` | bool | echoed |

`per_sample`, in dataset order:

- classification: `{"correct": [1, 0, 1, ...]}`, one 0/1 per image (`len == n_samples`);
  `score == mean(correct)`.
- segmentation: `{"intersection": [[...K]] * N, "union": [[...K]] * N}`, **pixel counts** per
  image and class (K = `n_classes`, class order as in the DatasetCard). `score` is the mean
  over classes with total union > 0 of `sum(intersection[:, k]) / sum(union[:, k])`.

Example (fake engine, m-eurosat, 2-model ensemble with TTA; `correct` truncated):

```json
{"task": "m-eurosat", "split": "val", "metric_name": "accuracy", "score": 0.942,
 "n_samples": 1000, "per_sample": {"correct": [1, 1, 1, 1, 1, 1, 1, 1]},
 "n_checkpoints": 2, "tta": true}
```

Example (m-pv4ger-seg, single model; first 2 images, classes `[background, solar_panel]`):

```json
{"task": "m-pv4ger-seg", "split": "val", "metric_name": "miou", "score": 0.7125,
 "n_samples": 403,
 "per_sample": {"intersection": [[88432, 2414], [85052, 3609]],
                "union": [[110235, 3719], [109399, 6740]]},
 "n_checkpoints": 1, "tta": false}
```

## predict

```python
predict(task: str, checkpoint_path: Path | list[Path], images: np.ndarray, tta: bool = False) -> dict
```

- `images`: float array **N x C x H x W**, raw values in the dataset's units, with `C` and band
  order exactly as `describe_dataset(task).bands`. **The engine normalises** (train-split band
  stats). Wrong channel count: `ValueError` naming the expected shape.
- Ensembles and `tta` behave as in `evaluate`.
- Predictions for an image must not depend on the rest of the batch.

Returns numpy arrays:

| key | classification | segmentation |
|---|---|---|
| `probabilities` | float32, N x K, rows sum to 1 | float32, N x K x H x W, sums to 1 over K |
| `labels` | int64, N (argmax) | int64, N x H x W (argmax) |
| `n_checkpoints`, `tta` | as in `evaluate` | as in `evaluate` |

Example: `predict("m-eurosat", [ckpt_a, ckpt_b], images[2 x 13 x 64 x 64])` returns
`probabilities` of shape `(2, 10)` and `labels` like `array([6, 6])`.

## Unchanged from v1

- `train(config, budget_minutes, run_dir) -> TrainResult`: scored on **val**; finish within the
  budget (LR schedule tied to it); return status `"timeout"` with 0 epochs if not even one
  epoch fits; status `"error"` with the valid list for invalid backbone/weights combos.
- `describe_dataset(task) -> DatasetCard`: train/val only; `sensor` must say Sentinel-2 for
  m-eurosat and m-brick-kiln (any spelling of "sentinel-2").
- `diagnose(task, run_dir) -> dict`: `per_class`, `confusion_matrix` (rows = true class,
  row-normalised), `curve_verdict`, optional `error_grid_path` (any size PNG/JPEG; the tool
  downscales it).
