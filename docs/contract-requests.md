# Contract requests (Serena → Hilal)

Things the ML side needs from files Hilal owns. Nothing here is changed until we both agree.
Each entry: what, why, and the smallest change that would do it.

## 1. FYI: engine tests live in new files under `tests/` (no action needed)

- **What:** Serena's tests are new files only: `tests/test_engine_data.py` (M1), and later
  `tests/test_engine_train.py`, `tests/test_engine_diagnostics.py`. Existing test files are
  never edited.
- **Why:** `make test` collects `tests/` only; keeping engine tests there means one command
  runs everything. Agreed with Serena on 2026-10-31.
- **Data-dependent tests** skip themselves when the GEO-Bench files are not on disk, so the
  suite stays green on machines without data.

## 2. Meaning of `status="timeout"` under time budgets (please confirm)

- **What the engine does now:** `"ok"` = at least one full epoch was trained and validated,
  even when the run stopped because its budget ran out. `"timeout"` = the budget ran out inside
  the first epoch; a checkpoint is returned if one was saved, otherwise `checkpoint_path=None`.
- **Why:** the LR schedule is tied to the budget (cosine over the wall-clock time), so stopping
  at the budget is the *normal, planned* end of a trial, not a failure. `rank()`, `promote`,
  `diagnose` and `register_model` only accept `"ok"` runs, and `FakeEngine` also returns
  `"ok"` for budget-limited runs. Calling those runs `"timeout"` would make every trial
  un-promotable. `metrics.json` records `stop_reason` (`"budget"` or `"max_epochs"`) so the
  difference is still visible.
- **Change needed:** none if you agree. If you'd rather have `"timeout"` for every
  budget-stopped run, the tools' status gates would need to accept it too.

## 3. Checkpoint location (FYI)

- The real engine writes `run_dir/checkpoints/best.ckpt`, the same layout as `FakeEngine` and
  the "State on disk" section of CLAUDE.md, and returns that path in `TrainResult`. It also
  writes `history.json`, `metrics.json`, `val_predictions.npz` and, only if missing,
  `config.json` (the tools write that one first, so it is never overwritten).

## 4. combos.py vs torchgeo 0.10.1: they agree (no action needed)

- `VALID_COMBOS["ssl4eo_s2"] == ("resnet18", "resnet50")` matches TorchGeo exactly:
  `ResNet18_Weights.SENTINEL2_ALL_MOCO` and `ResNet50_Weights.SENTINEL2_ALL_MOCO`, both 13
  channels in the order B1..B8, B8a, B9..B12, pretrained on input / 10000.
  `tests/test_engine_train.py::test_combos_ssl4eo_list_matches_torchgeo` checks this
  against the installed torchgeo, so a future mismatch fails loudly.
- Engine detail: for segmentation, smp only reaches `convnext_tiny` through its timm wrapper,
  so the engine builds the U-Net encoder as `tu-convnext_tiny`. Same name for the agent.
- The engine now validates backbone + weights with `combos.check_combo` and keeps no list
  of its own. `describe_dataset()` sets `sensor="sentinel-2"` for every task with Sentinel-2
  bands (including m-brick-kiln, whose 8-bit TCI channels would otherwise make it
  "sentinel-2+aerial-rgb" and fail `is_sentinel2`).

## 5. The `_dev` home check vs Serena's SageMaker folders (please confirm)

- `config.py` refuses `GOTRAIN_ENGINE=fake` unless the GOTRAIN_HOME folder name ends in
  `_dev`. Serena's agreed folders are `~/gotrain_dev_serena` and `~/gotrain_home_serena`,
  which don't, so with the default (fake) engine any command there refuses to start.
- **What Serena does now:** every command sets `GOTRAIN_ENGINE=real` inline. The hand
  baselines don't use the engine at all, and engine runs are real anyway.
- **Possible change (yours):** also accept names containing `_dev_` (e.g. `gotrain_dev_serena`)
  as dev homes, if you want fake runs to work there.

## 6. Test-time augmentation flag on evaluate() (M6, needs a small interface change)

- **What exists:** `gotrain.engine.train.evaluate(task, checkpoint_path, split, tta=False)`.
  `tta=True` averages the model's outputs over the 8 flip/rot90 views (for segmentation, each
  view's mask is rotated back before averaging). The result dict also carries `"tta"`.
- **What's missing (your files):** `Engine.evaluate` / `RealEngine.evaluate` in
  `gotrain/engine/interface.py` don't pass the flag, so the agent can't use it yet.
  Smallest change: add `tta: bool = False` to the Protocol and to `RealEngine.evaluate`,
  forwarding it; `FakeEngine.evaluate` can accept and ignore it.
- **Rule to keep:** choose TTA on VALIDATION only (compare `evaluate(..., "val", tta=True)`
  with `tta=False`), then pass the same choice to `final_test`. Never decide it on test.
