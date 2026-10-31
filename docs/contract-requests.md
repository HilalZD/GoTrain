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
