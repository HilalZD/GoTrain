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

## 6. Test-time augmentation flag on evaluate() (RESOLVED by contract v2)

- **What exists:** `gotrain.engine.train.evaluate(task, checkpoint_path, split, tta=False)`.
  `tta=True` averages the model's outputs over the 8 flip/rot90 views (for segmentation, each
  view's mask is rotated back before averaging). The result dict also carries `"tta"`.
- **What's missing (your files):** `Engine.evaluate` / `RealEngine.evaluate` in
  `gotrain/engine/interface.py` don't pass the flag, so the agent can't use it yet.
  Smallest change: add `tta: bool = False` to the Protocol and to `RealEngine.evaluate`,
  forwarding it; `FakeEngine.evaluate` can accept and ignore it.
- **Rule to keep:** choose TTA on VALIDATION only (compare `evaluate(..., "val", tta=True)`
  with `tta=False`), then pass the same choice to `final_test`. Never decide it on test.

## 7. contract v2 `predict()`: each checkpoint uses its own normalisation (approved by Serena)

- **v2 says:** "The engine normalises (train-split band stats)".
- **What the engine does:** `predict()` and `evaluate()` take raw images in the card's band
  order, and **each checkpoint applies the normalisation it was trained with**:
  - `imagenet` / `none`: the train-split z-score (exactly v2's wording);
  - `ssl4eo_s2`: /10000, bands reordered BY NAME to TorchGeo's 13-band order, missing bands
    zero-filled. That is how those weights were pretrained; z-scoring their input would
    break them.
  The caller sees no difference: raw images in, probabilities out. Ensembles may mix both
  kinds; every member gets its own inputs (and its own `img_size`).
- **Change needed:** none; perhaps reword v2 to "each checkpoint's own normalisation".

## 8. contract v2 `per_sample` for multilabel tasks (approved by Serena; not one of our 4 tasks)

- v2 defines `per_sample` for classification and segmentation only.
- **What the engine does for multilabel:** `per_sample = {"tp": [...], "fp": [...], "fn":
  [...]}`, one count per image (labels decided at probability 0.5); `score` = micro-F1 =
  2·ΣTP / (2·ΣTP + ΣFP + ΣFN), `metric_name` stays `"f1"`. `predict()` returns sigmoid
  `probabilities` [N, K] and 0/1 `labels` [N, K].
- **Change needed:** add this to contract-v2.md if multilabel tasks (e.g. m-bigearthnet) are
  ever used.


## 9. m-chesapeake: ignore class 0 "no data"? DECIDED (Serena, 2026-10-31): no, extra metric only

- **Verified:** class 0 is "no data", not land cover. GEO-Bench's converter reads the masks
  through TorchGeo's `ChesapeakeCVPRDataModule(class_set=7)`, which keeps the raw Chesapeake
  values (1 water … 6 impervious roads); TorchGeo fills out-of-raster areas with 0 and draws 0
  as transparent. Class 0 is exactly 125 whole blank train tiles (8,192,000 px = 125 × 256²).
  Names are now applied via `CLASS_NAME_FIXES` (0 no data, 1 water, …, 6 impervious roads).
- **Proposed:** map class 0 to the loss's ignore index (no training signal from blank tiles),
  never predict it (its probability set to 0), and exclude its pixels from intersection/union.
- **Why it touches the contract:** v2's formula stays the same, but the number changes: class 0's
  column in `per_sample` becomes all zeros (so v2's "classes with total union > 0" rule drops
  it), and no-data pixels no longer count in other classes' unions. The chesapeake mIoU becomes
  a 6-class mIoU, not comparable with a 7-class one or with the fake engine's.
- **Decision: no.** `score` stays the official 7-class mIoU (comparable with GEO-Bench
  baselines; no contract change) and training uses all 7 classes. Extra, clearly labelled
  number: `miou_excl_nodata` (6 classes, pixels whose truth is class 0 removed; real pixels
  predicted as class 0 still count as errors) in `metrics.json` and in `diagnose()`'s output,
  driven by `NODATA_CLASSES` in `gotrain/engine/data.py`. Nothing changes for the tools.

## 10. Task name: `m-chesapeake` on disk vs `m-chesapeake-landcover` in our docs. DECIDED: `m-chesapeake`

- GEO-Bench's Hugging Face zip is `segmentation_v1.0/m-chesapeake.zip` and it unzips into a
  folder named `m-chesapeake`; the task's own name is `m-chesapeake`. The real engine finds tasks
  by that folder name, so `describe_dataset("m-chesapeake-landcover")` fails with
  FileNotFoundError.
- `docs/gameplan-changes.md` says "in code we use GEO-Bench's full name,
  `m-chesapeake-landcover`", and `FakeEngine.CARDS` uses that key.
- **Proposed:** use `m-chesapeake` everywhere (fake engine card, prompts, budgets). The engine
  could alternatively accept an alias, but one name is simpler to audit.
- **Decision (Serena, 2026-10-31):** use `m-chesapeake`, the real folder name; Hilal switches the
  fake engine card and docs.

## Hilal's replies (2026-10-31, after merging PR #2)

- **#2 Accepted.** Budget-stopped runs with a valid checkpoint are `"ok"`; 0-epoch runs stay
  `"timeout"`. The tools record `stopped_by_budget` per trial and show it in trial tables, the
  leaderboard and model cards; `promote` accepts such runs (even at `max_epochs` epochs, since
  a cut-short last epoch is validated). **Small ask:** `metrics.json` has `stop_reason` but no
  `"stopped_by_budget"` key yet; please add `"stopped_by_budget": stop_reason == "budget"`.
  Until then the tools read `stop_reason == "budget"`, so nothing breaks.
- **#5 Closed.** Names containing `_dev_` are dev homes (done with the PR #1 follow-ups).
- **#6 Accepted.** `RealEngine.evaluate` forwards `tta` and checkpoint lists. The ensemble/TTA
  comparison tool comes next; `final_test` always uses whatever won on VALIDATION.
- **#7 Accepted.** contract-v2.md now says each checkpoint uses its own normalisation.
- **#8 Accepted.** Multilabel `per_sample` (per-image tp/fp/fn, micro-F1) is in contract-v2.md.
- **#9 Confirmed.** The official m-chesapeake score is the 7-class mIoU; `miou_excl_nodata` is
  stored with the trial and shown in model cards, never used for selection.
- **#10 Done.** `m-chesapeake` everywhere on our side (fake engine, targets, prompts, docs).

## 11. `describe_dataset`: majority-class baseline and flat-tile fraction (Hilal → Serena)

- **What:** two automatic fields in the recon card that `describe_dataset` returns:
  - `majority_class_baseline`: the val score of always predicting the most common train
    class (accuracy for classification; mIoU of an all-majority mask for segmentation).
  - `flat_tile_fraction`: per class, the fraction of train images that are (nearly) flat,
    e.g. every band's spatial std below a small threshold you pick.
- **Why:** data shortcuts should be visible in the card the agent reads, without a human
  telling it. Example: many of m-brick-kiln's "not brick kiln" tiles are flat, so a model can
  score well by spotting flat tiles; and a score is only meaningful next to the majority
  baseline.
- **Change needed (Serena's files):** compute both in the recon step and pass them through to
  `to_dataset_card`. `DatasetCard` ignores unknown keys today, so once the names are agreed
  Hilal adds the two optional fields to `contracts.py` and shows them in `inspect_dataset`'s
  summary. Facts only: no recipe or advice in the card.
- **Serena's reply (done):** both are computed generically in `inspect_task()` (no task-specific
  logic), on the same seeded 2,000-tile train sample as the stats, and cached in
  `stats/<task>_card.json` under the final names:
  - `majority_class_baseline` (float): VAL score of always predicting the most common TRAIN
    class, with the official metric (accuracy; contract-v2 mIoU of an all-majority mask, where
    the other classes present in val get IoU 0; micro-F1 for multilabel).
  - `flat_tile_fraction` (dict class -> fraction of sampled train tiles below the threshold).
  - Flatness: a tile is flat when the median over bands of (p98 − p2 of its valid pixels in that
    band) / (that band's train std) is below **0.25**. Chosen from data: eurosat's contrast is
    bimodal (5% of tiles < 0.06, 90% > 0.5). The definition and sample size are stored next to
    the fields (`flat_tile_definition`, `flat_tile_n`).
  - `to_dataset_card()` passes both straight through (ignored until the fields exist), and the
    card's notes state them as facts only.

## 12. Optional confidence keys in `predict()` (Serena → Hilal; wire into the MCP output)

- **Exact key names (both optional; existing keys unchanged):**
  - `confidence`: float32, shape `[N]` (classification) or `[N, H, W]` (segmentation, per
    pixel). The temperature-calibrated top-class probability.
  - `needs_review`: bool, same shape as `confidence`. True where `confidence` is below the
    run's threshold, chosen on VALIDATION as the largest coverage whose confident predictions
    have ≤ 1% error.
- **When they're present:** only if the checkpoint's run has `run_dir/calibration.json` (written
  by `gotrain.engine.calibration.calibrate(task, run_dir)`), `predict()` gets a single
  checkpoint, and `tta=False`. That's the setting the temperature and threshold were fitted
  for. For ensembles, TTA or uncalibrated runs, the keys are absent (not None).
- `probabilities` and `labels` are unchanged: temperature scaling never changes the argmax.
- **Also in calibration.json (segmentation):** a tile-level rule (needs review when > 20% of
  the pixels predicted as a target class are uncertain), its flagged share and mean tile mIoU
  for flagged vs unflagged tiles. All numbers there are measured on VALIDATION, where T and the
  threshold were fitted; the file says so in `note`.
- **Where the logits come from:** `val_predictions.npz` stores softmax **probabilities**
  (`probs` [N, K], float32), not logits, so calibration uses log-probabilities as logits
  (exact for temperature scaling). Segmentation runs don't save per-pixel probabilities, so
  `calibrate()` recomputes them from the checkpoint on VALIDATION (CPU).
