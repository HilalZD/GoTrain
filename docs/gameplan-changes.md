# Gameplan changes

Agreed edits to `docs/gameplan.pdf`. Where the two disagree, this file wins.

## Roles

- **Serena: ML and data.** Owns `scratch/`, `stats/`, `docs/data_notes.md`,
  `docs/figures/`, `gotrain/engine/data.py`, `gotrain/engine/train.py`,
  `gotrain/engine/models.py`.
- **Hilal: agent, tools, logging, registry, storage, MCP.** Owns everything else.
- Never create or edit the other person's files. To change someone else's file, ask them.

## Tasks

Four GEO-Bench tasks:

| Task | Type | Sensor |
|---|---|---|
| `m-eurosat` | classification | Sentinel-2 |
| `m-brick-kiln` | classification | Sentinel-2 |
| `m-pv4ger-seg` | segmentation | aerial RGB |
| `m-chesapeake` | segmentation | NAIP RGBN |

We use GEO-Bench's real task name, `m-chesapeake` (the dataset's folder name). It was
`m-chesapeake-landcover` until request #10; see the incident log.

## Pretrained weights

Sentinel-2 tasks start from Sentinel-2 pretrained weights (SSL4EO, `weights="ssl4eo_s2"`).
Other tasks start from `"imagenet"`.

## Memory proof (ablation)

We prove that memory helps with an ablation. Run `m-brick-kiln` twice with the same budget:

1. with an empty `memory/lessons.md`;
2. with the lessons learned from `m-eurosat`.

Then compare the best val score reached and how many trials it took to get there.

## Time-budgeted trials

- Each trial gets a time budget in minutes. The LR schedule is tied to that budget,
  so the run finishes its schedule however many epochs fit.
- Optional `train_subset_frac` lets a trial train on part of TRAIN for quick scouting.
- Validation always uses the same fixed VAL set, never subsampled, so scores stay comparable.

## SHOULD-haves

- Test-time augmentation (flips + rot90) at evaluation.
- A thin MCP server (`gotrain/mcp_server.py`) exposing the tools.

## Overnight run

- Tasks run one after another, never in parallel.
- Bedrock calls retry with exponential backoff.
- Run inside `tmux`. If tmux is missing, use `nohup ... &`.

## Engine contract (added with the scaffold)

The real engine plugs in through `gotrain/engine/interface.py`:
`gotrain.engine.train` provides `train()`, `evaluate()` and `diagnose()`, and
`gotrain.engine.data` provides `describe_dataset()`. Until those exist, `GOTRAIN_ENGINE=fake`.

`diagnose(task, run_dir) -> dict` was added in agreement with Serena. It returns
`per_class` metrics, `confusion_matrix` (rows = true class, row-normalised),
`curve_verdict` (`"overfitting"` | `"underfitting"` | `"fine"`) and an optional `error_grid_path`.

## Budget

Each task has a fixed GPU budget in minutes, set once by a human (`init_task_budget`).
Tools refuse to start a trial that would exceed what remains. The agent cannot raise it.

## Valid backbone + weights combinations (confirmed by Serena)

| weights | backbones | tasks |
|---|---|---|
| `imagenet`, `none` | resnet18, resnet34, resnet50, convnext_tiny | all |
| `ssl4eo_s2` | resnet18, resnet50 | Sentinel-2 only (m-eurosat, m-brick-kiln) |

Anything else returns status "error" listing these options. The single source of truth is
`gotrain/engine/combos.py` (`VALID_COMBOS`, `check_combo`). The tools reject invalid combos
before training, and the fake engine fails on them the same way the real engine does.

## Smarter GPU use: successive halving and seed checks (tune phase only)

- **`run_halving(task, configs, short_minutes, long_minutes, keep_top)`** is the default
  exploration round. Rung 1 trains every valid config for `short_minutes`; the `keep_top`
  best by val score are re-trained for `long_minutes` as child runs (parent link, as with
  `promote`). The worst-case cost `n * short + keep_top * long` is checked against the
  remaining budget *before* anything starts; if it doesn't fit, the tool refuses and suggests
  a round that would. Every trial of both rungs is a `trial` event in the audit log.
- **`confirm_seeds(run_id, n_seeds=2)`** re-trains the same config (same budget) with new
  seeds and groups the runs (`seed_group`), reporting every score, the mean and the std.
  `leaderboard` shows mean ± std for runs in a seed group and ranks the groups by mean.
- **Noise rule:** a gain between two candidates only counts if it is larger than the measured
  seed noise, i.e. the larger std of the two. Before registering (budget permitting) the
  agent seed-checks its top 2 candidates and registers the one with the best **mean**, saying
  so in the model card notes.
- The fake engine's seed noise is ~±0.3 points (std ≈ 0.003), deterministic per
  (config, seed), so seed checks behave like real ones in dev runs.

## Budget discipline (after the first live m-brick-kiln run)

The first live tune run (fake engine) spent its budget on a promote and could no longer
afford a seed check, so it registered a single-seed score. Its rung-1 screen at 1.5 min was
also inconclusive: scores 0.667-0.683, a spread no bigger than seed noise, because each
trial finished only one epoch of the 15k-sample train split.

- **Budget plan.** The agent's first output is a BUDGET PLAN: ~40% explore (`run_halving`),
  ~35% exploit (`promote` / long runs), ~25% confirm (`confirm_seeds` on the top 1-2),
  re-checked after every round.
- **Seed-check reserve.** `run_trials`, `run_halving` and `promote` report
  `seed_check_cost_min` (confirm_seeds with 2 seeds on the current best = 2 x its budget)
  and `can_afford_seed_check`. `promote` refuses if afterwards the budget could not pay
  for a seed check of the best run (today's best or the new child, whichever costs more),
  unless `force=true`.
- **Screening quality.** `run_halving` reports `screening_spread` (best minus worst rung-1
  score) and a `warnings` list:
  - spread under 0.005: "rung 1 was inconclusive; consider a longer short_minutes or
    train_subset_frac";
  - any rung-1 trial that finished fewer than 3 epochs: "rung 1 trials finished only N
    epochs; screening is unreliable. Use train_subset_frac 0.2-0.3 or a longer
    short_minutes", naming those runs. This is the check that catches the brick-kiln
    incident: its spread (0.016) was above 0.005, but every trial ran a single epoch.

  The >5,000-samples rule for `screening_hint` was replaced by an epoch-time estimate; see
  "Sizing the screen" below.
- **Seed-check reserve is deliberately strict** (agreed): it covers a seed check of today's
  best *or* the promoted child, whichever costs more, because a winning child is what gets
  registered.
- **Registering single-seed** is allowed only when the plan explicitly ran out of budget,
  and the model card notes must say so.
- **Fake engine.** Epoch time scales with `train_subset_frac`, so a 0.25 subsample fits ~4x
  the epochs in the same minutes; its reachable score is capped slightly lower
  (3 points x (1 - frac)). Deterministic as before.

## Visual diagnostics: the error grid

- **Engine contract.** `diagnose(task, run_dir)` may return `error_grid_path`: a PNG with
  rows of image | truth | prediction for the worst validation samples. The fake engine draws
  a synthetic one (numpy + PIL) into `<run_dir>/diagnostics/error_grid.png`, so the loop works
  before the real diagnostics land.
- **Only `diagnose` shows images.** The tool downscales the grid to at most 1024 px on the
  long side and under ~1 MB (`gotrain/images.py`), saves it as
  `diagnostics/error_grid_for_model.png` in the run folder, and returns metadata only. Its
  agent adapter (`diagnose_for_model`) sends the JSON summary as a text block plus the PNG as
  an image block. If the file is missing or unreadable, the agent gets text only with an
  `image_note`. A rule test fails if any other module builds an image block.
- **Strands 1.63.0 detail.** A tool result is `{"status", "content": [...]}`; an image block is
  `{"image": {"format": "png", "source": {"bytes": ...}}}` (raw bytes). The Bedrock provider
  keeps only `content`, `toolUseId` and `status`, so all other data (including `budget`)
  must live inside the text block.
- **Audit.** An `image_shown` event records path, size and dimensions. Image bytes are never
  logged.
- **Prompt.** After every diagnose the agent writes a **VISUAL OBSERVATION** (what the worst
  predictions have in common) and a linked **HYPOTHESIS**, and the next config's reason must
  cite them.

## Re-screening on the epoch warning

`run_halving` rows include `minutes_per_epoch`. If the rung-1 epoch warning fires and at least
half of the explore share remains, the agent re-screens with `short_minutes` of at least
3 x the observed `minutes_per_epoch` (the warning states that number) instead of trusting the
ranking.

## After the first live m-pv4ger-seg run

That run showed two gaps: after `diagnose` showed the image, the agent made its next tool
call without writing the VISUAL OBSERVATION (it only appeared in the final summary); and its
first `run_halving` round ran 6 trials at 1.5 min on full data, all of which timed out with 0
epochs, wasting 9 min.

- **Observation gate (enforced, tune phase).** A Strands hook (`ObservationGate`,
  `BeforeToolCallEvent.cancel_tool`) arms after a successful `diagnose` and cancels every
  tool call (any tool) until an assistant message contains both "VISUAL OBSERVATION" and
  "HYPOTHESIS" (case-insensitive). Writing them in the same message as the next tool call
  counts. The cancel message says exactly what to write. Every block is an
  `observation_enforced` audit event, and the release is an `observation_written` event.
  `diagnose`'s text result ends with "Next: write VISUAL OBSERVATION and HYPOTHESIS before
  any other tool call." Blocked calls count toward the tool-call limit, so an agent that
  never complies is still stopped.
- **Fail fast.** In `run_trials` and rung 1 of `run_halving`, a trial that ends with status
  "timeout" and 0 epochs stops the round at once: the remaining configs never start and are
  listed in `not_run`, only the minutes actually used are charged, rung 2 is skipped
  (`stopped_early: true`), and a warning says how long the trial ran and which
  `train_subset_frac` / minutes would fit 3 epochs. Errors (e.g. invalid bands) don't stop
  the round.
- **Sizing the screen** (`gotrain/engine/throughput.py`). A deliberately simple heuristic:

      minutes_per_epoch ~= train_samples x train_subset_frac x height x width / PIXELS_PER_MINUTE

  `PIXELS_PER_MINUTE` (20,000 64x64 images per minute) is exactly the fake engine's timing
  model; the fake engine computes epoch time with this function and states the model in
  `describe_dataset`'s notes. `inspect_dataset` returns a `screening_hint` when 3 full-data
  epochs don't fit in a 2-minute screening trial, with the `train_subset_frac` (or
  `short_minutes`) that would fit. With the fake numbers: m-eurosat 0.1 min/epoch (no hint),
  m-brick-kiln 0.75, m-chesapeake 2.4, m-pv4ger-seg 3.75 (hint: subsample <= 0.15).
  **For the real engine the constant is a placeholder** until measured on the T4; observed
  `minutes_per_epoch` from trials beats the estimate.

## Harness fix: sessions that ended early (second live m-pv4ger-seg run)

After `diagnose`, the agent put its observation in a config's `reason`, was blocked 4
times, then wrote VISUAL OBSERVATION + HYPOTHESIS as a text-only message ("I'm about to run
them"). In Strands a text-only reply ends the agent loop, so the session finished with 26 of
40 minutes unused and nothing registered.

- **Gate message and tool-args acceptance.** The `ObservationGate` cancel text now says:
  "write VISUAL OBSERVATION and HYPOTHESIS as plain text in your message AND include your
  next tool call in the SAME response; do not end your turn." The gate also accepts the
  observation when the blocked call's own arguments (e.g. a config `reason`) contain both
  keywords. `observation_written` events record the `source` (`message` or `tool_args`).
- **Continuations** (`gotrain/driver.py`, used by the CLI). When a tune session ends its
  turn while budget remains above the cost of one short trial (the cheapest per-trial
  budget used so far, or 2 min) and the agent has registered nothing in this session, the
  driver sends "Continue your plan. Remaining budget: X min. Remember to finish with
  register_model." Up to 3 per session, each a `continuation` audit event and a journal
  line; a 4th is refused (`continuations_exhausted`). No continuation once the tool-call
  limit has ended the loop, and none in the finalise phase.
- **Registration safety net.** If a tune session ends with nothing registered in it but a
  successful run exists, the driver registers the best run by validation (a seed group
  counts by its mean, registered via its base run) with the notes "auto-registered by
  harness: agent did not register", plus the basis. Audit event `auto_registered`; the
  journal shows "HARNESS: AUTO-REGISTERED ... not chosen by the agent"; the CLI prints a
  warning; `session_end` records `registered_by: "harness"`. If the best run is already
  registered from an earlier session, nothing is duplicated (`auto_register_failed`).
- **Prompt.** Never end a turn with a text-only reply while the plan and budget remain;
  write observations in the same response as the next tool call.

## Final feature list (v2)

The target feature set for the final demo. Status as of this commit: **done** = built and
tested on the fake engine; **partial** = groundwork exists; **planned** = not started.

| Feature | Status |
|---|---|
| Ensemble of top checkpoints + flip/rot90 test-time augmentation, chosen on validation | done: `evaluate_candidates` (tune), ensemble registration, `final_test` scores the registered members + TTA |
| Agent knows the published GEO-Bench best score per task | planned |
| Forecast-driven halving (learning-curve extrapolation) | planned (successive halving done) |
| Enforced visual self-diagnosis | done (error grid + ObservationGate) |
| Seed-verified decisions | done (`confirm_seeds`, mean ± std, noise rule) |
| Budget plan + fail-fast + guaranteed registration | done (budget plan prompt, seed-check reserve, 0-epoch fail-fast, continuations + safety net) |
| Cross-task memory with same-task ablation | partial: lessons memory done; ablation runs planned |
| Data integrity scan (train<->val near-duplicates) | planned |
| Test vault | done (single entry point `final_test`, TEST_LOCK, rule test) |
| Confidence intervals on test scores | partial: `per_sample` in contract v2 enables bootstrap CIs |
| One-command replay | planned |
| Incident log | planned (incidents recorded by hand in this file so far) |
| Model packages: agent-readable card, calibrated confidence, cost receipt | partial: model card done |
| MCP server (train, status, find_model, predict) | planned (`predict` in contract v2) |
| Local <-> S3 storage | partial: local done, S3 stub (PutObject denied for now) |
| 4 tasks + zero-touch 5th | partial: 4 tasks on the fake engine |
| Label-efficiency curve | planned (`train_subset_frac` exists) |

## Engine contract v2

Full spec for the real engine: `docs/contract-v2.md`. Backward compatible with v1.

- `evaluate(task, checkpoint_path | [paths], split, tta=False)` adds `n_samples` and
  `per_sample` (classification: 0/1 correct per image; segmentation: per-image, per-class
  intersection and union pixel counts). A list of checkpoints is an ensemble that averages
  probabilities.
- `predict(task, checkpoint_path | [paths], images N x C x H x W, tta=False)` returns
  `probabilities` and `labels`.
- The fake engine implements both. Its reported score is computed from `per_sample`, ensembles
  gain up to +0.6 points and TTA +0.4 points (deterministic), and `predict` genuinely averages
  each fake model's probabilities (its models are flip/rotation invariant, so `tta` has no
  effect there).

## After merging Serena's engine (PR #1)

- **Shared file.** `docs/contract-requests.md` is shared: either of us may append requests.
- **Engine-only rules up front** (`gotrain/engine/rules.py`). The real engine rejects:
  losses that don't fit the task type (classification: `ce`, `focal`, `bce`; segmentation:
  `ce`, `dice_ce`, `focal`, `bce`, `dice_bce`), `bce`/`dice_bce` on segmentation with other
  than 2 classes, and segmentation `img_size` not a multiple of 32. `run_trials`,
  `run_halving`, `promote` and `confirm_seeds` now reject these before training with a fix-it
  message, the fake engine fails on them too, `inspect_dataset` shows them as
  `training_rules`, and the tune prompt lists them. A test fails if `rules.py` drifts from
  the engine's own `LOSSES`, `BINARY_SEG_LOSSES` and `UNET_STRIDE`.
- **Dev homes.** A home is a dev home if its name ends in `_dev` **or contains `_dev_`**, so
  `~/gotrain_dev_serena` works with the fake engine (Serena's request #5).
- **Fake names = real names.** The fake engine builds its cards from Serena's cached recon
  cards (`stats/<task>_card.json`, via her `to_dataset_card`) where they exist, so band and
  class names (e.g. "Red", "Annual Crop", "Sea and Lake") match the real engine and lessons
  and configs carry over. Tasks without a cached card keep the hand-written fake card.
- **Request #2 answered:** "timeout" = the budget ran out inside the first epoch, as the
  engine does; it is exactly what fail-fast assumes. **Request #6** was covered by contract v2.
- **Hardening.** `tests/test_rules.py` also flags any `allow_test=` (other than `False`, or
  forwarding the caller's own flag) outside the top-level `evaluate()` of
  `gotrain/engine/train.py`. The diagnose summary picks the per-class metric by name
  (`iou`, then `recall`, then `f1`), not by dict order.

## Incident log

### 2026-10-31: test-vault loophole (found in review, fixed before any real test)

- **Found:** while reviewing Serena's PR, before any real-engine final test had run.
- **Problem:** the one-time test lock (`registry/<task>/TEST_LOCK.json`) lives inside each
  `GOTRAIN_HOME`. A second home has no lock, so with the real engine a task's test set
  could be scored once per home, which would make "tested once" meaningless.
- **Fix:**
  - `GOTRAIN_OFFICIAL_HOME` (default `~/gotrain_home`): with `GOTRAIN_ENGINE=real`, the
    finalise phase and `final_test` refuse to run unless `GOTRAIN_HOME` is the official home.
  - A machine-wide, append-only test ledger (`GOTRAIN_TEST_LEDGER`, default
    `~/.gotrain_test_ledger.jsonl`), outside every home. Each real test appends "started"
    before evaluating, then "done" (with the TestRecord) or "failed". A real `final_test` is
    refused if the ledger already has the task, in any status, from any home.
  - Fake-engine tests stay per-home and never touch the ledger.
- **Impact:** none. No real test evaluation had been run.

### 2026-10-31: wrong task name `m-chesapeake-landcover` (request #10)

- **Found:** by Serena, running the real engine: `describe_dataset("m-chesapeake-landcover")`
  failed with FileNotFoundError.
- **Cause:** the name came from the fake engine's hand-written card (and the docs copied it);
  GEO-Bench's task and folder are called `m-chesapeake`.
- **Fix:** renamed to `m-chesapeake` everywhere on our side (fake engine, targets, prompts,
  docs, tests). A test fails if the old name reappears in the package. The fake engine now
  loads Serena's real m-chesapeake card (classes start with "no data").
- **Impact:** none on results. Any dev home with runs under the old name keeps them under that
  name; they are not mixed with m-chesapeake runs.

## Calibrated timing (real T4)

- **Measurement** (SageMaker T4; m-eurosat, resnet50, batch 64, fp16; 2,000 train + 1,000 val
  images per epoch): ~1.9 s per epoch at 64 px, ~10 s at 224 px.
- **Model** (`gotrain/engine/throughput.py`, `REAL_T4`, the one place to update): seconds per
  image = pixels / 1,024 Mpx/min + 0.393 ms overhead, for train *and* val images. One
  pixels-per-minute constant can't fit both points (~388 vs ~903 Mpx/min): small images are
  dominated by a fixed per-image cost. The two terms reproduce both measurements exactly.
- **Segmentation** (Serena's measurement: m-pv4ger-seg, U-Net resnet34, 320 px, batch 16,
  3,000 train + 403 val images): ~34 s per epoch. The classification rate predicts only
  ~21.8 s (the U-Net decoder costs extra), so segmentation tasks get their own pixel rate,
  ~640 Mpx/min, with the same 0.393 ms per-image overhead. It reproduces 34 s; the
  classification fit is unchanged. m-chesapeake (256 px, 4,000 images) is extrapolated:
  ~26 s per epoch. No real-T4 screening hint fires on any of our four tasks.
- **Fake engine keeps its own, slower `FAKE_TIMING`** (20,000 64x64 images per minute, train
  only). At real speed every dev trial would finish all its epochs in seconds, so budgets,
  halving, fail-fast and the epoch warnings would never be exercised. Hints and fail-fast
  advice use the active engine's model, so they stay consistent in dev runs too.

## Goal-aware agent: published reference scores

- `gotrain/knowledge/targets.json` holds, per task, published GEO-Bench reference **TEST**
  scores (metric, value as a fraction, model, source), plus a `human_baseline` (Serena's
  hand-trained models, VALIDATION scores) for m-eurosat (accuracy 0.986) and m-pv4ger-seg
  (mIoU 0.959):

  | Task | Metric | Best reference | Model |
  |---|---|---|---|
  | m-eurosat | accuracy | 0.986 (also 0.9816, MoCo ResNet50) | TerraFM (fine-tuned) |
  | m-brick-kiln | accuracy | 0.9879 | Prithvi-EO-2.0-600M |
  | m-pv4ger-seg | mIoU | 0.9523 | Prithvi-EO-2.0-600M |
  | m-chesapeake | mIoU | 0.7081 | DINO ResNet50 |

- **Scores only.** The schema forbids any other key, and a test fails if the file contains a
  hyperparameter-like key (lr, backbone, batch_size, ...): humans never pick hyperparameters.
- `inspect_dataset` returns the task's `targets` (with `best_reference` and a note that these
  are test numbers while the agent sees validation scores). The tune prompt uses them:
  far below (> ~3 points) explore more; in between exploit; within ~1 point or above stop
  exploring and spend the rest on seed checks, then ensembles/TTA (`evaluate_candidates`).
- **Sources:** m-brick-kiln, m-pv4ger-seg, m-chesapeake and the MoCo ResNet50 m-eurosat
  number come from the Prithvi-EO-2.0 paper appendix, Tables A1-A4 (arXiv 2412.02732);
  m-eurosat 0.986 from the TerraFM paper (arXiv 2506.06281).
- Serena's other measured facts (which configs her hand baselines used, etc.) are **not** put
  in prompts, memory or targets: that would hand the agent a human-tuned recipe.

## Serena's contract requests after PR #2

PR #2 (real engine, contract v2, recon for m-brick-kiln and m-chesapeake) merged into main
after the contract checks and the full local suite passed; the GPU suite re-runs on SageMaker.

- **#2 budget-stopped runs are "ok".** A run stopped by its time budget with at least one
  validated epoch and a checkpoint has status "ok" and `stopped_by_budget: true` (read from
  the engine's `metrics.json`; `stop_reason == "budget"` also counts). Trial tables, the
  leaderboard (with a note: the curve may still be rising) and model cards show it, and
  `promote` accepts such runs even when they show `max_epochs` epochs (the real engine
  validates a cut-short last epoch). 0-epoch runs stay "timeout".
- **#5** closed: `_dev_` homes are dev homes (done with the PR #1 follow-ups).
- **#6** `RealEngine.evaluate` forwards `tta` and checkpoint lists (contract v2). The
  ensemble/TTA comparison tool is the next chunk; `final_test` always uses whatever won on
  VALIDATION.
- **#7** normalisation is per checkpoint and owned by the engine; contract-v2.md reworded.
- **#8** multilabel `per_sample` = per-image tp/fp/fn, score = micro-F1 (in contract-v2.md).
- **#9** the official m-chesapeake score is the 7-class mIoU (GEO-Bench comparable);
  `miou_excl_nodata` is recorded with the trial and shown in model cards, never used for
  selection.
- **#10** the task is `m-chesapeake` (see the incident log).
- **New request #11** (to Serena): `describe_dataset` should compute
  `majority_class_baseline` and a per-class flat-tile fraction, so data shortcuts appear in
  the card the agent reads.

## Ensembles and test-time augmentation (after the first real m-eurosat run)

The first real-engine agent run on m-eurosat reached 0.9857 ± 0.0006 val in 17 min and
stopped with 3 min left. Leftover budget can now buy ensembles and TTA, chosen on
VALIDATION only.

- **`evaluate_candidates(task, candidates)`** (tune phase). A candidate is
  `{"run_ids": [1-4 runs], "tta": bool}`; at most 6 per call. Members must be "ok" runs of
  the task; any invalid candidate rejects the whole call before anything runs. Each candidate
  is scored by `engine.evaluate(checkpoint list, split="val", tta=...)`; the single best run
  without TTA is always scored too, as the baseline. Returns a ranked table (members, TTA,
  val score, `delta_vs_single_best`, `ties_with_top`), the seed noise used, and a
  `recommended` pick.
- **Budget.** An up-front estimate (val images x members x views at each member's img_size,
  priced at training speed per image, so on the safe side; TTA = 8 views) must fit the
  remaining budget. Real runs are charged the measured inference time; the fake engine is
  charged the estimate (it scores instantly, like its simulated training time).
- **Rules** (prompt and tool description): only once the score is in the target zone or
  exploration is done; compare single best ± TTA and the top-2 and top-3 ensembles ± TTA;
  **on a tie within the measured seed noise (the largest std from `confirm_seeds`), choose
  the simpler candidate: fewer members first, then no TTA** (`recommended` applies this;
  without a seed check only exact ties count); **at most 2 calls per session**, enforced by
  the `SelectionQuota` hook in `gotrain/agent.py` (calls that returned an error don't count),
  to limit selection overfitting on validation.
- **Every result is stored** in `registry/<task>/candidates.jsonl`.
- **`register_model(run_ids=[...], tta=...)`** registers an ensemble and/or TTA model, but only
  one scored by `evaluate_candidates` with exactly those members and flag; its measured val
  score is registered. Member folders are copied to `registry/<task>/vN/members/<run_id>/`;
  `ModelVersion` gains `members` (best first; `run_id` and `config` are the best member's) and
  `tta`, and the model card lists the members. `register_model(run_id)` works as before
  (old versions load with `members=[run_id]`, `tta=false`).
- **`final_test`** passes the registered checkpoint list and TTA flag to `engine.evaluate`
  (a single checkpoint without TTA is called exactly as before).

## Lessons during tune, overnight queue (M1, M2)

- **M1:** at the end of every tune session (agent- or harness-registered) the driver asks the
  agent for 2-5 lessons as JSON, from VALIDATION results only, and writes them with
  `write_lesson`, tag `source:tune` (+ sensor and task type). If the agent gives fewer than 2
  or fails, one automatic lesson is written from the leaderboard (best config + val score).
  Finalise lessons are unchanged.
- **M2:** `scripts/overnight.py <plan.yaml>` runs queue entries sequentially, tune phase only
  (never finalise). Status in `<home>/queue_status.json`, logs in `<home>/logs/queue/`. Kill
  switch: `~/GOTRAIN_STOP` (finish the current entry, then stop; `GOTRAIN_STOP_FILE` overrides).
  `GOTRAIN_MEMORY=empty` uses `memory/lessons_isolated.md` (emptied at the start of the entry).
  `label_fracs` retrains the registered config with `engine.train` directly (outside the task
  budget, per-run limit = the registered run's minutes, at least 2) and writes validation
  scores to `<home>/label_efficiency/<task>.json`. Choices made: the 4th entry of
  `overnight_a.yaml` is a label_fracs entry that skips tuning when a model is already registered
  in its home (it reuses the one the first m-eurosat entry registered; if none exists it tunes).
  `pyyaml` added to the dependencies.

## Confidence intervals on the test score (M3)

`final_test` computes a 95% bootstrap CI (1000 resamples of test images, seed 0,
`gotrain/bootstrap.py`) from `per_sample`: accuracy (classification), mIoU recomputed from
per-image intersection/union (segmentation), micro-F1 (multilabel). `TestRecord` gains optional
`ci_low`, `ci_high`, `ci_level`, `ci_resamples` (old records load with `None`); the lock file and
ledger carry them, and a "Test result" section is appended to the version's model card. A CI
failure never loses the score: the record is then written without a CI.

## Results export (M5)

`gotrain export --out results/ [--home H ...]` (default GOTRAIN_HOME) copies leaderboards, seed groups,
budget, test records, model cards/versions, journals, queue_status, label-efficiency files and
lessons into `results/<home name>/` (never checkpoints) and writes `results/summary.md`: a table
per task (val mean ± std, test + CI, GPU minutes, Bedrock tokens from session_end usage) with
published references and the human baseline from `targets.json`.

## Shared-home queues, seed-group registration

- Queue files are per plan: `queue_status_<plan>.json`, `logs/queue/<plan>/`; export reads all `queue_status_*.json`. Appends (lessons, journals, leaderboards) take a lock file next to the target, so two writers on one disk never interleave.
- `register_model(run_id)` of a seed-group member registers the group ORIGINAL run (as the harness does). `ModelVersion` gains `val_std`, `val_n`, `single_val_score`; `val_score` is then the group mean, the model card headline is mean ± std with the single-run score second, and leaderboard rows carry `val_headline`. Ensembles/TTA keep their measured score.

## Single-GPU queue, imbalance fields

- `configs/overnight.yaml` replaces overnight_a/b (one queue, one GPU); `configs/smoke.yaml` is a 4-min real-engine check in `~/gotrain_smoke`. `launch_overnight.sh` exports `GOTRAIN_DATA_CACHE` (default `~/data_cache`, the engine default) and refuses if it is missing.
- `DatasetCard` gains optional `majority_class_baseline` and `flat_tile_fraction`; `inspect_dataset` shows them when set; the tune prompt says to use minority-class F1/recall when the baseline is high or a class has many flat tiles.
