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
| `m-chesapeake-landcover` ("m-chesapeake") | segmentation | NAIP RGBN |

We say "m-chesapeake" for short. In code we use GEO-Bench's full name, `m-chesapeake-landcover`.

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
