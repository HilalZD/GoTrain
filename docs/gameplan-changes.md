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
`gotrain.engine.train` provides `train()` and `evaluate()`, and
`gotrain.engine.data` provides `describe_dataset()`. Until those exist, `GOTRAIN_ENGINE=fake`.
