# GoTrain

An AI agent that trains geospatial models on [GEO-Bench](https://github.com/ServiceNow/geo-bench)
by itself. It explores the data, proposes configs, trains, scores on validation, and iterates.
Built for the AUB ACM × RASID hackathon.

> Status: scaffold. Contracts, a fake engine and tests are in place. The agent and tools are stubs.

## Quick start

```bash
git clone https://github.com/HilalZD/GoTrain.git && cd GoTrain
make install   # pip install -e ".[dev]"
make test
make lint
```

Everything runs on the **fake engine** by default (`GOTRAIN_ENGINE=fake`), so no GPU or data is needed.

## Two homes

All agent state (runs, registry, logs, lessons) lives in `GOTRAIN_HOME`, never in the repo.
`GOTRAIN_HOME` must be set:

| Home | Use it for |
|---|---|
| `~/gotrain_dev` | fake-engine runs and experiments. With the fake engine the name **must** end in `_dev` |
| `~/gotrain_home` | real training runs on SageMaker |

The `_dev` rule means a fake run can never write fake results into the real registry or use up a
real task's one-time test.

## Running the agent

```bash
export GOTRAIN_HOME=~/gotrain_home GOTRAIN_ENGINE=real
gotrain run --task m-eurosat --budget-min 120 --phase tune

# overnight on SageMaker (no tmux): one task at a time
nohup python scripts/run_gotrain.py run --task m-eurosat --budget-min 120 \
    > ~/gotrain_home/nohup-m-eurosat.out 2>&1 &

# the ONE-TIME test evaluation: explicit phase + type the task name to confirm
gotrain run --task m-eurosat --phase finalise
```

Each session writes `logs/<task>/<session_id>.jsonl` (one JSON event per line) and a readable
`<session_id>_journal.md` under `GOTRAIN_HOME`.

## Why ML libraries are optional

`torch`, `torchvision`, `timm`, `torchgeo` and `segmentation-models-pytorch` live in an
optional extra, `ml`, and have **no version pins**.

The SageMaker instance already has a CUDA build of torch that matches its GPU driver. If torch
were a core dependency, or were pinned, a plain `pip install -e .` could download a different
torch over it and break GPU training. So:

- **On SageMaker:** `pip install -e .` (or `make install`). It uses the torch that's already there.
- **On a fresh machine with no torch:** `pip install -e ".[ml,dev]"` (or `make install-ml`).

Tip: run `pip install --dry-run -e .` on SageMaker first to see exactly what would change.

## Configuration

All settings are environment variables. See `gotrain/config.py`.

| Variable | Default | Meaning |
|---|---|---|
| `GOTRAIN_ENGINE` | `fake` | `fake` or `real` training engine |
| `GOTRAIN_STORAGE` | `local` | `local` or `s3` |
| `GOTRAIN_BUCKET` | none | S3 bucket (required for `s3`) |
| `GEO_BENCH_DIR` | none | where the GEO-Bench data lives |
| `GOTRAIN_HOME` | **required** | root for `runs/`, `registry/`, `logs/`, `memory/` |
| `GOTRAIN_STATS_DIR` | `<repo>/stats` | per-task band stats JSON |
| `BEDROCK_MODEL_ID` | `eu.anthropic.claude-sonnet-5-5` | agent model |
| `AWS_REGION` | `eu-central-1` | Bedrock region |
| `GOTRAIN_FAKE_SLEEP_S` | `1.0` | seconds the fake engine sleeps per trial |

## Layout

```
gotrain/            the package
  contracts.py      Pydantic models passed between agent, tools and engine
  config.py         env-var settings
  engine/           training engines (interface, fake; real one by the ML side)
  tools/            the agent's tools (plain typed functions)
  prompts/          system prompts for the tune and finalise phases
  agent.py          builds the Strands agent for one task and phase
  audit.py          audit hook: JSONL log + journal of every session
  storage.py        local storage (S3 stub)
  cli.py            `gotrain run ...`
stats/              per-band normalisation stats (train split)
scripts/            run_gotrain.py (same as the CLI, nohup-friendly)
tests/              pytest suite
docs/               gameplan and its agreed changes
```

See `CLAUDE.md` for the project rules and `docs/gameplan-changes.md` for the plan.
