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
| `GOTRAIN_HOME` | repo root | root for `registry/`, `logs/`, `memory/` |
| `GOTRAIN_STATS_DIR` | `<repo>/stats` | per-task band stats JSON |
| `BEDROCK_MODEL_ID` | `eu.anthropic.claude-sonnet-5-5` | agent model |
| `AWS_REGION` | `eu-central-1` | Bedrock region |

## Layout

```
gotrain/            the package
  contracts.py      Pydantic models passed between agent, tools and engine
  config.py         env-var settings
  engine/           training engines (interface, fake; real one by the ML side)
  tools/            the agent's tools (stubs)
  agent.py, audit.py, storage.py, mcp_server.py, cli.py   (stubs)
memory/lessons.md   lessons carried between tasks
registry/           registered model versions
logs/               audit logs of agent runs
stats/              per-band normalisation stats (train split)
scripts/            run / overnight helper scripts
tests/              pytest suite
docs/               gameplan and its agreed changes
```

See `CLAUDE.md` for the project rules and `docs/gameplan-changes.md` for the plan.
