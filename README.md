# GoTrain 

**An AI agent that trains geospatial models on its own.**
Give it a GEO-Bench dataset and a GPU time budget. It explores the data, plans how to spend its budget, screens ideas, looks at its own mistakes, verifies gains against seed noise, registers the best model, and packages it as a tool GoPilot can call through MCP. No human in the loop.

Built for the AUB ACM × RASID GoTrain Challenge (31 Oct – 1 Nov 2026) with Strands Agents, Claude Sonnet on Amazon Bedrock, and a SageMaker T4 GPU.

---

## Results

Tuned on **validation only**. The test split was touched **once per task**, at the end, by `final_test`.

| Task | Type | Metric | Val (agent) | Test (agent, 95% CI) | Published reference | Human baseline (val) |
|---|---|---|---|---|---|---|
| m-eurosat | classification | accuracy | 0.986 ± 0.002 | **0.983** [0.975, 0.990] | 0.982 (MoCo-RN50) / 0.986 (TerraFM) | 0.986 |
| m-brick-kiln | classification | accuracy | 0.990 ± 0.002 | **0.982** [0.973, 0.989] | 0.988 | — |
| m-pv4ger-seg | segmentation | mIoU | 0.958 ± 0.002 | **0.955** [0.947, 0.961] | 0.952 | 0.959 |
| m-chesapeake | segmentation | mIoU (7 classes) | 0.777 | **0.757** [0.740, 0.772] | 0.708 (DINO-RN50) | — |

Published references: Prithvi-EO-2.0 (arXiv 2412.02732, appendix A1–A4) and TerraFM (arXiv 2506.06281). Human baselines are hand-tuned scripts written by our team before the agent existed; only their **scores** are shown to the agent, never their settings.

Full tables, model cards, journals and confidence intervals: [`results/summary.md`](results/summary.md).

---

## How the agent works

```
inspect & audit data → budget plan → screen ideas (successive halving)
   → look at its own errors (vision) → promote the leader
   → seed check (is the gain real?) → ensemble / TTA only if it beats the noise
   → register → write lessons → export as a GoPilot tool
```

- **Budget plan first.** ~40% explore, 35% exploit, 25% confirm, with a reserve that the agent cannot spend before verifying its winner.
- **Config-driven training.** The agent picks a validated `TrainConfig`; a fixed training engine executes it. Invalid combinations (wrong weights for a sensor, wrong loss for a task) are rejected before any GPU time is spent.
- **Successive halving.** Many short trials, the best get more epochs.
- **Visual self-diagnosis.** `diagnose` returns an image of the model's errors; the agent must write a visual observation and a hypothesis before its next action.
- **Seed-verified decisions.** A gain only counts if it is larger than the spread across random seeds. The headline score is the seed-group mean ± std.
- **Honest ensembles.** `evaluate_candidates` compares the single best run with 2- and 3-model ensembles, with and without test-time augmentation. On a tie, the simpler model wins.
- **Goal-aware.** The agent knows the published reference for each task and stops exploring once it is close, spending the rest on verification.
- **Memory.** Lessons are written at the end of every session (validation evidence only) and read at the start of the next.
- **Data audit.** Before training, the agent reads an audit of the dataset: mislabelled class names, shortcut risks (majority-class baseline, flat tiles), train/val leaks.

## Honesty by design

- `split="test"` appears in exactly one file (`gotrain/tools/final_test.py`); a test enforces it.
- The test phase is a separate agent with different tools, a typed confirmation, a lock file, and a **global append-only ledger** that refuses a second test on the same task from any folder.
- Every tool call is logged as JSONL, plus a human-readable journal per session.
- Bootstrap 95% confidence intervals on every test score.

---

## Install

```bash
git clone https://github.com/HilalZD/GoTrain.git && cd GoTrain
make install            # pip install -e ".[dev]" (never upgrades torch/numpy)
pytest -q               # full suite; engine tests need torch + data
```

Environment:

| Variable | Meaning |
|---|---|
| `GOTRAIN_HOME` | working folder (registry, runs, logs, memory) |
| `GOTRAIN_ENGINE` | `fake` (instant, for development) or `real` (GPU) |
| `GEO_BENCH_DIR` | where GEO-Bench data lives |
| `GOTRAIN_DATA_CACHE` | pre-built data cache, default `~/data_cache` |

Build the data cache once (CPU, train + val only):
```bash
python scripts/build_cache.py m-eurosat m-brick-kiln m-pv4ger-seg m-chesapeake
```

## Run

```bash
# tune on validation
gotrain run --task m-eurosat --budget-min 40 --phase tune

# rehearse the final step on validation (no lock, no ledger)
gotrain run --task m-eurosat --phase finalise --rehearsal

# the real, one-shot test (official home only)
gotrain run --task m-eurosat --phase finalise

# unattended queue of tasks (tune only), with a kill switch at ~/GOTRAIN_STOP
bash scripts/launch_overnight.sh configs/overnight.yaml

# copy results into the repo
gotrain export --out results/ --home ~/gotrain_home
```

## Use the models from GoPilot (MCP)

Every registered model is packaged automatically as a bundle with a model card, then served as an MCP tool:

```bash
python -m gotrain.mcp.models_server     # list_models, detect_brick_kilns, segment_solar_panels, ...
python scripts/demo_gopilot.py          # a Strands agent asks "Are there brick kilns in this tile?"
```

Classification tools return a label, probabilities and confidence. Segmentation tools return a mask GeoTIFF, GeoJSON polygons and area per class. See [`docs/gopilot_export.md`](docs/gopilot_export.md).

---
## Repository

```
gotrain/
  agent.py, driver.py, cli.py      agent phases, continuations, CLI
  contracts.py, config.py          Pydantic schemas and settings
  tools/                           inspect, trials, search, diagnose, candidates,
                                   registry, final_test, memory
  engine/                          interface, fake engine, real engine
                                   (data, models, train, diagnostics, audit),
                                   rules, combos, throughput
  prompts/                         tune.md, finalise.md
  knowledge/targets.json           published scores (scores only, no recipes)
  mcp/models_server.py             GoPilot-ready MCP server
scripts/                           overnight queue, cache builder, demo
configs/                           overnight and smoke plans
docs/                              gameplan, contract, data notes, figures
results/                           exported leaderboards, cards, journals
tests/                             ~400 tests
```

## Production roadmap

- Storage backend on S3 (local filesystem today) for datasets, checkpoints and model versions.
- GoTrain itself as an MCP server for GoPilot: `train`, `status`, `find_model`, `predict`.
- Parallel trials across several GPUs, with the same budget ledger.

## Team

Hilal Zeidan · Serena Dalal — AUB.
