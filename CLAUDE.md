# GoTrain

## Mission

GoTrain is an AI agent that trains geospatial models on GEO-Bench by itself.
It explores the data, proposes training configs, trains, scores on VALIDATION, and iterates.
The agent is built with the Strands Agents SDK, using Claude Sonnet on Amazon Bedrock (eu-central-1).
Code is written locally, pushed to GitHub, and run on a SageMaker GPU instance via `git pull`.
At the end, each task's best model is registered and scored once on TEST.

## Read first

- `docs/gameplan.pdf` is the master plan.
- `docs/gameplan-changes.md` lists our agreed edits. It wins when the two disagree.

## File ownership

- **Serena** owns `scratch/`, `stats/`, `docs/data_notes.md`, `docs/figures/`,
  `gotrain/engine/data.py`, `gotrain/engine/train.py`, `gotrain/engine/models.py`.
- **Hilal** owns everything else.
- **Never create or edit the other person's files.**

## Permanent rules

- **The TEST split is sacred.** Only `gotrain/tools/final_test.py` may request the test
  split, and only through `engine.evaluate`. `tests/test_rules.py` enforces this.
- **The agent tunes on VALIDATION only.**
- **Tools take and return contract models** (`gotrain/contracts.py`) or JSON-serializable dicts.
- Use type hints and small, readable functions. Every tool gets a test. Keep the code ruff-clean
  (`make lint`).
- **Humans fix bugs in tools. Humans never pick hyperparameters for agent runs.**
- The audit logger is `gotrain/audit.py`. Never add a `logging.py`, because it would shadow
  the stdlib module.

## Environment

- Settings come from env vars only (`gotrain/config.py`): `GOTRAIN_ENGINE` (fake|real),
  `GOTRAIN_STORAGE` (local|s3), `GOTRAIN_BUCKET`, `GEO_BENCH_DIR`, `GOTRAIN_HOME`,
  `GOTRAIN_STATS_DIR`, `BEDROCK_MODEL_ID`, `AWS_REGION`.
- Get an engine with `gotrain.engine.get_engine()`, never by importing a concrete engine.
  The default is the fake engine, so everything runs without a GPU.
- **ML libraries are an optional extra (`.[ml]`) with no version pins.** On SageMaker, run
  `pip install -e .` (or `make install`), never `.[ml]`. The instance's CUDA torch must not be
  replaced. Don't add torch, timm and similar packages to core dependencies or pin them.
- Bedrock only works on SageMaker. Locally, use the fake engine and the tests.
