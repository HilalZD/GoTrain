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
- **Tools never raise to the agent.** Wrap every tool in `@tool` (`gotrain/tools/_common.py`):
  failures come back as `{"error", "hint"}`, and every output carries `"budget"`.
- Tools touch files only through `get_storage()` and storage keys, never raw paths.
- The audit logger is `gotrain/audit.py`. Never add a `logging.py`, because it would shadow
  the stdlib module.

## State on disk (under GOTRAIN_HOME)

`GOTRAIN_HOME` is required and is never the repo. Use two homes:

- `~/gotrain_dev` for fake-engine runs. With `GOTRAIN_ENGINE=fake`, the home name must end
  in `_dev` or nothing starts, so a fake run can never lock a real task's test set.
- `~/gotrain_home` for real runs.

```
runs/<task>/<run_id>/          config.json, result.json, git_commit, checkpoints/
registry/<task>/leaderboard.jsonl, budget.json, v1/, v2/ ..., TEST_LOCK.json
memory/lessons.md
```

`run_id` = `<task>-<YYYYmmdd-HHMMSS>-<4 hex>`.

## Environment

- Settings come from env vars only (`gotrain/config.py`): `GOTRAIN_ENGINE` (fake|real),
  `GOTRAIN_STORAGE` (local|s3), `GOTRAIN_BUCKET`, `GEO_BENCH_DIR`, `GOTRAIN_HOME`,
  `GOTRAIN_STATS_DIR`, `BEDROCK_MODEL_ID`, `AWS_REGION`, `GOTRAIN_FAKE_SLEEP_S`.
- Get an engine with `gotrain.engine.get_engine()`, never by importing a concrete engine.
  The default is the fake engine, so everything runs without a GPU.
- Run the agent with `gotrain run --task <task> --budget-min <n> --phase tune`. Finalise
  (the one-time test) needs `--phase finalise` and typing the task name. There is no "all" mode.
- Agent phases own their toolsets (`gotrain/agent.py`): `final_test` exists only in finalise.
  System prompts are editable files in `gotrain/prompts/`.
- **ML libraries are an optional extra (`.[ml]`) with no version pins.** On SageMaker, run
  `pip install -e .` (or `make install`), never `.[ml]`. The instance's CUDA torch must not be
  replaced. Don't add torch, timm and similar packages to core dependencies or pin them.
- Bedrock only works on SageMaker. Locally, use the fake engine and the tests.

## Hardware & environment (SageMaker)

- **GPU: NVIDIA T4, 16 GB.** Train with fp16 autocast + `GradScaler`. **Never bf16** (the T4
  has no bf16 support). ResNet18/34/50, ConvNeXt-Tiny, and U-Net/FPN on those backbones fit.
- **RAM: 15 GB.** Cache dataset splits as `.npy` files instead of holding everything in memory.
- **Python 3.12, torch 2.8.0, numpy 2.5.4. Never change them** (no upgrades, no reinstalls).
- **strands-agents 1.63.0 is pinned.** Bedrock access is confirmed working from the instance.
- **Do NOT import `strands_tools`.** It is preinstalled but incompatible with 1.63.0.
  `tests/test_rules.py` enforces this.
- **S3 `PutObject` is denied for now**, so `GOTRAIN_STORAGE` defaults to `local`.
- **No tmux.** Run long jobs with `nohup ... &` (see `scripts/run_gotrain.py`).
- **The clock is UTC.** All timestamps in runs and logs are UTC.
