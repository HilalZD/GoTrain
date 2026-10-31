"""Tool: register a finished run as the next model version, registry/<task>/vN/.

A version folder holds the run's config, result, git commit and checkpoint, plus
model_card.md and model_version.json (a ModelVersion).
"""

import json
import re

from gotrain.contracts import ModelVersion, TrainConfig
from gotrain.engine import get_engine
from gotrain.tools._common import (
    error,
    find_entry,
    now_utc,
    run_key,
    storage,
    tool,
    unknown_run,
)
from gotrain.tools.inspect import summarize_card

VERSION_DIR = re.compile(r"^registry/[^/]+/(v[1-9][0-9]*)/")


def version_key(task: str, version: str) -> str:
    return f"registry/{task}/{version}"


def list_versions(task: str) -> list[str]:
    """Registered versions of a task, oldest first: ['v1', 'v2', ...]."""
    found = {m[1] for k in storage().list(f"registry/{task}") if (m := VERSION_DIR.match(k))}
    return sorted(found, key=lambda v: int(v[1:]))


def load_version(task: str, version: str) -> ModelVersion | None:
    key = f"{version_key(task, version)}/model_version.json"
    if not storage().exists(key):
        return None
    return ModelVersion.model_validate_json(storage().read_text(key))


def lineage(run_id: str) -> list[str]:
    """[run_id, parent, grandparent, ...] following promote() links."""
    chain = [run_id]
    entry = find_entry(run_id)
    while entry and entry.get("parent") and entry["parent"] not in chain:
        chain.append(entry["parent"])
        entry = find_entry(entry["parent"])
    return chain


def data_summary(task: str) -> str:
    try:
        card = summarize_card(get_engine().describe_dataset(task))
    except Exception as exc:  # the card is nice to have; registering must not depend on it
        return f"(dataset card unavailable: {exc})"
    keys = ("task_type", "sensor", "n_bands", "n_classes", "image_size", "split_sizes")
    return "\n".join(f"- {k}: {card[k]}" for k in keys)


def model_card(mv: ModelVersion, entry: dict, chain: list[str], notes: str) -> str:
    return f"""# {mv.task} {mv.version}

- Run: `{mv.run_id}`
- Validation {entry["metric_name"]}: **{mv.val_score:.4f}**
- Training time: {entry["minutes"]:.1f} min, {entry["epochs"]} epochs
- Lineage (newest first): {" <- ".join(f"`{r}`" for r in chain)}
- Git commit: `{mv.git_commit}`
- Registered: {mv.created_at.isoformat()}

## Data

{data_summary(mv.task)}

## Config

```json
{json.dumps(mv.config.model_dump(mode="json"), indent=2)}
```

## Notes

{notes.strip() or "_none_"}
"""


@tool
def register_model(run_id: str, notes: str = "") -> dict:
    """Copy a successful run into registry/<task>/vN with a model card. Returns the version.

    Args:
        run_id: The run to register (normally the best by validation score).
        notes: Why this run was chosen and anything a reader of the model card should know.
    """
    entry = find_entry(run_id)
    if entry is None:
        return unknown_run(run_id)
    if entry["status"] != "ok":
        return error(
            f"run {run_id} has status {entry['status']!r}; only successful runs can be registered",
            "Register the best 'ok' run from leaderboard().",
        )
    task = entry["task"]
    versions = list_versions(task)
    for v in versions:
        existing = load_version(task, v)
        if existing and existing.run_id == run_id:
            return error(
                f"run {run_id} is already registered as {v}", "No need to register it again."
            )

    version = f"v{int(versions[-1][1:]) + 1 if versions else 1}"
    src, dst = run_key(task, run_id), version_key(task, version)
    n_files = storage().copy_dir(src, dst)
    mv = ModelVersion(
        task=task,
        version=version,
        run_id=run_id,
        config=TrainConfig.model_validate_json(storage().read_text(f"{src}/config.json")),
        val_score=entry["val_score"],
        created_at=now_utc(),
        git_commit=storage().read_text(f"{src}/git_commit").strip(),
        artifact_path=storage().uri(dst),
    )
    storage().write_text(f"{dst}/model_version.json", mv.model_dump_json(indent=2))
    storage().write_text(f"{dst}/model_card.md", model_card(mv, entry, lineage(run_id), notes))
    return {
        "task": task,
        "version": version,
        "run_id": run_id,
        "val_score": mv.val_score,
        "files_copied": n_files,
        "model_card": f"{dst}/model_card.md",
    }
