"""Tool: register a finished run as the next model version, registry/<task>/vN/.

A version folder holds the run's config, result, git commit and checkpoint, plus
model_card.md and model_version.json (a ModelVersion).
"""

import json
import re
from pathlib import Path

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


def extra_metric_lines(entry: dict) -> str:
    """Extra engine numbers, labelled as such: shown to readers, never used to pick a model."""
    lines = ""
    if entry.get("miou_excl_nodata") is not None:
        lines += (
            f"- Extra (not the official score, not used for selection): mIoU excluding the "
            f"no-data class {entry['miou_excl_nodata']:.4f}\n"
        )
    if entry.get("stopped_by_budget"):
        lines += "- Stopped by its time budget before max_epochs (the curve may still rise)\n"
    return lines


def describe_model(mv: ModelVersion) -> str:
    """'single run', 'ensemble of 3 runs', '... with TTA'."""
    text = f"ensemble of {len(mv.members)} runs" if len(mv.members) > 1 else "single run"
    return text + (" with TTA" if mv.tta else "")


def members_section(mv: ModelVersion) -> str:
    if len(mv.members) == 1 and not mv.tta:
        return ""
    how = "averaged over 8 flip/rot90 views (TTA)" if mv.tta else "used as is"
    lines = [
        "## Members",
        "",
        f"Model: {describe_model(mv)}. The members' probabilities are averaged and {how}. "
        "The validation score above is the whole model's, measured by evaluate_candidates.",
        "",
    ]
    for rid in mv.members:
        e = find_entry(rid) or {}
        alone = f"{e['val_score']:.4f}" if e.get("val_score") is not None else "?"
        lines.append(f"- `{rid}`: val {alone} alone; {e.get('config', '?')}")
    return "\n".join(lines) + "\n\n"


def model_card(mv: ModelVersion, entry: dict, chain: list[str], notes: str) -> str:
    return f"""# {mv.task} {mv.version}

- Run: `{mv.run_id}`{" (best member)" if len(mv.members) > 1 else ""}
- Model: {describe_model(mv)}
- Validation {entry["metric_name"]}: **{mv.val_score:.4f}**
{extra_metric_lines(entry)}- Training time: {entry["minutes"]:.1f} min, {entry["epochs"]} epochs
- Lineage (newest first): {" <- ".join(f"`{r}`" for r in chain)}
- Git commit: `{mv.git_commit}`
- Registered: {mv.created_at.isoformat()}

{members_section(mv)}## Data

{data_summary(mv.task)}

## Config{" (best member)" if len(mv.members) > 1 else ""}

```json
{json.dumps(mv.config.model_dump(mode="json"), indent=2)}
```

## Notes

{notes.strip() or "_none_"}
"""


def member_key(task: str, version: str, run_id: str, n_members: int) -> str:
    """Where a member's run folder lives in a version: the version itself for a single run."""
    if n_members == 1:
        return version_key(task, version)
    return f"{version_key(task, version)}/members/{run_id}"


def version_checkpoints(mv: ModelVersion) -> list[Path] | None:
    """The registered copies of every member checkpoint, in member order (None if missing)."""
    paths = []
    for rid in mv.members:
        entry = find_entry(rid)
        if entry is None or not entry.get("checkpoint"):
            return None
        key = member_key(mv.task, mv.version, rid, len(mv.members))
        paths.append(storage().local_path(key) / entry["checkpoint"])
    return paths


@tool
def register_model(
    run_id: str = "", notes: str = "", run_ids: list[str] | None = None, tta: bool = False
) -> dict:
    """Register a single run, or an ensemble of runs, as registry/<task>/vN with a model card.

    A single run without TTA is registered with its training validation score. An ensemble
    (run_ids) or any model with tta=true must first be scored by evaluate_candidates with
    exactly the same members and TTA flag; that validation score is registered.

    Args:
        run_id: The run to register (normally the best by validation score).
        notes: Why this model was chosen and anything a reader of the model card should know.
        run_ids: Instead of run_id: 1-4 runs averaged as an ensemble.
        tta: Use test-time augmentation (8 flip/rot90 views) at test time.
    """
    if bool(run_id) == bool(run_ids):
        return error(
            "pass exactly one of run_id (a single run) or run_ids (an ensemble)",
            'e.g. register_model(run_id="<run>") or register_model(run_ids=["<a>", "<b>"]).',
        )
    from gotrain.tools.candidates import MAX_MEMBERS, find_candidate_result

    members = list(run_ids) if run_ids else [run_id]
    if len(set(members)) != len(members) or not 1 <= len(members) <= MAX_MEMBERS:
        return error(
            f"run_ids must be 1-{MAX_MEMBERS} distinct runs, got {members!r}",
            "List each member once.",
        )
    entries = []
    for rid in members:
        entry = find_entry(rid)
        if entry is None:
            return unknown_run(rid)
        if entry["status"] != "ok":
            return error(
                f"run {rid} has status {entry['status']!r}; only successful runs can be registered",
                "Register the best 'ok' run from leaderboard().",
            )
        entries.append(entry)
    task = entries[0]["task"]
    if any(e["task"] != task for e in entries):
        return error("all members must be runs of the same task", "Check the run ids.")

    if len(members) == 1 and not tta:
        val_score = entries[0]["val_score"]
    else:
        measured = find_candidate_result(task, members, tta)
        if measured is None:
            return error(
                f"members {members} with tta={tta} have no validation score yet",
                "Score exactly this candidate with evaluate_candidates first, then register it.",
            )
        val_score = measured["val_score"]

    versions = list_versions(task)
    for v in versions:
        existing = load_version(task, v)
        if existing and sorted(existing.members) == sorted(members) and existing.tta == tta:
            return error(
                f"this model is already registered as {v}", "No need to register it again."
            )

    entries.sort(key=lambda e: -e["val_score"])  # best member first: it is the primary run
    primary = entries[0]
    version = f"v{int(versions[-1][1:]) + 1 if versions else 1}"
    dst = version_key(task, version)
    n_files = 0
    for e in entries:
        member_dst = member_key(task, version, e["run_id"], len(entries))
        n_files += storage().copy_dir(run_key(task, e["run_id"]), member_dst)
    src = run_key(task, primary["run_id"])
    mv = ModelVersion(
        task=task,
        version=version,
        run_id=primary["run_id"],
        config=TrainConfig.model_validate_json(storage().read_text(f"{src}/config.json")),
        val_score=val_score,
        created_at=now_utc(),
        git_commit=storage().read_text(f"{src}/git_commit").strip(),
        artifact_path=storage().uri(dst),
        members=[e["run_id"] for e in entries],
        tta=tta,
    )
    storage().write_text(f"{dst}/model_version.json", mv.model_dump_json(indent=2))
    card = model_card(mv, primary, lineage(primary["run_id"]), notes)
    storage().write_text(f"{dst}/model_card.md", card)
    return {
        "task": task,
        "version": version,
        "run_id": mv.run_id,
        "members": mv.members,
        "tta": mv.tta,
        "val_score": mv.val_score,
        "files_copied": n_files,
        "model_card": f"{dst}/model_card.md",
    }
