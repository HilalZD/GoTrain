"""Published GEO-Bench reference scores per task (gotrain/knowledge/targets.json).

Scores only: the schema forbids any other key, so no hyperparameter or recipe can be loaded
from this file (humans never pick hyperparameters for agent runs). References are TEST
scores from papers; the agent itself only ever sees validation scores.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TARGETS_PATH = Path(__file__).parent / "targets.json"


class Reference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float = Field(ge=0.0, le=1.0, description="Metric as a fraction, e.g. 0.9816.")
    split: Literal["test", "val"]
    model: str = Field(min_length=1, description="Which model reached it (a name, no recipe).")
    source: str = Field(min_length=1)


class TaskTargets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: Literal["accuracy", "miou", "f1"]
    references: list[Reference] = Field(min_length=1)
    human_baseline: Reference | None = None


class Targets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    about: str
    tasks: dict[str, TaskTargets]


def load_targets(path: Path = TARGETS_PATH) -> Targets:
    return Targets.model_validate(json.loads(path.read_text(encoding="utf-8")))


TARGETS_NOTE = (
    "References are TEST scores reported in papers; you only see VALIDATION scores, which "
    "are not directly comparable (val is usually a little optimistic after tuning on it). "
    "Use them as a rough bar, not an exact goal."
)


def targets_for(task: str) -> dict | None:
    """The task's targets as the agent sees them, or None if none are recorded."""
    targets = load_targets().tasks.get(task)
    if targets is None:
        return None
    best = max(targets.references, key=lambda r: r.value)
    out = {"metric": targets.metric, "best_reference": best.value, **targets.model_dump()}
    return out | {"note": TARGETS_NOTE}
