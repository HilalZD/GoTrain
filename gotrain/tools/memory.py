"""Tools: write and read cross-task lessons in memory/lessons.md.

Each lesson is one dated, tagged entry:

    ## 2026-10-31 m-eurosat
    tags: sentinel-2, classification
    SSL4EO weights beat imagenet by ~3 points; rot90 helps.
"""

import re

from gotrain.config import get_settings
from gotrain.tools._common import error, now_utc, storage, tool

LESSONS_KEY = "memory/lessons.md"
ISOLATED_LESSONS_KEY = "memory/lessons_isolated.md"  # GOTRAIN_MEMORY=empty (memory ablation)
MAX_LESSON_CHARS = 1500
MAX_RETURNED = 20
ENTRY = re.compile(
    r"^## (?P<date>\d{4}-\d{2}-\d{2}) (?P<task>\S+)\n"
    r"tags: (?P<tags>[^\n]*)\n"
    r"(?P<text>.*?)(?=^## |\Z)",
    re.MULTILINE | re.DOTALL,
)


def lessons_key() -> str:
    return ISOLATED_LESSONS_KEY if get_settings().memory == "empty" else LESSONS_KEY


def normalize_tags(tags: list[str]) -> list[str]:
    return sorted({t.strip().lower() for t in tags if t.strip()})


def all_lessons() -> list[dict]:
    """Every lesson in the file, oldest first."""
    if not storage().exists(lessons_key()):
        return []
    return [
        {
            "date": m["date"],
            "task": m["task"],
            "tags": normalize_tags(m["tags"].split(",")),
            "lesson": m["text"].strip(),
        }
        for m in ENTRY.finditer(storage().read_text(lessons_key()))
    ]


def matching_lessons(tags: list[str]) -> list[dict]:
    """Newest-first lessons sharing at least one tag (all lessons if tags is empty)."""
    wanted = set(normalize_tags(tags))
    found = [x for x in all_lessons() if not wanted or wanted & set(x["tags"])]
    return found[::-1][:MAX_RETURNED]


@tool
def write_lesson(task: str, lesson: str, tags: list[str]) -> dict:
    """Append a dated, tagged lesson to memory/lessons.md.

    Args:
        task: The task the lesson was learned on.
        lesson: One concrete, reusable finding with its evidence (1-3 sentences).
        tags: e.g. sensor, task_type and resolution: ["sentinel-2", "classification", "64px"].
    """
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        return error("tags must be a list of strings", 'e.g. ["sentinel-2", "classification"]')
    if get_settings().rehearsal:  # a rehearsal must never leave a lesson that looks real
        lesson, tags = f"REHEARSAL (val): {lesson}", [*tags, "rehearsal"]
    clean_tags = normalize_tags(tags)
    if not clean_tags or any("," in t for t in clean_tags):
        return error(
            "give at least one tag, without commas",
            "Use the sensor and task_type from inspect_dataset, e.g. ['sentinel-2'].",
        )
    text = "\n".join(line.lstrip("#").rstrip() for line in str(lesson).strip().splitlines())
    if not text.strip():
        return error("lesson is empty", "State one concrete, reusable finding in 1-3 sentences.")
    if len(text) > MAX_LESSON_CHARS:
        return error(
            f"lesson is {len(text)} chars; the limit is {MAX_LESSON_CHARS}",
            "Keep only the finding and the evidence (scores), not the whole history.",
        )
    entry = f"\n## {now_utc():%Y-%m-%d} {task}\ntags: {', '.join(clean_tags)}\n{text}\n"
    storage().append_text(lessons_key(), entry)
    return {"task": task, "saved": True, "tags": clean_tags, "n_lessons": len(all_lessons())}


@tool
def read_lessons(tags: list[str], task: str | None = None) -> dict:
    """Lessons sharing any of `tags`, newest first. Pass `task` to get its budget back.

    Args:
        tags: Tags to match (any overlap), e.g. ["sentinel-2"]; [] returns all lessons.
        task: Optional current task name, only used to report its budget.
    """
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        return error("tags must be a list of strings", 'e.g. ["sentinel-2"], or [] for all')
    lessons = matching_lessons(tags)
    return {"tags": normalize_tags(tags), "n_found": len(lessons), "lessons": lessons}
