"""Audit trail of an agent session: one JSON line per event, plus a readable journal.

    <GOTRAIN_HOME>/logs/<task>/<session_id>.jsonl          machine-readable, one event per line
    <GOTRAIN_HOME>/logs/<task>/<session_id>_journal.md     the same story for humans

Every line has: ts (UTC), session_id, task, phase, event. Events:
    session_start / session_end   written by the CLI
    assistant_text                the agent's reasoning, as written
    tool_start                    tool name and args, before it runs
    tool_call                     name, args, result summary, duration_s, status, error
    trial                         one per trial a tool ran (its "trials" rows): run_id, rung,
                                  parent, seed_group, status, val_score, minutes
    model_error                 a failed Bedrock call (it may be retried)

Named audit.py (not logging.py) so it never shadows the stdlib logging module.
"""

import json
import uuid
from typing import Any

from strands.hooks import (
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeToolCallEvent,
    HookProvider,
    HookRegistry,
    MessageAddedEvent,
)

from gotrain.storage import get_storage
from gotrain.tools._common import now_utc

MAX_FIELD_CHARS = 2000  # args and result summaries in the JSONL
MAX_TEXT_CHARS = 8000  # assistant reasoning
MAX_JOURNAL_CHARS = 300  # args and results in the journal


def new_session_id(phase: str) -> str:
    return f"{phase}-{now_utc():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"


def truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[:limit]}... [{len(text) - limit} chars cut]"


def result_text(result: dict) -> str:
    """The text a Strands ToolResult carries (our tools return one JSON text block)."""
    parts = []
    for block in result.get("content", []):
        if "text" in block:
            parts.append(block["text"])
        elif "json" in block:
            parts.append(json.dumps(block["json"], default=str))
    return "\n".join(parts)


def result_error(result: dict, text: str) -> str | None:
    """The tool's own {"error": ...} message, or the text of a failed tool result."""
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    if isinstance(data, dict) and data.get("error"):
        return str(data["error"])
    return text if result.get("status") == "error" else None


class AuditLog(HookProvider):
    """Strands hook that records every tool call and assistant message of one session."""

    def __init__(self, task: str, phase: str, session_id: str | None = None, echo: bool = False):
        self.task = task
        self.phase = phase
        self.session_id = session_id or new_session_id(phase)
        self.echo = echo
        self.jsonl_key = f"logs/{task}/{self.session_id}.jsonl"
        self.journal_key = f"logs/{task}/{self.session_id}_journal.md"
        self._storage = get_storage()

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeToolCallEvent, self._before_tool)
        registry.add_callback(AfterToolCallEvent, self._after_tool)
        registry.add_callback(MessageAddedEvent, self._message_added)
        registry.add_callback(AfterModelCallEvent, self._after_model)

    # ------------------------------------------------------------ writing

    def log(self, event: str, **fields: Any) -> dict:
        record = {
            "ts": now_utc().isoformat(),
            "session_id": self.session_id,
            "task": self.task,
            "phase": self.phase,
            "event": event,
            **fields,
        }
        self._storage.append_text(self.jsonl_key, json.dumps(record, default=str) + "\n")
        return record

    def journal(self, markdown: str) -> None:
        self._storage.append_text(self.journal_key, markdown.rstrip() + "\n\n")
        if self.echo:
            print(markdown.rstrip(), flush=True)

    def start(self, **info: Any) -> None:
        self.log("session_start", **info)
        details = "\n".join(f"- {k}: `{v}`" for k, v in info.items())
        self.journal(f"# {self.task} / {self.phase} / {self.session_id}\n\n{details}")

    def end(self, **info: Any) -> None:
        self.log("session_end", **info)
        details = "\n".join(f"- {k}: `{v}`" for k, v in info.items())
        self.journal(f"## Session end ({now_utc():%Y-%m-%d %H:%M:%S} UTC)\n\n{details}")

    # ------------------------------------------------------------ hook callbacks

    def _before_tool(self, event: BeforeToolCallEvent) -> None:
        tool_use = event.tool_use
        self.log(
            "tool_start",
            tool=tool_use["name"],
            tool_use_id=tool_use["toolUseId"],
            args=truncate(json.dumps(tool_use.get("input", {}), default=str), MAX_FIELD_CHARS),
        )

    def _after_tool(self, event: AfterToolCallEvent) -> None:
        tool_use, result = event.tool_use, event.result
        text = result_text(result)
        error = result_error(result, text)
        if event.exception is not None:
            error = f"{type(event.exception).__name__}: {event.exception}"
        if event.cancel_message:
            error = event.cancel_message
        args = json.dumps(tool_use.get("input", {}), default=str)
        duration = round(event.duration, 3) if event.duration is not None else None
        self.log(
            "tool_call",
            tool=tool_use["name"],
            tool_use_id=tool_use["toolUseId"],
            args=truncate(args, MAX_FIELD_CHARS),
            result=truncate(text, MAX_FIELD_CHARS),
            duration_s=duration,
            status=result.get("status"),
            cancelled=event.cancel_message is not None,
            error=truncate(error, MAX_FIELD_CHARS) if error else None,
        )
        warn = f" **ERROR:** {truncate(error, MAX_JOURNAL_CHARS)}" if error else ""
        self.journal(
            f"-> `{tool_use['name']}` ({duration or 0:.1f} s){warn}\n"
            f"   - args: `{truncate(args, MAX_JOURNAL_CHARS)}`\n"
            f"   - result: `{truncate(text, MAX_JOURNAL_CHARS)}`"
        )
        self._log_trials(tool_use["name"], text)

    def _log_trials(self, tool_name: str, text: str) -> None:
        """One untruncated `trial` event per row of a tool's "trials" list."""
        try:
            trials = json.loads(text).get("trials") or []
        except (ValueError, AttributeError):
            return
        for row in trials:
            self.log("trial", tool=tool_name, **row)
            rung = f"rung {row['rung']} " if row.get("rung") else ""
            parent = f" <- {row['parent']}" if row.get("parent") else ""
            self.journal(
                f"   - trial {rung}`{row.get('run_id')}`{parent}: {row.get('status')}, "
                f"val {row.get('val_score')}, {row.get('minutes')} min"
            )

    def _message_added(self, event: MessageAddedEvent) -> None:
        message = event.message
        if message.get("role") != "assistant":
            return
        for block in message.get("content", []):
            text = block.get("text", "").strip()
            if text:
                self.log("assistant_text", text=truncate(text, MAX_TEXT_CHARS))
                self.journal(f"**Agent:** {text}")

    def _after_model(self, event: AfterModelCallEvent) -> None:
        if event.exception is not None:
            error = f"{type(event.exception).__name__}: {event.exception}"
            self.log("model_error", error=truncate(error, MAX_FIELD_CHARS))
            self.journal(f"**Model error** (may be retried): {truncate(error, MAX_JOURNAL_CHARS)}")
