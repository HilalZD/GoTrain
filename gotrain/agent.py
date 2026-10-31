"""The GoTrain agent: a Strands agent on Bedrock (Claude Sonnet) wired to one phase's tools.

Two phases, two different toolsets:
    tune      explore, train, diagnose and register; final_test is NOT available
    finalise  score the chosen version on TEST once, then write lessons

Do not import strands_tools: it is preinstalled on SageMaker but incompatible.
"""

import inspect
import json
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

from botocore.config import Config as BotoConfig
from strands import Agent, ModelRetryStrategy
from strands import tool as strands_tool
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.hooks import (
    AfterToolCallEvent,
    AfterToolsEvent,
    BeforeToolCallEvent,
    HookProvider,
    HookRegistry,
    MessageAddedEvent,
)
from strands.models import BedrockModel, Model
from strands.tools.executors import SequentialToolExecutor

from gotrain.audit import AuditLog, result_error, result_text
from gotrain.config import Settings, get_settings
from gotrain.contracts import TrainConfig
from gotrain.tools import (
    confirm_seeds,
    diagnose,
    evaluate_candidates,
    final_test,
    inspect_dataset,
    leaderboard,
    promote,
    read_lessons,
    register_model,
    run_halving,
    run_trials,
    write_lesson,
)
from gotrain.tools.candidates import MAX_CALLS_PER_SESSION, SELECTION_RULES
from gotrain.tools.diagnose import OBSERVATION_MARKERS, diagnose_for_model
from gotrain.tools.registry import list_versions, load_version
from gotrain.tools.trials import budget_status

Phase = Literal["tune", "finalise"]
PROMPTS_DIR = Path(__file__).parent / "prompts"
PHASE_TOOLS = {
    "tune": [
        inspect_dataset,
        run_halving,
        run_trials,
        promote,
        confirm_seeds,
        evaluate_candidates,
        leaderboard,
        diagnose,
        register_model,
        read_lessons,
    ],
    "finalise": [leaderboard, final_test, write_lesson, read_lessons],
}
# After the tool-call limit, only these may still run (a few times) so the agent can wrap up.
WRAP_UP_TOOLS = {"tune": {"leaderboard", "register_model"}, "finalise": {"write_lesson"}}
WRAP_UP_GRACE = 3
DEFAULT_MAX_TOOL_CALLS = {"tune": 60, "finalise": 12}
WINDOW_SIZE = 40  # messages kept; the first (task + budget) message is pinned


class ToolCallLimit(HookProvider):
    """Caps tool calls per session so the agent can never loop forever.

    Past `max_calls`, only wrap-up tools run (at most `grace` more calls); everything else
    is cancelled with an explanation. At `max_calls + grace` the agent loop is ended.
    """

    def __init__(self, max_calls: int, wrap_up_tools: set[str], grace: int = WRAP_UP_GRACE):
        self.max_calls = max_calls
        self.wrap_up_tools = wrap_up_tools
        self.grace = grace
        self.calls = 0

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeToolCallEvent, self._before_tool)
        registry.add_callback(AfterToolsEvent, self._after_tools)

    @property
    def hard_limit(self) -> int:
        return self.max_calls + self.grace

    @property
    def exhausted(self) -> bool:
        """True once the loop has been (or would be) ended for this session."""
        return self.calls >= self.hard_limit

    def _before_tool(self, event: BeforeToolCallEvent) -> None:
        self.calls += 1
        if self.calls <= self.max_calls:
            return
        allowed = sorted(self.wrap_up_tools)
        if event.tool_use["name"] in self.wrap_up_tools and self.calls <= self.hard_limit:
            return
        event.cancel_tool = (
            f"Tool-call limit of {self.max_calls} reached for this session. Only {allowed} "
            "may still be called. Wrap up now and finish."
        )

    def _after_tools(self, event: AfterToolsEvent) -> None:
        if self.calls >= self.hard_limit:
            event.end_turn = f"Stopped: tool-call limit ({self.hard_limit}) reached."


class SelectionQuota(HookProvider):
    """At most MAX_CALLS_PER_SESSION successful evaluate_candidates calls per session: every
    extra comparison on the same validation set makes the winner's score more optimistic.
    Calls that returned an error (nothing evaluated) don't count."""

    TOOL = "evaluate_candidates"

    def __init__(self, audit: AuditLog, max_calls: int = MAX_CALLS_PER_SESSION):
        self.audit = audit
        self.max_calls = max_calls
        self.used = 0

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeToolCallEvent, self._before_tool)
        registry.add_callback(AfterToolCallEvent, self._after_tool)

    def _before_tool(self, event: BeforeToolCallEvent) -> None:
        if event.tool_use["name"] != self.TOOL or self.used < self.max_calls:
            return
        event.cancel_tool = (
            f"Blocked: evaluate_candidates was already used {self.used} times this session "
            f"(the limit). {SELECTION_RULES} Register your choice from the earlier results."
        )
        self.audit.log("selection_quota_enforced", used=self.used, max=self.max_calls)

    def _after_tool(self, event: AfterToolCallEvent) -> None:
        if event.tool_use["name"] != self.TOOL or event.cancel_message:
            return
        result = event.result
        if result.get("status") == "success" and not result_error(result, result_text(result)):
            self.used += 1


class ObservationGate(HookProvider):
    """After a successful diagnose, cancel every tool call until the agent has written both
    "VISUAL OBSERVATION" and "HYPOTHESIS".

    Accepted in an assistant message (Strands adds the message before running its tools,
    so text plus tool call in one response works) or in the next tool call's own arguments,
    e.g. a config's reason. Every block and every release is an audit event.
    """

    def __init__(self, audit: AuditLog):
        self.audit = audit
        self.pending_run: str | None = None  # the diagnosed run awaiting an observation

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(AfterToolCallEvent, self._after_tool)
        registry.add_callback(MessageAddedEvent, self._message_added)
        registry.add_callback(BeforeToolCallEvent, self._before_tool)

    def _after_tool(self, event: AfterToolCallEvent) -> None:
        if event.tool_use["name"] != "diagnose" or event.cancel_message:
            return
        result = event.result
        if result.get("status") == "success" and not result_error(result, result_text(result)):
            self.pending_run = event.tool_use.get("input", {}).get("run_id", "?")

    def _message_added(self, event: MessageAddedEvent) -> None:
        message = event.message
        if self.pending_run is None or message.get("role") != "assistant":
            return
        text = "\n".join(block.get("text", "") for block in message.get("content", []))
        if has_observation(text):
            self._release("message")

    def _before_tool(self, event: BeforeToolCallEvent) -> None:
        if self.pending_run is None:
            return
        if has_observation(json.dumps(event.tool_use.get("input", {}))):
            self._release("tool_args")
            return
        name = event.tool_use["name"]
        event.cancel_tool = (
            f"Blocked: you diagnosed {self.pending_run} but have not written your observation. "
            "Write VISUAL OBSERVATION and HYPOTHESIS as plain text in your message AND include "
            "your next tool call in the SAME response; do not end your turn.\n"
            "VISUAL OBSERVATION: what the worst predictions have in common (from the error "
            "grid, or from the per-class scores and confusions if no image was shown).\n"
            "HYPOTHESIS: the fix that observation suggests.\n"
            f"Then call {name} again in that same response; the next config's reason must "
            "cite them."
        )
        self.audit.log(
            "observation_enforced",
            blocked_tool=name,
            tool_use_id=event.tool_use["toolUseId"],
            diagnosed_run=self.pending_run,
        )

    def _release(self, source: str) -> None:
        self.audit.log("observation_written", diagnosed_run=self.pending_run, source=source)
        self.pending_run = None


def has_observation(text: str) -> bool:
    upper = text.upper()
    return all(marker in upper for marker in OBSERVATION_MARKERS)


def load_prompt(phase: Phase) -> str:
    return (PROMPTS_DIR / f"{phase}.md").read_text(encoding="utf-8")


def config_reference() -> str:
    """One line per TrainConfig field, generated so it never drifts from the contract."""
    lines = []
    for name, field in TrainConfig.model_fields.items():
        line = f"- {name}: {field.description or ''}"
        options = get_args(field.annotation) if get_origin(field.annotation) is Literal else ()
        if options:
            line += f" Options: {', '.join(map(str, options))}."
        if not field.is_required():
            default = field.get_default(call_default_factory=True)
            line += f" Default: {default!r}."
        lines.append(line)
    return "\n".join(lines)


CONFIG_TOOLS = (run_trials, run_halving)  # tools whose `configs` argument needs the field list
# The only tool allowed to show the model an image: its adapter adds the error grid.
IMAGE_ADAPTERS = {diagnose: diagnose_for_model}


def to_strands_tool(fn: Any) -> Any:
    """Expose a plain GoTrain tool to Strands; config-taking tools also get the field list."""
    if fn in IMAGE_ADAPTERS:
        return strands_tool(IMAGE_ADAPTERS[fn])
    if fn not in CONFIG_TOOLS:
        return strands_tool(fn)
    summary = inspect.cleandoc(fn.__doc__).split("\n\nArgs:")[0]
    description = f"{summary}\n\nEach config is an object with these fields:\n{config_reference()}"
    return strands_tool(fn, description=description)


def build_model(settings: Settings) -> BedrockModel:
    boto_config = BotoConfig(
        retries={"max_attempts": 10, "mode": "adaptive"},
        connect_timeout=30,
        read_timeout=600,
    )
    return BedrockModel(
        model_id=settings.bedrock_model_id,
        region_name=settings.aws_region,
        boto_client_config=boto_config,
    )


def make_limit(phase: Phase, max_tool_calls: int | None = None) -> ToolCallLimit:
    return ToolCallLimit(max_tool_calls or DEFAULT_MAX_TOOL_CALLS[phase], WRAP_UP_TOOLS[phase])


def build_agent(
    task: str,
    phase: Phase,
    *,
    audit: AuditLog | None = None,
    max_tool_calls: int | None = None,
    model: Model | None = None,
    limit: ToolCallLimit | None = None,
) -> Agent:
    """A ready-to-run agent for one task and phase.

    `model` is injectable for tests; pass your own `limit` (see make_limit) to inspect it
    after the session, as the driver does.
    """
    if phase not in PHASE_TOOLS:
        raise ValueError(f"phase must be one of {sorted(PHASE_TOOLS)}, got {phase!r}")
    limit = limit or make_limit(phase, max_tool_calls)
    audit = audit or AuditLog(task, phase)
    hooks: list[HookProvider] = [audit, limit]
    if phase == "tune":  # diagnose and evaluate_candidates only exist in tune
        hooks += [ObservationGate(audit), SelectionQuota(audit)]
    return Agent(
        model=model or build_model(get_settings()),
        tools=[to_strands_tool(fn) for fn in PHASE_TOOLS[phase]],
        system_prompt=load_prompt(phase),
        hooks=hooks,
        conversation_manager=SlidingWindowConversationManager(window_size=WINDOW_SIZE, pin_first=1),
        tool_executor=SequentialToolExecutor(),  # never two trainings on one GPU at once
        retry_strategy=ModelRetryStrategy(max_attempts=8, initial_delay=4, max_delay=120),
        callback_handler=None,  # the audit journal is the readable output
        name=f"gotrain-{phase}",
    )


def kickoff_message(task: str, phase: Phase, max_tool_calls: int | None = None) -> str:
    """The first user message: the task, the budget, and (finalise) the registered versions."""
    calls = max_tool_calls or DEFAULT_MAX_TOOL_CALLS[phase]
    budget = budget_status(task)
    lines = [
        f"Task: {task}",
        f"Phase: {phase}",
        f"GPU budget: {budget['used_min']:g} min used, {budget['remaining_min']:g} min remaining.",
        f"Tool-call limit for this session: {calls}.",
    ]
    if phase == "tune":
        lines.append("Start with your BUDGET PLAN, then inspect the dataset and read lessons.")
    else:
        lines.append("Registered versions (validation scores):")
        for version in list_versions(task):
            mv = load_version(task, version)
            model = f"run {mv.run_id}" if len(mv.members) == 1 else f"ensemble {mv.members}"
            tta = ", TTA" if mv.tta else ""
            lines.append(f"- {version}: val_score={mv.val_score:.4f} ({model}{tta})")
    return "\n".join(lines)
