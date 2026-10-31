"""The GoTrain agent: a Strands agent on Bedrock (Claude Sonnet) wired to one phase's tools.

Two phases, two different toolsets:
    tune      explore, train, diagnose and register; final_test is NOT available
    finalise  score the chosen version on TEST once, then write lessons

Do not import strands_tools: it is preinstalled on SageMaker but incompatible.
"""

import inspect
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

from botocore.config import Config as BotoConfig
from strands import Agent, ModelRetryStrategy
from strands import tool as strands_tool
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.hooks import AfterToolsEvent, BeforeToolCallEvent, HookProvider, HookRegistry
from strands.models import BedrockModel, Model
from strands.tools.executors import SequentialToolExecutor

from gotrain.audit import AuditLog
from gotrain.config import Settings, get_settings
from gotrain.contracts import TrainConfig
from gotrain.tools import (
    confirm_seeds,
    diagnose,
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


def to_strands_tool(fn: Any) -> Any:
    """Expose a plain GoTrain tool to Strands; config-taking tools also get the field list."""
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


def build_agent(
    task: str,
    phase: Phase,
    *,
    audit: AuditLog | None = None,
    max_tool_calls: int | None = None,
    model: Model | None = None,
) -> Agent:
    """A ready-to-run agent for one task and phase. `model` is injectable for tests."""
    if phase not in PHASE_TOOLS:
        raise ValueError(f"phase must be one of {sorted(PHASE_TOOLS)}, got {phase!r}")
    limit = ToolCallLimit(max_tool_calls or DEFAULT_MAX_TOOL_CALLS[phase], WRAP_UP_TOOLS[phase])
    return Agent(
        model=model or build_model(get_settings()),
        tools=[to_strands_tool(fn) for fn in PHASE_TOOLS[phase]],
        system_prompt=load_prompt(phase),
        hooks=[audit or AuditLog(task, phase), limit],
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
            lines.append(f"- {version}: val_score={mv.val_score:.4f} (run {mv.run_id})")
    return "\n".join(lines)
