"""Agent tools: plain typed functions that never raise and always report the task budget.

Each takes/returns contract models or JSON-serializable dicts. The agent (Strands)
and the MCP server wrap these same functions.
"""

from gotrain.tools.diagnose import diagnose
from gotrain.tools.final_test import final_test
from gotrain.tools.inspect import inspect_dataset
from gotrain.tools.memory import read_lessons, write_lesson
from gotrain.tools.registry import register_model
from gotrain.tools.search import confirm_seeds, run_halving
from gotrain.tools.trials import init_task_budget, leaderboard, promote, run_trials

AGENT_TOOLS = [
    inspect_dataset,
    run_trials,
    run_halving,
    promote,
    confirm_seeds,
    leaderboard,
    diagnose,
    register_model,
    final_test,
    write_lesson,
    read_lessons,
]

__all__ = [
    "AGENT_TOOLS",
    "confirm_seeds",
    "diagnose",
    "final_test",
    "init_task_budget",
    "inspect_dataset",
    "leaderboard",
    "promote",
    "read_lessons",
    "register_model",
    "run_halving",
    "run_trials",
    "write_lesson",
]
