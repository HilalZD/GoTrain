"""Command-line entry point: run the agent on ONE task and ONE phase.

    gotrain run --task m-eurosat --budget-min 30 --phase tune
    gotrain run --task m-eurosat --phase finalise      (asks you to type the task name)

There is deliberately no "all tasks" or "tune then finalise" mode: the one-time test
evaluation only ever happens when a human asks for it explicitly.
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from gotrain.agent import DEFAULT_MAX_TOOL_CALLS, build_agent, kickoff_message, make_limit
from gotrain.audit import AuditLog
from gotrain.config import get_settings
from gotrain.driver import run_session
from gotrain.engine import get_engine
from gotrain.storage import get_storage
from gotrain.tools._common import git_commit
from gotrain.tools.final_test import lock_key
from gotrain.tools.registry import list_versions
from gotrain.tools.trials import budget_status, init_task_budget
from gotrain.vault import real_test_problem


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="gotrain", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the agent on one task and one phase")
    run.add_argument("--task", required=True, help="GEO-Bench task, e.g. m-eurosat")
    run.add_argument("--phase", choices=["tune", "finalise"], default="tune")
    run.add_argument("--budget-min", type=float, help="total GPU budget for the task (tune)")
    run.add_argument("--max-tool-calls", type=int, help="tool-call limit for this session")
    export = sub.add_parser("export", help="copy results from home(s) into the repo")
    export.add_argument("--out", required=True, type=Path, help="e.g. results/")
    export.add_argument("--home", action="append", type=Path, help="repeatable")
    return parser.parse_args(argv)


def fail(message: str, code: int = 1) -> int:
    print(f"gotrain: {message}", file=sys.stderr)
    return code


def prepare_tune(task: str, budget_min: float | None) -> str | None:
    """Set (or confirm) the task budget. Returns an error message, or None if ready."""
    if budget_min is None:
        return "--budget-min is required for the tune phase"
    out = init_task_budget(task, budget_min)
    if "error" in out:
        return f"{out['error']} ({out['hint']})"
    if out["budget"]["remaining_min"] <= 0:
        return f"the GPU budget for {task} is used up; nothing to tune"
    return None


def prepare_finalise(task: str, ask: Callable[[str], str]) -> str | None:
    """Check finalise can run and get the typed confirmation. Error message, or None."""
    if get_storage().exists(lock_key(task)):
        return f"{task} has already been tested ({lock_key(task)} exists); refusing"
    if problem := real_test_problem(task):
        return f"test vault: {problem}; refusing"
    versions = list_versions(task)
    if not versions:
        return f"no registered versions for {task}; run the tune phase first"
    print(f"Registered versions of {task}: {', '.join(versions)}")
    print("The finalise phase scores ONE version on the TEST set. This can only happen once.")
    try:
        answer = ask(f"Type the task name ({task}) to confirm: ")
    except EOFError:
        answer = ""
    if answer.strip() != task:
        return "confirmation did not match the task name; nothing was run"
    return None


def run(args: argparse.Namespace, ask: Callable[[str], str] = input) -> int:
    try:
        settings = get_settings()
        get_engine().describe_dataset(args.task)
    except ValidationError as exc:
        return fail(f"bad settings:\n{exc}", 2)
    except Exception as exc:
        return fail(f"cannot start on task {args.task!r}: {exc}", 2)

    if args.phase == "tune":
        problem = prepare_tune(args.task, args.budget_min)
    else:
        problem = prepare_finalise(args.task, ask)
    if problem:
        return fail(problem)

    max_calls = args.max_tool_calls or DEFAULT_MAX_TOOL_CALLS[args.phase]
    audit = AuditLog(args.task, args.phase, echo=True)
    audit.start(
        engine=settings.engine,
        home=str(settings.home),
        model=settings.bedrock_model_id,
        git_commit=git_commit(),
        max_tool_calls=max_calls,
        budget=budget_status(args.task),
    )
    limit = make_limit(args.phase, max_calls)
    agent = build_agent(args.task, args.phase, audit=audit, limit=limit)
    kickoff = kickoff_message(args.task, args.phase, max_calls)
    try:
        outcome = run_session(agent, audit, args.task, args.phase, kickoff, limit)
    except Exception as exc:
        audit.end(status="crashed", error=f"{type(exc).__name__}: {exc}")
        return fail(f"agent crashed: {exc} (see {audit.jsonl_key})")
    audit.end(status="finished", budget=budget_status(args.task), **outcome)
    if outcome["registered_by"] == "harness":
        print(
            f"WARNING: the agent did not register a model; the harness auto-registered "
            f"{outcome['version']} (see the journal)."
        )
    print(f"Done. Audit log: {settings.home / audit.jsonl_key}")
    print(f"Journal:   {settings.home / audit.journal_key}")
    return 0


def export(args: argparse.Namespace) -> int:
    from gotrain.export import export_results

    try:
        homes = [h.expanduser() for h in args.home] if args.home else [get_settings().home]
    except ValidationError as exc:
        return fail(f"bad settings:\n{exc}", 2)
    missing = [str(h) for h in homes if not h.is_dir()]
    if missing:
        return fail(f"no such home: {', '.join(missing)}")
    out = export_results(homes, args.out)
    print(f"Exported {out['files']} files from {len(homes)} home(s). Summary: {out['summary']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "run":
        return run(args)
    if args.command == "export":
        return export(args)
    return fail(f"unknown command {args.command!r}", 2)


if __name__ == "__main__":
    sys.exit(main())
