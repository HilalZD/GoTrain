"""GoPilot stand-in: a Strands agent that uses the exported models over MCP (stdio).

    GOTRAIN_HOME=~/gotrain_home python scripts/demo_gopilot.py --task m-brick-kiln

Starts gotrain.mcp.models_server as a subprocess, connects a Strands agent (Bedrock) to it,
and asks "Are there brick kilns in this tile?" about one VALIDATION tile. The agent must call
the model's tool and answer from its result. Runs on CPU: it never touches the GPU, and it
only ever reads the validation split (val_x.npy / val_y.npy of the data cache), never test.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

QUESTIONS = {
    "m-brick-kiln": "Are there brick kilns in this tile?",
    "m-eurosat": "What kind of land cover is in this tile?",
    "m-pv4ger-seg": "Are there solar panels in this image? Roughly how much area do they cover?",
    "m-chesapeake": "What land cover classes are in this image, and how much of each?",
}
SYSTEM_PROMPT = (
    "You are GoPilot, a geospatial assistant. You can call trained models as tools; use the "
    "model that fits the question on the image file you are given, then answer in two or three "
    "sentences using the tool's result (label, confidence, areas). If the result has "
    "needs_review=true, or the tool returns an error, say so."
)


def val_tile(task: str, out_dir: Path, positive: bool = True) -> tuple[Path, int | None]:
    """Save one VALIDATION tile as C x H x W .npy; returns (path, its label or None).

    Classification: the first val tile of the positive (last) class when `positive`, else the
    first of class 0. Reads only the val_* cache files.
    """
    from gotrain.engine.data import cache_root

    folder = cache_root() / task
    x = np.load(folder / "val_x.npy", mmap_mode="r")  # N x H x W x C, raw units, card band order
    y = np.load(folder / "val_y.npy", mmap_mode="r")
    index, label = 0, None
    if y.ndim == 1:  # classification labels
        wanted = int(y.max()) if positive else 0
        index = int(np.flatnonzero(np.asarray(y) == wanted)[0])
        label = int(y[index])
    path = out_dir / f"{task}_val_tile_{index}.npy"
    np.save(path, np.transpose(np.asarray(x[index]), (2, 0, 1)).astype(np.float32))
    return path, label


def fake_tile(task: str, out_dir: Path) -> tuple[Path, int | None]:
    """A synthetic tile for the fake engine (dev machines have no data)."""
    from gotrain.engine import get_engine

    card = get_engine().describe_dataset(task)
    height, width = card.image_size
    rng = np.random.default_rng(0)
    path = out_dir / f"{task}_fake_tile.npy"
    np.save(path, rng.uniform(0, 3000, (len(card.bands), height, width)).astype(np.float32))
    return path, None


def server_params() -> object:
    from mcp import StdioServerParameters

    env = {k: v for k, v in os.environ.items() if k.startswith(("GOTRAIN", "AWS", "GEO_BENCH"))}
    env.update({"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(Path(__file__).parents[1])})
    return StdioServerParameters(
        command=sys.executable, args=["-m", "gotrain.mcp.models_server"], env=env
    )


def run_demo(task: str, tile: Path, model=None, question: str | None = None) -> dict:
    """Ask the question through a Strands agent wired to the MCP server. Returns the evidence."""
    from mcp import stdio_client
    from strands import Agent
    from strands.tools.mcp import MCPClient

    from gotrain.agent import build_model
    from gotrain.config import get_settings

    question = question or QUESTIONS.get(task, "What is in this tile?")
    client = MCPClient(lambda: stdio_client(server_params()))
    with client:
        tools = client.list_tools_sync()
        names = [t.tool_name for t in tools]
        agent = Agent(
            model=model or build_model(get_settings()),
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            callback_handler=None,
        )
        result = agent(f"{question}\nImage file: {tile}")
    called = [
        block["toolUse"]["name"]
        for message in agent.messages
        for block in message.get("content", [])
        if "toolUse" in block
    ]
    return {"tools": names, "called": called, "answer": str(result)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", default="m-brick-kiln")
    parser.add_argument("--negative", action="store_true", help="use a tile of class 0 instead")
    parser.add_argument("--question")
    args = parser.parse_args(argv)
    out_dir = Path(tempfile.mkdtemp(prefix="gopilot_demo_"))
    if os.environ.get("GOTRAIN_ENGINE", "fake") == "fake":
        tile, label = fake_tile(args.task, out_dir)
    else:
        tile, label = val_tile(args.task, out_dir, positive=not args.negative)
    print(f"Tile: {tile} (VALIDATION, true class index: {label})")
    out = run_demo(args.task, tile, question=args.question)
    print(f"MCP tools: {out['tools']}\nCalled: {out['called']}\n\nGoPilot says:\n{out['answer']}")
    return 0 if out["called"] else 1  # the agent must have used a model tool


if __name__ == "__main__":
    sys.exit(main())
