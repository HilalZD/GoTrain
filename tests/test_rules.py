"""Guards for the permanent rules in CLAUDE.md."""

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "gotrain"
FINAL_TEST = PKG / "tools" / "final_test.py"
ENGINE_TRAIN = PKG / "engine" / "train.py"  # Serena's; its evaluate() opens the test data
# Any way of asking for the test split: split="test", split='test', evaluate(..., "test")
TEST_SPLIT = re.compile(r"""split\s*=\s*["']test["']|evaluate\([^)]*["']test["']""")


def code_files() -> list[Path]:
    return [p for d in ("gotrain", "scripts", "scratch") for p in (REPO / d).rglob("*.py")]


def test_only_final_test_requests_the_test_split():
    offenders = [
        str(p.relative_to(REPO))
        for p in code_files()
        if p != FINAL_TEST and TEST_SPLIT.search(p.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"only tools/final_test.py may request the test split: {offenders}"


def test_final_test_is_where_the_test_split_is_requested():
    assert 'split="test"' in FINAL_TEST.read_text(encoding="utf-8")


def test_no_module_shadows_stdlib_logging():
    assert not (PKG / "logging.py").exists()


def test_nothing_imports_strands_tools():
    """strands_tools is preinstalled on SageMaker but incompatible with strands 1.63.0."""
    pattern = re.compile(r"^\s*(import|from)\s+strands_tools\b", re.MULTILINE)
    offenders = [str(p) for p in code_files() if pattern.search(p.read_text(encoding="utf-8"))]
    assert not offenders


def allow_test_violations(source: str, path: Path) -> list[int]:
    """Lines passing `allow_test=` anything but False, outside train.py's evaluate().

    Forwarding the caller's own flag (`allow_test=allow_test`) is plumbing, so it is allowed;
    only gotrain/engine/train.py's top-level evaluate() may decide to open the test data.
    """
    tree = ast.parse(source)
    evaluate_spans = [
        (fn.lineno, fn.end_lineno)
        for fn in tree.body
        if path == ENGINE_TRAIN and isinstance(fn, ast.FunctionDef) and fn.name == "evaluate"
    ]
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            value = kw.value
            if kw.arg != "allow_test":
                continue
            if isinstance(value, ast.Constant) and value.value is False:
                continue
            if isinstance(value, ast.Name) and value.id == "allow_test":
                continue
            if any(start <= node.lineno <= end for start, end in evaluate_spans):
                continue
            lines.append(node.lineno)
    return lines


def test_only_engine_evaluate_opens_the_test_data():
    offenders = {
        str(p.relative_to(REPO)): lines
        for p in code_files()
        if (lines := allow_test_violations(p.read_text(encoding="utf-8"), p))
    }
    assert not offenders, f"only evaluate() in engine/train.py may set allow_test: {offenders}"


def test_allow_test_rule_catches_violations():
    elsewhere = PKG / "engine" / "diagnostics.py"
    assert allow_test_violations("open_split(t, s, allow_test=True)", elsewhere) == [1]
    assert allow_test_violations("ds(c, s, allow_test=s == 'x')", elsewhere) == [1]
    assert allow_test_violations("ds(c, s, allow_test=False)", elsewhere) == []
    assert allow_test_violations("ds(c, s, allow_test=allow_test)", elsewhere) == []
    inside = "def evaluate(t, c, split):\n    return ds(c, split, allow_test=True)\n"
    assert allow_test_violations(inside, ENGINE_TRAIN) == []
    assert allow_test_violations(inside, elsewhere) == [2]  # evaluate() of another file
    nested = "def train():\n    def evaluate():\n        ds(allow_test=True)\n"
    assert allow_test_violations(nested, ENGINE_TRAIN) == [3]  # only the top-level evaluate
