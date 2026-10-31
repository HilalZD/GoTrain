"""Guards for the permanent rules in CLAUDE.md."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "gotrain"
FINAL_TEST = PKG / "tools" / "final_test.py"
# Any way of asking for the test split: split="test", split='test', evaluate(..., "test")
TEST_SPLIT = re.compile(r"""split\s*=\s*["']test["']|evaluate\([^)]*["']test["']""")


def code_files() -> list[Path]:
    return [p for d in ("gotrain", "scripts") for p in (REPO / d).rglob("*.py")]


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
