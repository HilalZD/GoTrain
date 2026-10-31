"""Guards for the permanent rules in CLAUDE.md."""

import re
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "gotrain"
TEST_SPLIT = re.compile(r"""split\s*=\s*["']test["']|evaluate\([^)]*["']test["']""")
ALLOWED = {PKG / "tools" / "final_test.py"}


def test_only_final_test_mentions_the_test_split():
    offenders = [
        str(path.relative_to(PKG))
        for path in PKG.rglob("*.py")
        if path not in ALLOWED and TEST_SPLIT.search(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"only tools/final_test.py may request the test split: {offenders}"


def test_no_module_shadows_stdlib_logging():
    assert not (PKG / "logging.py").exists()
