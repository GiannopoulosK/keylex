"""Every ```python block in the README must run, and its ``# value`` comments must hold."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"
BLOCKS = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S)


def _checked_lines(code: str) -> list[tuple[str, str]]:
    """Pairs of (expression, expected literal) for lines like ``expr   # ['a', 'b']``."""
    pairs = []
    for line in code.splitlines():
        m = re.match(r"^(\S.*?)\s+#\s+(.+?)(?:\s+\(.*\))?$", line)
        if not m or "=" in m.group(1).split("(")[0] or m.group(1).startswith(("for ", "print", "from", "import")):
            continue
        expected = m.group(2).split("  ")[0]
        try:
            ast.literal_eval(expected)
            ast.parse(m.group(1), mode="eval")
        except (ValueError, SyntaxError):
            continue
        pairs.append((m.group(1), expected))
    return pairs


@pytest.mark.parametrize("index", range(len(BLOCKS)))
def test_readme_block(index: int):
    code = BLOCKS[index]
    if "import spacy" in code:
        pytest.importorskip("spacy")
    namespace: dict[str, object] = {}
    exec(compile(code, f"README block {index}", "exec"), namespace)
    for expr, expected in _checked_lines(code):
        assert eval(expr, namespace) == ast.literal_eval(expected), expr


def test_readme_has_blocks():
    assert len(BLOCKS) >= 10
