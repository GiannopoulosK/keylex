"""Render matches inside the original text."""

from __future__ import annotations

import dataclasses
import html
from collections.abc import Callable, Iterable

from .match import Match

__all__ = ["STYLES", "highlight"]

STYLES: dict[str, str] = {
    "brackets": "[{text}]",
    "labels": "[{text}]({keyword})",
    "markdown": "**{text}**",
    "ansi": "\033[1;4;33m{text}\033[0m",
    "html": '<mark data-keyword="{keyword}">{text}</mark>',
}


def _identity(s: str) -> str:
    return s


def highlight(
    text: str,
    matches: Iterable[Match],
    style: str | Callable[[Match], str] = "brackets",
    *,
    escape: bool | None = None,
) -> str:
    """Return ``text`` with every match wrapped according to ``style``.

    ``style`` is a preset name (``"brackets"``, ``"labels"``, ``"markdown"``, ``"ansi"``,
    ``"html"``), a format string using ``{text}``, ``{keyword}``, ``{term}``, ``{category}``,
    ``{score}``, or a callable ``Match -> str``. Overlapping matches are rendered once (the
    earliest/longest wins).

    ``escape`` HTML-escapes the document text (inside and between matches) and the values filled
    into the template. ``None`` (default) escapes when the template looks like HTML, i.e. contains
    ``"<"`` (so the ``"html"`` preset and templates such as ``"<mark>{text}</mark>"``); callables are
    escaped only with ``escape=True``, in which case they receive a copy of each :class:`Match` whose
    ``text``, ``keyword``, ``term`` and ``category`` are already escaped.
    """
    template: str | None = None
    if not callable(style):
        template = STYLES.get(style, style)
        if "{" not in template:
            raise ValueError(f"Unknown highlight style {style!r}; use one of {sorted(STYLES)} or a template")
    if escape is None:
        escape = template is not None and "<" in template
    esc: Callable[[str], str] = html.escape if escape else _identity

    if template is None:
        assert callable(style)
        user_render = style

        def render(m: Match) -> str:
            if escape:
                m = dataclasses.replace(
                    m,
                    text=esc(m.text),
                    keyword=esc(m.keyword),
                    term=esc(m.term),
                    category=esc(m.category) if m.category is not None else None,
                )
            return user_render(m)

    else:
        fmt = template

        def render(m: Match) -> str:
            return fmt.format(
                text=esc(m.text),
                keyword=esc(m.keyword),
                term=esc(m.term),
                category=esc(m.category or ""),
                score=round(m.score),
            )

    out: list[str] = []
    pos = 0
    for m in sorted(matches, key=lambda m: (m.start, -m.end)):
        if m.start < pos:
            continue  # overlaps something already rendered
        out.append(esc(text[pos : m.start]))
        out.append(render(m))
        pos = m.end
    out.append(esc(text[pos:]))
    return "".join(out)
