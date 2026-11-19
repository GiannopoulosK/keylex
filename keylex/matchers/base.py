"""Matcher base class: output formats, batch helpers and highlighting."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from ..highlight import highlight as _highlight
from ..keyword_list import KeywordList
from ..match import Match
from .enums import OutputFormat


class BaseMatcher(ABC):
    """Subclass and implement :meth:`find` (and ``keyword_list``) to build a custom matcher."""

    keyword_list: KeywordList

    @abstractmethod
    def find(self, text: str) -> list[Match]:
        """Return all matches in ``text``, sorted by position."""

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _validate_text(text: object) -> None:
        if not isinstance(text, str):
            raise TypeError(f"`text` must be a str, got {type(text).__name__}")

    @staticmethod
    def _validate_output_format(output: object) -> str:
        return OutputFormat.coerce(output, "output format")

    def __call__(self, text: str) -> list[Match]:
        return self.find(text)

    def match(self, text: str, output: str | OutputFormat = OutputFormat.MATCHES) -> Any:
        """Match ``text`` and return the result in the requested ``output`` format.

        Formats: ``"matches"`` (list of :class:`Match`, default), ``"detailed"`` (list of dicts),
        ``"keywords"`` (unique names, in order of appearance), ``"boolean"`` / ``"count"``
        (dict over *all* keywords), ``"categories"`` (counts per category), ``"score"`` (float).
        """
        fmt = self._validate_output_format(output)
        return self.format_output(self.find(text), fmt)

    def match_many(self, texts: Iterable[str], output: str | OutputFormat = OutputFormat.MATCHES) -> Iterator[Any]:
        """Lazily match many texts: ``for result in matcher.match_many(docs, "keywords"): ...``"""
        fmt = self._validate_output_format(output)
        for text in texts:
            yield self.format_output(self.find(text), fmt)

    def format_output(self, matches: list[Match], output: str | OutputFormat) -> Any:
        fmt = self._validate_output_format(output)
        if fmt == "matches":
            return matches
        if fmt == "detailed":
            return [m.to_dict() for m in matches]
        if fmt == "keywords":
            return list(dict.fromkeys(m.keyword for m in matches))
        if fmt == "boolean":
            found = {m.keyword for m in matches}
            return {name: name in found for name in self.keyword_list.names}
        if fmt == "count":
            counts = dict.fromkeys(self.keyword_list.names, 0)
            for m in matches:
                counts[m.keyword] = counts.get(m.keyword, 0) + 1
            return counts
        if fmt == "categories":
            cats: dict[str | None, int] = {}
            for m in matches:
                cats[m.category] = cats.get(m.category, 0) + 1
            return cats
        # score
        return sum(m.weight * m.score / 100.0 for m in matches)

    def contains(self, text: str) -> bool:
        """``True`` if any keyword occurs in ``text``."""
        return bool(self.find(text))

    def count(self, text: str) -> dict[str, int]:
        return self.match(text, OutputFormat.COUNT)  # type: ignore[no-any-return]

    def score(self, text: str) -> float:
        """Sum of matched keyword weights (fuzzy matches are scaled by their score)."""
        return float(self.match(text, OutputFormat.SCORE))

    def highlight(
        self, text: str, style: str | Callable[[Match], str] = "brackets", *, escape: bool | None = None
    ) -> str:
        """Return ``text`` with matches marked up (see :func:`keylex.highlight`; ``escape=None``
        HTML-escapes automatically when the template looks like HTML)."""
        return _highlight(text, self.find(text), style, escape=escape)
