"""Matcher base class: output formats, batch helpers and highlighting."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from typing import Any, Literal, overload

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

    # The overloads give every output format its precise return type for type checkers.
    @overload
    def match(self, text: str, output: Literal["matches", OutputFormat.MATCHES] = ...) -> list[Match]: ...
    @overload
    def match(self, text: str, output: Literal["detailed", OutputFormat.DETAILED]) -> list[dict[str, Any]]: ...
    @overload
    def match(self, text: str, output: Literal["keywords", OutputFormat.KEYWORDS]) -> list[str]: ...
    @overload
    def match(self, text: str, output: Literal["boolean", OutputFormat.BOOLEAN]) -> dict[str, bool]: ...
    @overload
    def match(self, text: str, output: Literal["count", OutputFormat.COUNT]) -> dict[str, int]: ...
    @overload
    def match(self, text: str, output: Literal["categories", OutputFormat.CATEGORIES]) -> dict[str | None, int]: ...
    @overload
    def match(self, text: str, output: Literal["score", OutputFormat.SCORE]) -> float: ...
    @overload
    def match(self, text: str, output: str | OutputFormat) -> Any: ...
    def match(self, text: str, output: str | OutputFormat = OutputFormat.MATCHES) -> Any:
        """Match ``text`` and return the result in the requested ``output`` format.

        Formats: ``"matches"`` (list of :class:`Match`, default), ``"detailed"`` (list of dicts),
        ``"keywords"`` (unique names, in order of appearance), ``"boolean"`` / ``"count"``
        (dict over *all* keywords), ``"categories"`` (counts per category), ``"score"`` (float).
        """
        fmt = self._validate_output_format(output)
        return self.format_output(self.find(text), fmt)

    @overload
    def match_many(
        self, texts: Iterable[str], output: Literal["matches", OutputFormat.MATCHES] = ...
    ) -> Iterator[list[Match]]: ...
    @overload
    def match_many(
        self, texts: Iterable[str], output: Literal["keywords", OutputFormat.KEYWORDS]
    ) -> Iterator[list[str]]: ...
    @overload
    def match_many(
        self, texts: Iterable[str], output: Literal["count", OutputFormat.COUNT]
    ) -> Iterator[dict[str, int]]: ...
    @overload
    def match_many(self, texts: Iterable[str], output: Literal["score", OutputFormat.SCORE]) -> Iterator[float]: ...
    @overload
    def match_many(self, texts: Iterable[str], output: str | OutputFormat) -> Iterator[Any]: ...
    def match_many(self, texts: Iterable[str], output: str | OutputFormat = OutputFormat.MATCHES) -> Iterator[Any]:
        """Lazily match many texts: ``for result in matcher.match_many(docs, "keywords"): ...``

        The output format is validated immediately, not on the first iteration.
        """
        fmt = self._validate_output_format(output)
        return (self.format_output(self.find(text), fmt) for text in texts)

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
        """``{keyword: number of matches}`` for every keyword."""
        return self.match(text, "count")

    def score(self, text: str) -> float:
        """Sum of matched keyword weights (fuzzy matches are scaled by their score)."""
        return float(self.match(text, "score"))

    def highlight(
        self, text: str, style: str | Callable[[Match], str] = "brackets", *, escape: bool | None = None
    ) -> str:
        """Return ``text`` with matches marked up (see :func:`keylex.highlight`; ``escape=None``
        HTML-escapes automatically when the template looks like HTML)."""
        return _highlight(text, self.find(text), style, escape=escape)
