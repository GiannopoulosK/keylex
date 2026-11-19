"""The :class:`Match` result object."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .keyword import Keyword

__all__ = ["KIND_PRIORITY", "Match"]

#: Lower is preferred when two terms of one keyword match the same span.
KIND_PRIORITY = {"keyword": 0, "synonym": 1, "acronym": 2, "variant": 3, "pattern": 4}


@dataclass(frozen=True)
class Match:
    """A single keyword occurrence in a text.

    Attributes:
        keyword: Canonical keyword name.
        start, end: Character offsets into the original text (``text[start:end]``).
        text: The surface text that matched, exactly as it appears in the document.
        term: The configured term that produced the match (e.g. the synonym ``"AI"``).
        kind: ``"keyword"``, ``"synonym"``, ``"acronym"``, ``"variant"`` or ``"pattern"``.
        method: ``"exact"``, ``"fuzzy"`` or ``"pattern"``.
        score: Similarity in ``[0, 100]``; always ``100`` for exact matches.
        category: The keyword's category, if any.
        keyword_obj: The :class:`~keylex.Keyword` itself (not part of equality).
    """

    keyword: str
    start: int
    end: int
    text: str
    term: str
    kind: str = "keyword"
    method: str = "exact"
    score: float = 100.0
    category: str | None = None
    keyword_obj: Keyword | None = field(default=None, compare=False, repr=False)

    @property
    def span(self) -> tuple[int, int]:
        return (self.start, self.end)

    def __len__(self) -> int:
        return self.end - self.start

    def overlaps(self, other: Match) -> bool:
        return self.start < other.end and other.start < self.end

    @property
    def weight(self) -> float:
        return self.keyword_obj.weight if self.keyword_obj is not None else 1.0

    def sort_key(self) -> tuple[int, int, str]:
        return (self.start, -self.end, self.keyword)

    def to_dict(self) -> dict[str, Any]:
        return {
            "keyword": self.keyword,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "term": self.term,
            "kind": self.kind,
            "method": self.method,
            "score": round(self.score, 2),
            "category": self.category,
        }

    def __str__(self) -> str:
        extra = "" if self.method == "exact" else f", {self.method} {self.score:.0f}"
        return f"{self.text!r} -> {self.keyword} @{self.start}:{self.end}{extra}"
