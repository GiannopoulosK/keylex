"""String enums for matcher options; every option also accepts the plain string value."""

from __future__ import annotations

from enum import Enum


class _ValuesEnum(str, Enum):
    @classmethod
    def list(cls) -> list[str]:
        return [e.value for e in cls]

    @classmethod
    def coerce(cls, value: object, what: str) -> str:
        if isinstance(value, cls):
            return str(value.value)
        if isinstance(value, str):
            lowered = value.lower()
            for member in cls:
                if member.value == lowered:
                    return str(member.value)
        raise ValueError(f"Invalid {what}: {value!r}. Supported: {', '.join(cls.list())}")

    def __str__(self) -> str:
        return str(self.value)


class OutputFormat(_ValuesEnum):
    """What :meth:`KeywordMatcher.match` returns."""

    MATCHES = "matches"  #: list[Match] (default)
    DETAILED = "detailed"  #: list[dict] (JSON-friendly Match.to_dict())
    KEYWORDS = "keywords"  #: unique keyword names in order of first appearance
    BOOLEAN = "boolean"  #: {keyword: bool} for every keyword in the list
    COUNT = "count"  #: {keyword: int} for every keyword in the list
    CATEGORIES = "categories"  #: {category: int}
    SCORE = "score"  #: float, sum of keyword weights (x fuzzy score / 100)


class MatchType(_ValuesEnum):
    """How :class:`KeywordMatcher` searches: exact only, or exact plus typo-tolerant."""

    EXACT = "exact"
    FUZZY = "fuzzy"  #: exact + windowed fuzzy matching


class Overlaps(_ValuesEnum):
    """How overlapping matches of *different* spans are resolved."""

    LONGEST = "longest"  #: keep the longest, drop what it overlaps (default)
    LEFTMOST = "leftmost"  #: scan left to right, longest at each position wins
    ALL = "all"  #: keep everything (nested and overlapping)
