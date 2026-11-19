"""Deduplication, context filtering and overlap resolution."""

from __future__ import annotations

from bisect import bisect_left, bisect_right, insort

from ..match import KIND_PRIORITY, Match
from ..text import Token, TokenNormalizer, sentence_spans, word_tokens


class IntervalSet:
    """Non-overlapping half-open intervals with O(log n) overlap checks."""

    __slots__ = ("_items",)

    def __init__(self) -> None:
        self._items: list[tuple[int, int]] = []  # sorted by start (and therefore by end)

    def overlaps(self, start: int, end: int) -> bool:
        items = self._items
        i = bisect_left(items, (start, start))
        if i < len(items) and items[i][0] < end:
            return True
        return i > 0 and items[i - 1][1] > start

    def add(self, start: int, end: int) -> None:
        insort(self._items, (start, end))

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._items)


def _priority(m: Match) -> tuple[int, int, float]:
    return (0 if m.method != "fuzzy" else 1, KIND_PRIORITY.get(m.kind, 9), -m.score)


def dedupe(matches: list[Match]) -> list[Match]:
    """One match per (keyword, span); exact beats fuzzy, keyword beats synonym, etc."""
    best: dict[tuple[str, int, int], Match] = {}
    for m in matches:
        key = (m.keyword, m.start, m.end)
        current = best.get(key)
        if current is None or _priority(m) < _priority(current):
            best[key] = m
    return list(best.values())


def drop_fuzzy_shadowed(matches: list[Match]) -> list[Match]:
    """Remove fuzzy matches overlapping an exact match of the same keyword (O(n log n))."""
    exact_by_kw: dict[str, tuple[list[int], list[int]]] = {}
    for m in sorted((m for m in matches if m.method != "fuzzy"), key=lambda m: m.start):
        starts, max_ends = exact_by_kw.setdefault(m.keyword, ([], []))
        starts.append(m.start)
        max_ends.append(max(m.end, max_ends[-1]) if max_ends else m.end)  # running max of ends

    def shadowed(m: Match) -> bool:
        spans = exact_by_kw.get(m.keyword)
        if spans is None:
            return False
        starts, max_ends = spans
        i = bisect_left(starts, m.end)  # exact matches starting before the fuzzy one ends
        return i > 0 and max_ends[i - 1] > m.start

    return [m for m in matches if m.method != "fuzzy" or not shadowed(m)]


def _contains_seq(haystack: list[str], needle: list[str]) -> bool:
    n = len(needle)
    if n == 0 or n > len(haystack):
        return False
    first = needle[0]
    return any(haystack[i] == first and haystack[i : i + n] == needle for i in range(len(haystack) - n + 1))


def context_filter(
    text: str,
    matches: list[Match],
    normalizer: TokenNormalizer,
    words: list[Token] | None = None,
    *,
    scope: str = "sentence",
) -> list[Match]:
    """Apply each keyword's ``exclude`` / ``require`` context phrases.

    With ``scope="sentence"`` the context window is clipped to the sentence containing the match.
    """
    if not any(m.keyword_obj is not None and (m.keyword_obj.exclude or m.keyword_obj.require) for m in matches):
        return matches
    if words is None:
        words = word_tokens(text)
    starts = [w.start for w in words]
    ends = [w.end for w in words]
    folded = [normalizer(w.text) for w in words]
    phrase_cache: dict[str, list[str]] = {}
    sentences = sentence_spans(text) if scope == "sentence" else []
    sent_starts = [a for a, _ in sentences]

    def phrase(p: str) -> list[str]:
        if p not in phrase_cache:
            phrase_cache[p] = [normalizer(t.text) for t in word_tokens(p)]
        return phrase_cache[p]

    kept: list[Match] = []
    for m in matches:
        kw = m.keyword_obj
        if kw is None or not (kw.exclude or kw.require):
            kept.append(m)
            continue
        w = kw.context_window
        left_end = bisect_right(ends, m.start)  # words fully before the match
        right_start = bisect_left(starts, m.end)  # words starting after the match
        lo, hi = max(0, left_end - w), right_start + w
        if sentences:
            k = bisect_right(sent_starts, m.start) - 1
            if k >= 0:
                s_start, s_end = sentences[k]
                lo = max(lo, bisect_left(starts, s_start))
                hi = min(hi, bisect_right(ends, s_end))
        left = folded[lo:left_end] if lo < left_end else []
        right = folded[right_start:hi] if right_start < hi else []
        if any(_contains_seq(left, phrase(p)) or _contains_seq(right, phrase(p)) for p in kw.exclude):
            continue
        if kw.require and not any(
            _contains_seq(left, phrase(p)) or _contains_seq(right, phrase(p)) for p in kw.require
        ):
            continue
        kept.append(m)
    return kept


def resolve_overlaps(matches: list[Match], strategy: str) -> list[Match]:
    """Resolve overlaps between different spans. Identical spans (e.g. two keywords sharing a
    surface form) are always kept together."""
    if strategy == "all" or len(matches) < 2:
        return matches
    by_span: dict[tuple[int, int], list[Match]] = {}
    for m in matches:
        by_span.setdefault(m.span, []).append(m)

    def rank(span: tuple[int, int]) -> tuple[float, ...]:
        group = by_span[span]
        best_score = max(m.score for m in group)
        length = span[1] - span[0]
        if strategy == "leftmost":
            return (span[0], -length, -best_score)
        return (-length, -best_score, span[0])

    chosen = IntervalSet()
    for span in sorted(by_span, key=rank):
        if span[1] > span[0] and chosen.overlaps(*span):
            continue
        chosen.add(*span)
    return [m for span in chosen for m in by_span[span]]
