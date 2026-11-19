"""Windowed fuzzy matching.

The text is split into word tokens; every window of ``n-1 .. n+1`` tokens (``n`` = number of
words in the term) is compared with the term using a RapidFuzz scorer. The best
non-overlapping windows above the threshold become matches with exact character offsets.
Identical windows are scored once, so repetitive text is cheap. Windows never cross a sentence
boundary or clause punctuation (``,;:()`` ...), so "the machine. Learning" is not "machine learning".
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from rapidfuzz import fuzz, process

from ..match import Match
from ..text import Token, TokenNormalizer, sentence_spans, word_tokens
from ._compile import CompiledTerm
from ._postprocess import IntervalSet

Scorer = Callable[..., float]

SCORERS: dict[str, Scorer] = {
    "ratio": fuzz.ratio,
    "partial_ratio": fuzz.partial_ratio,
    "token_sort_ratio": fuzz.token_sort_ratio,
    "token_set_ratio": fuzz.token_set_ratio,
    "WRatio": fuzz.WRatio,
    "QRatio": fuzz.QRatio,
}


def resolve_scorer(scorer: str | Scorer) -> Scorer:
    if callable(scorer):
        return scorer
    try:
        return SCORERS[scorer]
    except KeyError:
        raise ValueError(f"Invalid fuzzy scorer {scorer!r}. Available: {', '.join(SCORERS)}") from None


_np: Any
try:  # numpy enables RapidFuzz's batched, multi-threaded cdist (imported dynamically: it's optional)
    _np = importlib.import_module("numpy")
except ImportError:  # pragma: no cover - numpy is optional
    _np = None

_CHUNK = 256  # terms scored per cdist call (bounds memory to CHUNK x windows scores)
#: Punctuation between two words that a fuzzy window may not span (sentence ends are found with
#: :func:`sentence_spans`, which knows about abbreviations such as "Dr.").
_CLAUSE_CHARS = frozenset(',;:!?()[]{}"|/\\\u201c\u201d\u2026')


def _segments(text: str, words: list[Token], clauses: bool) -> list[int]:
    """A segment id per word; a window is valid only if its first and last words share one."""
    seg = [0] * len(words)
    sentences = sentence_spans(text)
    k = 0
    current = 0
    for i, w in enumerate(words):
        if i:
            while k + 1 < len(sentences) and sentences[k + 1][0] <= w.start:
                k += 1
                current += 1
            if clauses and not _CLAUSE_CHARS.isdisjoint(text[words[i - 1].end : w.start]):
                current += 1
        seg[i] = current
    return seg


def _window_sizes(n: int) -> list[int]:
    return [s for s in (n, n + 1, n - 1) if s >= 1]


class FuzzyEngine:
    def __init__(
        self,
        terms: list[CompiledTerm],
        normalizer: TokenNormalizer,
        scorer: Scorer,
        threshold: float,
    ) -> None:
        self.normalizer = normalizer
        self.scorer = scorer
        self.threshold = threshold
        self.terms: list[tuple[CompiledTerm, str, int]] = []
        #: terms that contain clause punctuation themselves ("Smith, John") may span it
        self.spans_clauses: list[bool] = []
        for ct in terms:
            if not ct.fuzzy or ct.term.kind == "pattern":
                continue
            text = ct.text or ct.term.text
            toks = [t.text for t in word_tokens(text)]
            if not toks:
                continue
            norm = normalizer.exact if ct.case_sensitive else normalizer
            self.terms.append((ct, " ".join(norm(t) for t in toks), len(toks)))
            self.spans_clauses.append(not _CLAUSE_CHARS.isdisjoint(text))

    def _score(self, idxs: list[int], keys: list[str]) -> list[tuple[int, str, float]]:
        """Score terms ``idxs`` against window strings ``keys``; return hits above the threshold."""
        hits: list[tuple[int, str, float]] = []
        if not keys:
            return hits
        if _np is not None and self.threshold > 0 and len(idxs) * len(keys) > 2_000:
            for lo in range(0, len(idxs), _CHUNK):
                chunk = idxs[lo : lo + _CHUNK]
                scores: Any = process.cdist(
                    [self.terms[i][1] for i in chunk],
                    keys,
                    scorer=self.scorer,
                    score_cutoff=self.threshold,
                    dtype=_np.float64,
                    workers=-1,
                )
                rows, cols = _np.nonzero(scores >= self.threshold)
                for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
                    hits.append((chunk[r], keys[c], float(scores[r, c])))
            return hits
        for i in idxs:
            results: Any = process.extract(
                self.terms[i][1], keys, scorer=self.scorer, score_cutoff=self.threshold, limit=None
            )
            hits.extend((i, choice, float(score)) for choice, score, _ in results)
        return hits

    def find(self, text: str, words: list[Token] | None = None) -> list[Match]:
        if not self.terms:
            return []
        if words is None:
            words = word_tokens(text)
        if not words:
            return []
        n_words = len(words)
        folded = [self.normalizer(w.text) for w in words]
        raw: list[str] | None = None
        segments: dict[bool, list[int]] = {}
        # (case_sensitive, size, may span clauses) -> {window string: [start indices]}
        windows: dict[tuple[bool, int, bool], dict[str, list[int]]] = {}
        choices: dict[tuple[bool, int, bool], list[str]] = {}

        def get_windows(cs: bool, size: int, spans_clauses: bool) -> tuple[list[str], dict[str, list[int]]]:
            nonlocal raw
            key = (cs, size, spans_clauses)
            if key not in windows:
                if cs and raw is None:
                    raw = [self.normalizer.exact(w.text) for w in words]
                src = raw if cs else folded
                assert src is not None
                seg = None
                if size > 1:
                    if spans_clauses not in segments:
                        segments[spans_clauses] = _segments(text, words, clauses=not spans_clauses)
                    seg = segments[spans_clauses]
                table: dict[str, list[int]] = {}
                for i in range(n_words - size + 1):
                    if seg is None or seg[i] == seg[i + size - 1]:
                        table.setdefault(" ".join(src[i : i + size]), []).append(i)
                windows[key] = table
                choices[key] = list(table)
            return choices[key], windows[key]

        # term index -> [(score, size penalty, start, size)]
        found: dict[int, list[tuple[float, int, int, int]]] = {}
        groups: dict[tuple[bool, int, bool], list[int]] = {}
        for idx, (ct, _, n) in enumerate(self.terms):
            for size in _window_sizes(n):
                if size <= n_words:
                    groups.setdefault((ct.case_sensitive, size, self.spans_clauses[idx]), []).append(idx)
        for (cs, size, spans_clauses), idxs in groups.items():
            keys, table = get_windows(cs, size, spans_clauses)
            for idx, choice, score in self._score(idxs, keys):
                n = self.terms[idx][2]
                bucket = found.setdefault(idx, [])
                for start in table[choice]:
                    bucket.append((score, abs(size - n), start, size))

        matches: list[Match] = []
        for idx, (ct, _, _) in enumerate(self.terms):
            candidates = found.get(idx)
            if not candidates:
                continue
            candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
            taken = IntervalSet()
            for score, _, start, size in candidates:
                end = start + size
                if taken.overlaps(start, end):
                    continue
                taken.add(start, end)
                cstart, cend = words[start].start, words[end - 1].end
                kw = ct.keyword
                matches.append(
                    Match(
                        keyword=kw.name,
                        start=cstart,
                        end=cend,
                        text=text[cstart:cend],
                        term=ct.term.text,
                        kind=ct.term.kind,
                        method="fuzzy",
                        score=score,
                        category=kw.category,
                        keyword_obj=kw,
                    )
                )
        return matches
