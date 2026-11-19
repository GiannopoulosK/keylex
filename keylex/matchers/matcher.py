"""The main :class:`KeywordMatcher`."""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from ..keyword import _MUTATIONS, Keyword
from ..keyword_list import KeywordList
from ..match import Match
from ..text import TokenNormalizer, normalize_text, resolve_stemmer, tokenize, word_tokens
from ._compile import resolve_terms
from ._postprocess import context_filter, dedupe, drop_fuzzy_shadowed, resolve_overlaps
from .base import BaseMatcher
from .enums import MatchType, Overlaps
from .exact import ExactEngine
from .fuzzy import FuzzyEngine, Scorer, resolve_scorer

__all__ = ["BaseMatcher", "ExactMatcher", "FuzzyMatcher", "KeywordMatcher", "Matcher"]

KeywordsInput = KeywordList | Iterable[Keyword | str | Mapping[str, Any]] | Mapping[str, Any] | str


def as_keyword_list(keywords: KeywordsInput) -> KeywordList:
    """Coerce lists of strings / Keywords / dicts (or a dict) into a :class:`KeywordList`."""
    if isinstance(keywords, KeywordList):
        return keywords
    if isinstance(keywords, (str, Keyword)):
        return KeywordList([keywords])
    if isinstance(keywords, Mapping):
        return KeywordList.from_dict(keywords)
    if isinstance(keywords, Iterable):
        return KeywordList(keywords)
    raise TypeError(f"keywords must be a KeywordList, a list or a dict, got {type(keywords).__name__}")


class KeywordMatcher(BaseMatcher):
    """Find keywords (and their synonyms, acronyms, variants and patterns) in text.

    Example::

        matcher = KeywordMatcher(["machine learning", Keyword("United States", acronyms="US")])
        matcher.find("ML in the U.S.")                 # -> [Match(...)]
        matcher.match(text, "count")                   # -> {"machine learning": 0, ...}

    Args:
        keywords: A :class:`KeywordList`, or anything it can be built from (list of strings,
            Keywords or dicts; a dict). A ``KeywordList`` is used *live*: later changes to it or
            its keywords are picked up automatically.
        match_type: ``"exact"`` (default) or ``"fuzzy"`` (exact + typo-tolerant windowed matching).
        threshold: Minimum fuzzy similarity (0-100).
        fuzzy_algo: Deprecated alias of ``scorer``.
        scorer: RapidFuzz scorer name (``"ratio"``, ``"partial_ratio"``, ``"token_sort_ratio"``,
            ``"token_set_ratio"``, ``"WRatio"``, ``"QRatio"``) or a callable.
        min_fuzzy_length: Terms shorter than this are never fuzzy-matched.
        overlaps: ``"longest"`` (default), ``"leftmost"`` or ``"all"``.
        case_sensitive: Default for terms that don't set it (acronyms are case-sensitive anyway).
        ignore_accents: Treat ``"cafe"`` and ``"café"`` as equal.
        stemmer: ``"light"`` (English plurals), any ``str -> str`` callable (e.g. a Porter stemmer),
            or ``None``. Applied to both text and keywords so inflections match.
        unicode_form: Unicode normalisation applied to text and terms before matching
            (``"NFKC"`` default: decomposed accents, full-width letters and ligatures match their
            plain forms; ``"NFC"``; or ``None`` to disable). Offsets always refer to the original text.
        acronym_case: How acronym-like terms (``NASA``, ``G7``) are matched when they don't set
            ``case_sensitive``: ``"strict"`` (default, exact spelling only), ``"title"`` (also
            ``Nato``) or ``"ignore"`` (any case).
        context_scope: Where ``exclude`` / ``require`` context words are looked for:
            ``"sentence"`` (default, the window never crosses a sentence boundary) or ``"text"``.
    """

    def __init__(
        self,
        keywords: KeywordsInput,
        match_type: str | MatchType = "exact",
        threshold: float = 85,
        fuzzy_algo: str | None = None,
        *,
        scorer: str | Scorer = "ratio",
        min_fuzzy_length: int = 4,
        overlaps: str | Overlaps = "longest",
        case_sensitive: bool = False,
        ignore_accents: bool = False,
        stemmer: str | Callable[[str], str] | None = None,
        unicode_form: str | None = "NFKC",
        acronym_case: str = "strict",
        context_scope: str = "sentence",
    ) -> None:
        self.keyword_list = as_keyword_list(keywords)
        self.match_type = MatchType.coerce(match_type, "match_type")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 100:
            raise ValueError(f"threshold must be a number between 0 and 100, got {threshold!r}")
        self.threshold = float(threshold)
        if fuzzy_algo is not None:
            warnings.warn("`fuzzy_algo` is deprecated; use `scorer=`", DeprecationWarning, stacklevel=2)
            scorer = fuzzy_algo
        self._scorer = resolve_scorer(scorer)
        self.scorer = scorer
        if not isinstance(min_fuzzy_length, int) or min_fuzzy_length < 1:
            raise ValueError("min_fuzzy_length must be a positive integer")
        self.min_fuzzy_length = min_fuzzy_length
        self.overlaps = Overlaps.coerce(overlaps, "overlaps strategy")
        self.case_sensitive = bool(case_sensitive)
        self.ignore_accents = bool(ignore_accents)
        self._stemmer = resolve_stemmer(stemmer)
        self.stemmer = stemmer
        if unicode_form not in (None, "NFC", "NFKC"):
            raise ValueError(f"unicode_form must be 'NFKC', 'NFC' or None, got {unicode_form!r}")
        self.unicode_form = unicode_form
        if acronym_case not in ("strict", "title", "ignore"):
            raise ValueError(f"acronym_case must be 'strict', 'title' or 'ignore', got {acronym_case!r}")
        self.acronym_case = acronym_case
        if context_scope not in ("sentence", "text"):
            raise ValueError(f"context_scope must be 'sentence' or 'text', got {context_scope!r}")
        self.context_scope = context_scope
        self._signature: object = None
        self._mutations_seen = -1
        self._compiled_list: KeywordList | None = None
        self._exact: ExactEngine | None = None
        self._fuzzy: FuzzyEngine | None = None
        self._normalizer = TokenNormalizer(self.ignore_accents, self._stemmer)

    # legacy attribute
    @property
    def fuzzy_algo(self) -> str | Scorer:
        return self.scorer

    def compile(self, force: bool = False) -> KeywordMatcher:
        """(Re)build the matching index. Called automatically when the keywords change."""
        seen = _MUTATIONS[0]
        if not force and self._exact is not None and self._compiled_list is self.keyword_list:
            if seen == self._mutations_seen:  # nothing anywhere changed: O(1)
                return self
            sig = self.keyword_list.signature()
            if sig == self._signature:  # something changed, but not our keywords
                self._mutations_seen = seen
                return self
        else:
            sig = self.keyword_list.signature()
        terms = resolve_terms(
            self.keyword_list,
            case_sensitive=self.case_sensitive,
            min_fuzzy_length=self.min_fuzzy_length,
            unicode_form=self.unicode_form,
            acronym_case=self.acronym_case,
        )
        self._normalizer = TokenNormalizer(self.ignore_accents, self._stemmer)
        self._exact = ExactEngine(terms, self._normalizer)
        self._fuzzy = (
            FuzzyEngine(terms, self._normalizer, self._scorer, self.threshold) if self.match_type == "fuzzy" else None
        )
        self._signature = sig
        self._mutations_seen = seen
        self._compiled_list = self.keyword_list
        return self

    def find(self, text: str) -> list[Match]:
        self._validate_text(text)
        self.compile()
        assert self._exact is not None
        original = text
        text, offsets = normalize_text(text, self.unicode_form)
        matches = self._exact.find(text, tokenize(text))
        words = None
        if self._fuzzy is not None:
            words = word_tokens(text)
            matches = drop_fuzzy_shadowed(matches + self._fuzzy.find(text, words))
        matches = dedupe(matches)
        matches = context_filter(text, matches, self._normalizer, words, scope=self.context_scope)
        if offsets is not None:
            remapped = []
            for m in matches:
                start, end = offsets.span(m.start, m.end)
                remapped.append(dataclasses.replace(m, start=start, end=end, text=original[start:end]))
            matches = remapped
        matches = resolve_overlaps(matches, self.overlaps)
        return sorted(matches, key=Match.sort_key)

    def __repr__(self) -> str:
        return (
            f"KeywordMatcher({len(self.keyword_list)} keywords, match_type={self.match_type!r}, "
            f"overlaps={self.overlaps!r})"
        )


class ExactMatcher(KeywordMatcher):
    """:class:`KeywordMatcher` with ``match_type="exact"``."""

    def __init__(self, keywords: KeywordsInput, **kwargs: Any) -> None:
        super().__init__(keywords, "exact", **kwargs)


class FuzzyMatcher(KeywordMatcher):
    """:class:`KeywordMatcher` with ``match_type="fuzzy"`` (exact + typo-tolerant)."""

    def __init__(self, keywords: KeywordsInput, threshold: float = 85, **kwargs: Any) -> None:
        super().__init__(keywords, "fuzzy", threshold, **kwargs)


#: Backwards-compatible name of the old intermediate base class.
Matcher = BaseMatcher
