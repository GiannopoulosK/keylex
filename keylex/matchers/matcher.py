"""The main :class:`KeywordMatcher`."""

from __future__ import annotations

import dataclasses
import threading
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


@dataclasses.dataclass(frozen=True, slots=True)
class _Compiled:
    """Everything ``find`` needs, built together and swapped in atomically."""

    keywords: KeywordList
    signature: object
    options: tuple[Any, ...]
    unicode_form: str | None
    overlaps: str
    context_scope: str
    normalizer: TokenNormalizer
    exact: ExactEngine
    fuzzy: FuzzyEngine | None


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

    #: Public options; changing one after construction recompiles the index on the next call.
    _OPTIONS = (
        "match_type",
        "threshold",
        "scorer",
        "min_fuzzy_length",
        "overlaps",
        "case_sensitive",
        "ignore_accents",
        "stemmer",
        "unicode_form",
        "acronym_case",
        "context_scope",
    )

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
        if fuzzy_algo is not None:
            warnings.warn("`fuzzy_algo` is deprecated; use `scorer=`", DeprecationWarning, stacklevel=2)
            scorer = fuzzy_algo
        self.match_type: str = match_type
        self.threshold: float = threshold
        self.scorer: str | Scorer = scorer
        self.min_fuzzy_length = min_fuzzy_length
        self.overlaps: str = overlaps
        self.case_sensitive = case_sensitive
        self.ignore_accents = ignore_accents
        self.stemmer = stemmer
        self.unicode_form = unicode_form
        self.acronym_case = acronym_case
        self.context_scope = context_scope
        self._lock = threading.Lock()
        self._state: _Compiled | None = None
        self._mutations_seen = -1
        self._validate_options()

    def _validate_options(self) -> tuple[Any, ...]:
        """Check (and canonicalise) the public options; return them as a snapshot tuple."""
        self.match_type = MatchType.coerce(self.match_type, "match_type")
        t = self.threshold
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 <= t <= 100:
            raise ValueError(f"threshold must be a number between 0 and 100, got {t!r}")
        self.threshold = float(t)
        resolve_scorer(self.scorer)
        m = self.min_fuzzy_length
        if isinstance(m, bool) or not isinstance(m, int) or m < 1:
            raise ValueError("min_fuzzy_length must be a positive integer")
        self.overlaps = Overlaps.coerce(self.overlaps, "overlaps strategy")
        self.case_sensitive = bool(self.case_sensitive)
        self.ignore_accents = bool(self.ignore_accents)
        resolve_stemmer(self.stemmer)
        if self.unicode_form not in (None, "NFC", "NFKC"):
            raise ValueError(f"unicode_form must be 'NFKC', 'NFC' or None, got {self.unicode_form!r}")
        if self.acronym_case not in ("strict", "title", "ignore"):
            raise ValueError(f"acronym_case must be 'strict', 'title' or 'ignore', got {self.acronym_case!r}")
        if self.context_scope not in ("sentence", "text"):
            raise ValueError(f"context_scope must be 'sentence' or 'text', got {self.context_scope!r}")
        return self._current_options()

    def _current_options(self) -> tuple[Any, ...]:
        return tuple(getattr(self, name) for name in self._OPTIONS)

    def configure(self, **options: Any) -> KeywordMatcher:
        """Change options in place: ``matcher.configure(threshold=80, stemmer="light")``.

        Assigning the attributes directly works too; either way the index is rebuilt on the next
        call. Invalid values raise and leave the matcher unchanged.
        """
        unknown = set(options) - set(self._OPTIONS)
        if unknown:
            raise TypeError(f"Unknown option(s): {', '.join(sorted(unknown))}; available: {', '.join(self._OPTIONS)}")
        old = self.options()
        for name, value in options.items():
            setattr(self, name, value)
        try:
            self._validate_options()
        except (TypeError, ValueError):
            for name, value in old.items():
                setattr(self, name, value)
            raise
        return self

    def options(self) -> dict[str, Any]:
        """The current options (``KeywordMatcher(keywords, **matcher.options())`` makes a twin)."""
        return {name: getattr(self, name) for name in self._OPTIONS}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        del state["_lock"]  # locks can't be pickled; the index is rebuilt lazily after unpickling
        state["_state"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()
        self._mutations_seen = -1

    # legacy attribute
    @property
    def fuzzy_algo(self) -> str | Scorer:
        return self.scorer

    def compile(self, force: bool = False) -> KeywordMatcher:
        """(Re)build the index. Called automatically when the keywords or the options change.

        Thread-safe: concurrent ``find`` calls always see one complete, consistent index.
        """
        self._compiled(force)
        return self

    def _compiled(self, force: bool = False) -> _Compiled:
        state = self._state
        seen = _MUTATIONS[0]
        if (
            not force
            and state is not None
            and seen == self._mutations_seen
            and state.keywords is self.keyword_list
            and state.options == self._current_options()
        ):  # nothing anywhere changed: O(1)
            return state
        with self._lock:
            options = self._validate_options()
            sig = self.keyword_list.signature()
            state = self._state
            if (
                not force
                and state is not None
                and state.keywords is self.keyword_list
                and state.signature == sig
                and state.options == options
            ):  # something changed, but not our keywords or options
                self._mutations_seen = seen
                return state
            terms = resolve_terms(
                self.keyword_list,
                case_sensitive=self.case_sensitive,
                min_fuzzy_length=self.min_fuzzy_length,
                unicode_form=self.unicode_form,
                acronym_case=self.acronym_case,
            )
            normalizer = TokenNormalizer(self.ignore_accents, resolve_stemmer(self.stemmer))
            fuzzy = None
            if self.match_type == "fuzzy":
                fuzzy = FuzzyEngine(terms, normalizer, resolve_scorer(self.scorer), self.threshold)
            state = _Compiled(
                keywords=self.keyword_list,
                signature=sig,
                options=options,
                unicode_form=self.unicode_form,
                overlaps=self.overlaps,
                context_scope=self.context_scope,
                normalizer=normalizer,
                exact=ExactEngine(terms, normalizer),
                fuzzy=fuzzy,
            )
            self._state = state  # one assignment: readers never see a half-built index
            self._mutations_seen = seen
            return state

    def find(self, text: str) -> list[Match]:
        self._validate_text(text)
        state = self._compiled()
        original = text
        text, offsets = normalize_text(text, state.unicode_form)
        matches = state.exact.find(text, tokenize(text))
        words = None
        if state.fuzzy is not None:
            words = word_tokens(text)
            matches = drop_fuzzy_shadowed(matches + state.fuzzy.find(text, words))
        matches = dedupe(matches)
        matches = context_filter(text, matches, state.normalizer, words, scope=state.context_scope)
        if offsets is not None:
            remapped = []
            for m in matches:
                start, end = offsets.span(m.start, m.end)
                remapped.append(dataclasses.replace(m, start=start, end=end, text=original[start:end]))
            matches = remapped
        elif text is not original:  # same offsets, but report the original surface text
            matches = [dataclasses.replace(m, text=original[m.start : m.end]) for m in matches]
        matches = resolve_overlaps(matches, state.overlaps)
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
