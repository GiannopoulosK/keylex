"""Composable rules evaluated over keyword matches.

Build rules with Python operators or helper functions::

    from keylex.rules import kw, near, same_sentence, category

    rule = (kw("bank") & ~near("bank", "river", within=5)) | category("finance").at_least(3)

or parse them from a string (see :func:`keylex.rules.parse_rule`)::

    parse_rule('bank AND NOT NEAR(bank, river, 5) OR ATLEAST(3, CATEGORY(finance))')

**Occurrences.** Every rule result carries an occurrence count (:attr:`RuleResult.count`), which
is what ``ATLEAST(n, X)`` compares against ``n``:

* keyword / ``CATEGORY`` / ``TAG``: the number of matches (distinct keyword + span);
* ``NEAR`` / ``BEFORE``: the number of disjoint qualifying pairs, paired greedily left to right
  (each match takes part in at most one pair);
* ``a AND b``: the smallest count among the positive (non-``NOT``) operands;
* ``a OR b``: the number of distinct matches (keyword + span) of the operands that count matches
  (keywords, ``CATEGORY``, ``TAG`` and ORs of those), plus the counts of the other operands; one
  match is one occurrence, so ``inflation OR CATEGORY(monetary)`` counts an "inflation" match once;
* ``ANY(n, ...)``: only operands satisfied by *different* matches count towards ``n`` (each counted
  operand needs a match, keyword + span, that no other counted operand uses; identical operands
  count once), so ``ANY(2, ML, "machine learning")`` does not fire on one "ML". The count is the
  ``n``-th largest count among those operands;
* ``SENTENCE`` / ``PARAGRAPH``: the number of qualifying sentences / paragraphs;
* ``NOT x`` and ``ATLEAST``: 1 / the inner count when the rule holds, else 0.

Rules are immutable values: equality, hashing and the per-document evaluation cache are based on
:meth:`Rule.to_dsl`.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..match import Match
from ..text import paragraph_spans, sentence_spans, word_tokens

if TYPE_CHECKING:
    from ..matchers.base import BaseMatcher

__all__ = [
    "AllOf",
    "AnyOf",
    "AtLeast",
    "CategoryRule",
    "Document",
    "KeywordRule",
    "Near",
    "Not",
    "Rule",
    "RuleResult",
    "SameParagraph",
    "SameSentence",
    "TagRule",
    "all_of",
    "any_of",
    "at_least",
    "before",
    "category",
    "kw",
    "near",
    "none_of",
    "not_",
    "same_paragraph",
    "same_sentence",
    "tag",
]


def _fold(name: str) -> str:
    return " ".join(name.split()).casefold()


class _TextInfo:
    """Text-derived positions shared by a document and all of its restricted views."""

    def __init__(self, text: str) -> None:
        self.text = text
        self._word_starts: list[int] | None = None
        self._units: dict[str, list[int]] = {}
        self.spans: dict[tuple[int, int], tuple[int, int]] = {}

    @property
    def word_starts(self) -> list[int]:
        if self._word_starts is None:
            self._word_starts = [w.start for w in word_tokens(self.text)]
        return self._word_starts

    def unit_starts(self, unit: str) -> list[int]:
        starts = self._units.get(unit)
        if starts is None:
            spans = sentence_spans(self.text) if unit == "sentence" else paragraph_spans(self.text)
            starts = self._units[unit] = [s for s, _ in spans] or [0]
        return starts


class Document:
    """A text plus its matches, with lazily computed token / sentence / paragraph positions.

    Sub-rule results are cached per document (see :meth:`evaluate`), so a sub-rule shared by
    several rules of a :class:`~keylex.rules.RuleEngine` is evaluated once per text.

    Args:
        text: The text the matches were found in.
        matches: The keyword matches.
        aliases: Optional ``{term: keyword name}`` map; a keyword rule naming ``term`` matches the
            keyword (used by :class:`~keylex.rules.RuleEngine` for rules naming a synonym/acronym).
    """

    def __init__(self, text: str, matches: Iterable[Match], *, aliases: Mapping[str, str] | None = None) -> None:
        self.text = text
        self.matches: list[Match] = sorted(matches, key=Match.sort_key)
        self.aliases: dict[str, str] = {_fold(k): v for k, v in (aliases or {}).items()}
        self._info = _TextInfo(text)
        self._cache: dict[tuple[type, str], RuleResult] = {}
        self._by_keyword: dict[str, list[Match]] | None = None

    def restrict(self, matches: Iterable[Match]) -> Document:
        """A view of this document that only contains ``matches`` (positions are shared)."""
        view = Document.__new__(Document)
        view.text = self.text
        view.matches = sorted(matches, key=Match.sort_key)
        view.aliases = self.aliases
        view._info = self._info
        view._cache = {}
        view._by_keyword = None
        return view

    def evaluate(self, rule: Rule) -> RuleResult:
        """Evaluate ``rule`` on this document, reusing the result of structurally equal rules."""
        key = (type(rule), rule._key())
        result = self._cache.get(key)
        if result is None:
            result = self._cache[key] = rule.evaluate(self)
        return result

    def keyword_matches(self, name: str) -> list[Match]:
        """Matches of the keyword ``name`` (case-insensitive; aliases resolved)."""
        if self._by_keyword is None:
            index: dict[str, list[Match]] = {}
            for m in self.matches:
                index.setdefault(_fold(m.keyword), []).append(m)
            self._by_keyword = index
        folded = _fold(name)
        hits = self._by_keyword.get(folded, [])
        target = self.aliases.get(folded)
        if target is not None and _fold(target) != folded:
            hits = sorted({*hits, *self._by_keyword.get(_fold(target), [])}, key=Match.sort_key)
        return hits

    @property
    def _word_starts(self) -> list[int]:
        return self._info.word_starts

    def word_index(self, char_pos: int) -> int:
        """Index of the word token containing / preceding ``char_pos``."""
        return max(0, bisect_right(self._info.word_starts, char_pos) - 1)

    def word_span(self, m: Match) -> tuple[int, int]:
        """(first, last) word indices covered by a match."""
        span = self._info.spans.get(m.span)
        if span is None:
            span = self._info.spans[m.span] = (self.word_index(m.start), self.word_index(max(m.start, m.end - 1)))
        return span

    @property
    def _sentence_starts(self) -> list[int]:
        return self._info.unit_starts("sentence")

    @property
    def _paragraph_starts(self) -> list[int]:
        return self._info.unit_starts("paragraph")

    def unit_count(self, unit: str) -> int:
        """Number of sentences (``unit="sentence"``) or paragraphs in the text."""
        return len(self._info.unit_starts(unit))

    def unit_of(self, m: Match, unit: str) -> int:
        """Index of the sentence (``unit="sentence"``) or paragraph that a match starts in."""
        return bisect_right(self._info.unit_starts(unit), m.start) - 1

    def sentence_of(self, m: Match) -> int:
        """Index of the sentence that a match starts in."""
        return self.unit_of(m, "sentence")

    def paragraph_of(self, m: Match) -> int:
        """Index of the paragraph that a match starts in."""
        return self.unit_of(m, "paragraph")


@dataclass(frozen=True)
class RuleResult:
    """Outcome of evaluating a rule; truthy when the rule matched.

    ``count`` is the number of occurrences of the rule (see the module docs); when a custom rule
    leaves it unset, :attr:`occurrences` falls back to the number of distinct matched spans.
    """

    rule: Rule
    matched: bool
    matches: tuple[Match, ...] = ()
    children: tuple[RuleResult, ...] = field(default=(), repr=False)
    detail: str = ""
    count: int | None = None

    def __bool__(self) -> bool:
        return self.matched

    @property
    def occurrences(self) -> int:
        """How many times the rule occurs (what ``ATLEAST`` counts); ``0`` when it did not match."""
        if self.count is not None:
            return self.count
        if not self.matched:
            return 0
        return len({m.span for m in self.matches}) or 1

    @property
    def keywords(self) -> list[str]:
        """Unique names of the keywords behind :attr:`matches`, in order of appearance."""
        return list(dict.fromkeys(m.keyword for m in self.matches))

    def explain(self, indent: int = 0, *, full_label: bool = True) -> str:
        """Human-readable tree showing why the rule did or didn't match.

        The top node shows the whole rule; nested nodes show a short label (operator plus a
        truncated form) so the output grows linearly with the size of the rule.
        """
        mark = "[+]" if self.matched else "[-]"
        label = self.rule.to_dsl() if full_label else self.rule.label()
        info = []
        if self.matches:
            shown = ", ".join(repr(m.text) for m in self.matches[:5])
            more = f" +{len(self.matches) - 5}" if len(self.matches) > 5 else ""
            info.append(f"{len(self.matches)} match(es): {shown}{more}")
        if self.detail:
            info.append(self.detail)
        line = "  " * indent + f"{mark} {label}" + (f"  [{'; '.join(info)}]" if info else "")
        return "\n".join([line, *(c.explain(indent + 1, full_label=False) for c in self.children)])


_RESERVED = {
    "AND",
    "OR",
    "NOT",
    "NEAR",
    "BEFORE",
    "ATLEAST",
    "ANY",
    "ALL",
    "NONE",
    "CATEGORY",
    "TAG",
    "SENTENCE",
    "PARAGRAPH",
    "KW",
}

_BARE_RE = re.compile(r"[^\W\d]\w*")
_LABEL_MAX = 60


def _quote(s: str) -> str:
    """``s`` as a bare word when it re-lexes as exactly that word, otherwise as a JSON string."""
    if _BARE_RE.fullmatch(s) and s.isidentifier() and s.upper() not in _RESERVED:
        return s
    return json.dumps(s, ensure_ascii=False)


def _truncate(s: str, limit: int = _LABEL_MAX) -> str:
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _unique(matches: Iterable[Match]) -> tuple[Match, ...]:
    return tuple(sorted(dict.fromkeys(matches), key=Match.sort_key))


def _check_int(value: object, what: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{what} must be an integer >= {minimum}, got {value!r}")
    return value


def _check_name(name: object, what: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{what} name must be a non-empty string, got {name!r}")
    if any(0xD800 <= ord(ch) <= 0xDFFF for ch in name):
        raise ValueError(f"{what} name contains a lone surrogate code point: {name!r}")
    return " ".join(name.split())


class Rule(ABC):
    """Base class. Combine rules with ``&`` (and), ``|`` (or), ``~`` (not).

    Subclasses implement :meth:`evaluate` and :meth:`to_dsl`. Composite rules should evaluate
    their children with ``doc.evaluate(child)`` so results are cached per document.
    """

    @abstractmethod
    def evaluate(self, doc: Document) -> RuleResult:
        """Evaluate the rule on ``doc``. Call ``doc.evaluate(rule)`` instead to use the cache."""

    @abstractmethod
    def to_dsl(self) -> str:
        """Serialise to the rule language accepted by :func:`parse_rule`."""

    def _key(self) -> str:
        """Cached :meth:`to_dsl` (rules are treated as immutable)."""
        cached: str | None = self.__dict__.get("_dsl_cache")
        if cached is None:
            cached = self.to_dsl()
            self.__dict__["_dsl_cache"] = cached
        return cached

    def keyword_names(self) -> set[str]:
        """Keyword names referenced by this rule (auto-added by :class:`RuleEngine`)."""
        names: set[str] = set()
        for child in self.children():
            names |= child.keyword_names()
        return names

    def children(self) -> Sequence[Rule]:
        """Direct sub-rules (empty for leaf rules such as keywords)."""
        return ()

    def label(self) -> str:
        """Short label used for nested nodes in :meth:`RuleResult.explain`."""
        return _truncate(self._key())

    def test(self, text: str, matcher: BaseMatcher) -> RuleResult:
        """Convenience: run ``matcher.find(text)`` and evaluate this rule."""
        return Document(text, matcher.find(text)).evaluate(self)

    def at_least(self, n: int) -> AtLeast:
        """This rule occurring at least ``n`` times: ``kw("bank").at_least(3)``."""
        return AtLeast(self, n)

    def __and__(self, other: Rule | str) -> AllOf:
        return AllOf(self, other)

    def __rand__(self, other: Rule | str) -> AllOf:
        return AllOf(other, self)

    def __or__(self, other: Rule | str) -> AnyOf:
        return AnyOf(self, other)

    def __ror__(self, other: Rule | str) -> AnyOf:
        return AnyOf(other, self)

    def __invert__(self) -> Not:
        return Not(self)

    def __repr__(self) -> str:
        return f"Rule({self._key()})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Rule) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())


def _r(x: Rule | str) -> Rule:
    if isinstance(x, Rule):
        return _collapse(x)
    if isinstance(x, str):
        return KeywordRule(x)
    raise TypeError(f"Expected a Rule or a keyword name, got {type(x).__name__}")


def _collapse(rule: Rule) -> Rule:
    """The operand of a single-operand OR / AND (which mean just that operand), else ``rule``.

    Operands are collapsed on construction so that e.g. ``near("a", any_of(not_("b")))`` is
    rejected like ``NEAR(a, NOT b)`` and every rule serialises to a string that parses back to it.
    """
    while isinstance(rule, (AnyOf, AllOf)) and len(rule.rules) == 1 and getattr(rule, "min_count", 1) == 1:
        rule = rule.rules[0]
    return rule


def _evidence(result: RuleResult) -> frozenset[tuple[str, tuple[int, int]]]:
    return frozenset((m.keyword, m.span) for m in result.matches)


def _counts_matches(rule: Rule) -> bool:
    """True for rules whose occurrence count is their number of distinct matches."""
    if isinstance(rule, (KeywordRule, CategoryRule, TagRule)):
        return True
    return isinstance(rule, AnyOf) and rule.min_count == 1 and all(_counts_matches(c) for c in rule.rules)


def _distinct_operands(results: Sequence[RuleResult]) -> list[RuleResult]:
    """A largest set of matched operands that are satisfied by pairwise different matches.

    Bipartite matching (operand -> one of its matches, keyword + span); operands without matches
    (e.g. ``NOT x``) are distinct unless they are the same rule. Higher counts are tried first.
    """
    unique: dict[str, RuleResult] = {}
    for res in results:
        if res.matched:
            unique.setdefault(res.rule._key(), res)
    cands = sorted(unique.values(), key=lambda r: -r.occurrences)
    # Operands without matches get a pseudo-match of their own (no real span is negative).
    evidence = [sorted(_evidence(r), key=lambda e: (e[1], e[0])) or [(r.rule._key(), (-1, -1))] for r in cands]
    owner: dict[tuple[str, tuple[int, int]], int] = {}

    def assign(i: int, seen: set[tuple[str, tuple[int, int]]]) -> bool:
        for e in evidence[i]:
            if e in seen:
                continue
            seen.add(e)
            if e not in owner or assign(owner[e], seen):
                owner[e] = i
                return True
        return False

    chosen = [i for i in range(len(cands)) if assign(i, set())]
    assigned = set(owner.values())
    return [cands[i] for i in chosen if i in assigned]


def _positive(x: Rule | str, where: str) -> Rule:
    rule = _r(x)
    if isinstance(rule, Not):
        raise ValueError(
            f"{where} cannot take a NOT operand ({rule.to_dsl()}): there is no position for an absence. "
            "Negate the whole expression instead, e.g. NOT NEAR(a, b, 5)."
        )
    return rule


class KeywordRule(Rule):
    """Matches when the keyword (canonical name, case-insensitive) occurs."""

    def __init__(self, name: str) -> None:
        self.name = _check_name(name, "keyword")

    def evaluate(self, doc: Document) -> RuleResult:
        hits = tuple(doc.keyword_matches(self.name))
        return RuleResult(self, bool(hits), hits, count=len({(m.keyword, m.span) for m in hits}))

    def keyword_names(self) -> set[str]:
        return {self.name}

    def to_dsl(self) -> str:
        return _quote(self.name)


class CategoryRule(Rule):
    """Matches when any keyword of the given category (case-insensitive) occurs."""

    def __init__(self, name: str) -> None:
        self.name = _check_name(name, "category")

    def evaluate(self, doc: Document) -> RuleResult:
        folded = _fold(self.name)
        hits = tuple(m for m in doc.matches if m.category is not None and _fold(m.category) == folded)
        return RuleResult(self, bool(hits), hits, count=len({(m.keyword, m.span) for m in hits}))

    def to_dsl(self) -> str:
        return f"CATEGORY({_quote(self.name)})"


class TagRule(Rule):
    """Matches when any keyword carrying the tag (case-insensitive) occurs."""

    def __init__(self, name: str) -> None:
        self.name = _check_name(name, "tag")

    def evaluate(self, doc: Document) -> RuleResult:
        folded = _fold(self.name)
        hits = tuple(
            m for m in doc.matches if m.keyword_obj is not None and any(_fold(t) == folded for t in m.keyword_obj.tags)
        )
        return RuleResult(self, bool(hits), hits, count=len({(m.keyword, m.span) for m in hits}))

    def to_dsl(self) -> str:
        return f"TAG({_quote(self.name)})"


class AnyOf(Rule):
    """At least ``min_count`` of the child rules match (default 1 = OR).

    With ``min_count > 1`` only children satisfied by different matches count (see the module
    docs). Occurrences: for OR the number of distinct matches of the match-counting children plus
    the counts of the others; for ``min_count > 1`` the ``min_count``-th largest count among the
    distinct children.
    """

    rules: tuple[Rule, ...]
    min_count: int

    def __init__(self, *rules: Rule | str, min_count: int = 1) -> None:
        if not rules:
            raise ValueError("AnyOf needs at least one rule")
        _check_int(min_count, "min_count", 1)
        flat: list[Rule] = []
        for r in map(_r, rules):  # flatten (a | b) | c
            flat.extend(r.rules if isinstance(r, AnyOf) and r.min_count == 1 and min_count == 1 else [r])
        if min_count > len(flat):
            raise ValueError(f"ANY({min_count}, ...) can never match: it has only {len(flat)} operand(s)")
        self.rules = tuple(flat)
        self.min_count = min_count

    def children(self) -> Sequence[Rule]:
        return self.rules

    def evaluate(self, doc: Document) -> RuleResult:
        results = tuple(doc.evaluate(r) for r in self.rules)
        n = sum(1 for r in results if r.matched)
        detail = ""
        if self.min_count == 1:
            ok = n >= 1
            spans: set[tuple[str, tuple[int, int]]] = set()
            others: dict[str, int] = {}
            for rule, res in zip(self.rules, results, strict=True):
                if not res.matched:
                    continue
                if _counts_matches(rule):
                    spans |= _evidence(res)
                else:
                    others.setdefault(rule._key(), res.occurrences)
            count = len(spans) + sum(others.values()) if ok else 0
        else:
            distinct = _distinct_operands(results)
            ok = len(distinct) >= self.min_count
            counts = sorted((r.occurrences for r in distinct), reverse=True)
            count = counts[self.min_count - 1] if ok else 0
            extra = f" ({len(distinct)} with distinct matches)" if len(distinct) != n else ""
            detail = f"{n}/{len(results)} matched{extra}, need {self.min_count}"
        hits = _unique(m for r in results if r.matched for m in r.matches) if ok else ()
        return RuleResult(self, ok, hits, results, detail, count)

    def label(self) -> str:
        return _truncate(("OR " if self.min_count == 1 and len(self.rules) > 1 else "") + self._key())

    def to_dsl(self) -> str:
        if self.min_count == 1:
            return (
                "(" + " OR ".join(r._key() for r in self.rules) + ")" if len(self.rules) > 1 else self.rules[0]._key()
            )
        return f"ANY({self.min_count}, " + ", ".join(r._key() for r in self.rules) + ")"


class AllOf(Rule):
    """All child rules match (AND). Occurrences: the smallest count of the positive children."""

    rules: tuple[Rule, ...]

    def __init__(self, *rules: Rule | str) -> None:
        if not rules:
            raise ValueError("AllOf needs at least one rule")
        flat: list[Rule] = []
        for r in map(_r, rules):
            flat.extend(r.rules if isinstance(r, AllOf) else [r])
        self.rules = tuple(flat)

    def children(self) -> Sequence[Rule]:
        return self.rules

    def evaluate(self, doc: Document) -> RuleResult:
        results = tuple(doc.evaluate(r) for r in self.rules)
        ok = all(r.matched for r in results)
        hits = _unique(m for r in results for m in r.matches) if ok else ()
        positive = [r.occurrences for r in results if not isinstance(r.rule, Not)]
        count = (min(positive) if positive else 1) if ok else 0
        return RuleResult(self, ok, hits, results, count=count)

    def label(self) -> str:
        return _truncate(("AND " if len(self.rules) > 1 else "") + self._key())

    def to_dsl(self) -> str:
        if len(self.rules) == 1:
            return self.rules[0]._key()
        return "(" + " AND ".join(r._key() for r in self.rules) + ")"


class Not(Rule):
    """Negation. Never contributes matches.

    At the top level (or inside AND / OR) it means "nowhere in the text". As an operand of
    :class:`SameSentence` / :class:`SameParagraph` it means "not in that sentence / paragraph".
    It is rejected as an operand of NEAR, BEFORE and ATLEAST.
    """

    def __init__(self, rule: Rule | str) -> None:
        self.rule = _r(rule)

    def children(self) -> Sequence[Rule]:
        return (self.rule,)

    def evaluate(self, doc: Document) -> RuleResult:
        inner = doc.evaluate(self.rule)
        return RuleResult(self, not inner.matched, (), (inner,), count=0 if inner.matched else 1)

    def to_dsl(self) -> str:
        return f"NOT {self.rule._key()}"


class AtLeast(Rule):
    """The child rule occurs at least ``n`` times (see the module docs for what an occurrence is)."""

    def __init__(self, rule: Rule | str, n: int) -> None:
        self.n = _check_int(n, "ATLEAST count n", 1)
        self.rule = _positive(rule, "ATLEAST")

    def children(self) -> Sequence[Rule]:
        return (self.rule,)

    def evaluate(self, doc: Document) -> RuleResult:
        inner = doc.evaluate(self.rule)
        count = inner.occurrences if inner.matched else 0
        ok = count >= self.n
        return RuleResult(
            self, ok, inner.matches if ok else (), (inner,), f"{count} found, need {self.n}", count if ok else 0
        )

    def to_dsl(self) -> str:
        return f"ATLEAST({self.n}, {self.rule._key()})"


class Near(Rule):
    """Two rules match within ``within`` words of each other (``ordered``: ``a`` before ``b``).

    With ``ordered=True`` (``BEFORE``) the ``a`` match must end before the ``b`` match starts (no
    overlap). Matches with identical spans never pair up. Occurrences: the number of disjoint
    qualifying pairs, formed greedily left to right (each match takes part in at most one pair).
    """

    def __init__(self, a: Rule | str, b: Rule | str, within: int = 5, ordered: bool = False) -> None:
        name = "BEFORE" if ordered else "NEAR"
        self.within = _check_int(within, f"{name} distance", 0)
        if not isinstance(ordered, bool):
            raise ValueError(f"ordered must be a bool, got {ordered!r}")
        self.a, self.b = _positive(a, name), _positive(b, name)
        self.ordered = ordered

    def children(self) -> Sequence[Rule]:
        return (self.a, self.b)

    def _gap_ok(self, ma: Match, sa: tuple[int, int], mb: Match, sb: tuple[int, int]) -> bool:
        if ma.span == mb.span:
            return False
        a0, a1 = sa
        b0, b1 = sb
        if self.ordered:
            return ma.end <= mb.start and max(0, b0 - a1 - 1) <= self.within
        if b0 > a1:
            return b0 - a1 - 1 <= self.within
        if a0 > b1:
            return a0 - b1 - 1 <= self.within
        return True  # overlapping spans

    def evaluate(self, doc: Document) -> RuleResult:
        ra, rb = doc.evaluate(self.a), doc.evaluate(self.b)
        pairs = 0
        hits: list[Match] = []
        if ra.matched and rb.matched and ra.matches and rb.matches:
            pairs, hits = self._pairs(doc, list(ra.matches), list(rb.matches))
        return RuleResult(
            self, pairs > 0, _unique(hits), (ra, rb), f"{pairs} pair(s) within {self.within} words", pairs
        )

    def _pairs(self, doc: Document, ams: list[Match], bms: list[Match]) -> tuple[int, list[Match]]:
        w = self.within
        a_list = sorted(((m, doc.word_span(m)) for m in ams), key=lambda x: (x[0].start, x[0].end))
        b_list = sorted(((m, doc.word_span(m)) for m in bms), key=lambda x: (x[0].start, x[0].end))
        b_starts = [m.start for m, _ in b_list]
        b_w0 = [s[0] for _, s in b_list]
        b_len = max(s[1] - s[0] for _, s in b_list)

        def window(m: Match, s: tuple[int, int]) -> tuple[int, int]:
            hi = bisect_right(b_w0, s[1] + w + 1)
            lo = bisect_left(b_starts, m.end) if self.ordered else bisect_left(b_w0, s[0] - w - 1 - b_len)
            return lo, hi

        hits: list[Match] = []
        # Which a-matches have a partner (first valid candidate in their window).
        for m, s in a_list:
            lo, hi = window(m, s)
            if any(self._gap_ok(m, s, *b_list[j]) for j in range(lo, hi)):
                hits.append(m)
        # Which b-matches have a partner (symmetric scan over a, sorted by end for BEFORE).
        a_by_end = sorted(a_list, key=lambda x: (x[0].end, x[0].start))
        a_ends = [m.end for m, _ in a_by_end]
        a_w1 = [s[1] for _, s in a_by_end]
        a_by_w0 = [s[0] for _, s in a_list]
        a_len = max(s[1] - s[0] for _, s in a_list)
        for m, s in b_list:
            if self.ordered:
                lo, hi = bisect_left(a_w1, s[0] - w - 1), bisect_right(a_ends, m.start)
                cands: Sequence[tuple[Match, tuple[int, int]]] = a_by_end
            else:
                lo, hi = bisect_left(a_by_w0, s[0] - w - 1 - a_len), bisect_right(a_by_w0, s[1] + w + 1)
                cands = a_list
            if any(self._gap_ok(cands[j][0], cands[j][1], m, s) for j in range(lo, hi)):
                hits.append(m)

        # Greedy disjoint pairs: each match (by identity) is used at most once, on either side.
        nxt = list(range(len(b_list) + 1))

        def find(i: int) -> int:
            root = i
            while nxt[root] != root:
                root = nxt[root]
            while nxt[i] != root:
                nxt[i], i = root, nxt[i]
            return root

        b_index: dict[Match, list[int]] = {}
        for j, (m, _) in enumerate(b_list):
            b_index.setdefault(m, []).append(j)
        used: set[Match] = set()

        def use(m: Match) -> None:
            used.add(m)
            for j in b_index.get(m, ()):
                nxt[j] = j + 1

        pairs = 0
        for m, s in a_list:
            if m in used:
                continue
            lo, hi = window(m, s)
            j = find(lo)
            while j < hi:
                mb, sb = b_list[j]
                if mb is not m and mb not in used and self._gap_ok(m, s, mb, sb):
                    use(m)
                    use(mb)
                    pairs += 1
                    break
                j = find(j + 1)
        return pairs, hits

    def to_dsl(self) -> str:
        name = "BEFORE" if self.ordered else "NEAR"
        return f"{name}({self.a._key()}, {self.b._key()}, {self.within})"


_BUILTIN_MONOTONE: tuple[type, ...] = ()  # filled in below


def _monotone(rule: Rule) -> bool:
    """True if restricting the matches can only turn the rule from true to false (no NOT inside)."""
    if isinstance(rule, Not) or not isinstance(rule, _BUILTIN_MONOTONE):
        return False
    return all(_monotone(c) for c in rule.children())


class _SameUnit(Rule):
    """All positive operands are satisfied inside one unit; ``NOT x`` operands are absent from it.

    Each operand is evaluated on the matches of the unit only, so ``SENTENCE(a AND b, c)`` needs
    ``a``, ``b`` and ``c`` in one sentence. Occurrences: the number of qualifying units.
    """

    unit = "sentence"
    keyword = "SENTENCE"

    def __init__(self, *rules: Rule | str) -> None:
        if len(rules) < 2:
            raise ValueError(f"{type(self).__name__} needs at least two rules")
        self.rules = tuple(map(_r, rules))
        if all(isinstance(r, Not) for r in self.rules):
            raise ValueError(f"{self.keyword}(...) needs at least one operand that is not a NOT")

    def children(self) -> Sequence[Rule]:
        return self.rules

    def _unit(self, doc: Document, m: Match) -> int:
        return doc.unit_of(m, self.unit)

    def evaluate(self, doc: Document) -> RuleResult:
        positive = [r for r in self.rules if not isinstance(r, Not)]
        candidates: set[int] | None = None
        for r in positive:
            if not _monotone(r):
                continue
            res = doc.evaluate(r)
            if not res.matched:
                return self._fail(doc)
            units = {self._unit(doc, m) for m in res.matches}
            candidates = units if candidates is None else candidates & units
            if not candidates:
                return self._fail(doc)
        if candidates is None:
            candidates = set(range(doc.unit_count(self.unit)))
        by_unit: dict[int, list[Match]] = {}
        for m in doc.matches:
            u = self._unit(doc, m)
            if u in candidates:
                by_unit.setdefault(u, []).append(m)
        qualifying = 0
        hits: list[Match] = []
        first: tuple[RuleResult, ...] = ()
        for u in sorted(candidates):
            sub = doc.restrict(by_unit.get(u, ()))
            results = tuple(sub.evaluate(r) for r in self.rules)
            if all(r.matched for r in results):
                qualifying += 1
                hits.extend(m for r in results for m in r.matches)
                first = first or results
        if not qualifying:
            return self._fail(doc)
        return RuleResult(self, True, _unique(hits), first, self._detail(qualifying), qualifying)

    def _detail(self, n: int) -> str:
        return f"{n} shared {self.unit}(s)"

    def _fail(self, doc: Document) -> RuleResult:
        results = tuple(doc.evaluate(r) for r in self.rules)
        return RuleResult(self, False, (), results, self._detail(0), 0)

    def to_dsl(self) -> str:
        return f"{self.keyword}(" + ", ".join(r._key() for r in self.rules) + ")"


class SameSentence(_SameUnit):
    """All rules match inside one sentence (``NOT x`` operands: ``x`` is not in that sentence)."""

    unit = "sentence"
    keyword = "SENTENCE"


class SameParagraph(_SameUnit):
    """All rules match inside one paragraph (blocks separated by blank lines)."""

    unit = "paragraph"
    keyword = "PARAGRAPH"


_BUILTIN_MONOTONE = (KeywordRule, CategoryRule, TagRule, AnyOf, AllOf, AtLeast, Near, _SameUnit)


def walk(rule: Rule) -> Iterator[Rule]:
    """Yield ``rule`` and all of its descendants (depth first)."""
    yield rule
    for child in rule.children():
        yield from walk(child)


# ---------------------------------------------------------------------- helper functions
def kw(*names: str) -> Rule:
    """``kw("bank")``; several names mean *any of them*: ``kw("bank", "lender")``."""
    if not names:
        raise ValueError("kw() needs at least one keyword name")
    return KeywordRule(names[0]) if len(names) == 1 else AnyOf(*names)


def category(name: str) -> CategoryRule:
    """Any keyword of category ``name`` occurs (case-insensitive). DSL: ``CATEGORY(name)``."""
    return CategoryRule(name)


def tag(name: str) -> TagRule:
    """Any keyword tagged ``name`` occurs (case-insensitive). DSL: ``TAG(name)``."""
    return TagRule(name)


def any_of(*rules: Rule | str, min_count: int = 1) -> AnyOf:
    """At least ``min_count`` of ``rules`` match (default 1: OR). DSL: ``ANY(n, a, b, ...)``.

    With ``min_count > 1`` operands must be satisfied by different matches, so
    ``any_of("ML", "machine learning", min_count=2)`` does not fire on a single "ML".
    Strings are keyword names.
    """
    return AnyOf(*rules, min_count=min_count)


def all_of(*rules: Rule | str) -> AllOf:
    """All of ``rules`` match (AND). DSL: ``ALL(a, b, ...)`` or ``a AND b``. Strings are keyword names."""
    return AllOf(*rules)


def none_of(*rules: Rule | str) -> Not:
    """None of ``rules`` occurs anywhere in the text. DSL: ``NONE(a, b, ...)``."""
    return Not(AnyOf(*rules))


def not_(rule: Rule | str) -> Not:
    """``rule`` does not match (same as ``~rule``). DSL: ``NOT rule``."""
    return Not(rule)


def at_least(n: int, rule: Rule | str) -> AtLeast:
    """``rule`` occurs at least ``n`` times. DSL: ``ATLEAST(n, rule)``.

    See the module docs for what counts as an occurrence of each kind of rule.
    """
    return AtLeast(rule, n)


def near(a: Rule | str, b: Rule | str, within: int = 5, ordered: bool = False) -> Near:
    """``a`` and ``b`` occur with at most ``within`` words between them. DSL: ``NEAR(a, b, within)``.

    Args:
        a: A rule or keyword name.
        b: A rule or keyword name.
        within: Maximum number of words between the two matches.
        ordered: Require ``a`` before ``b`` (the same as :func:`before`).
    """
    return Near(a, b, within, ordered)


def before(a: Rule | str, b: Rule | str, within: int = 5) -> Near:
    """``a`` occurs before ``b`` with at most ``within`` words in between. DSL: ``BEFORE(a, b, within)``."""
    return Near(a, b, within, ordered=True)


def same_sentence(*rules: Rule | str) -> SameSentence:
    """All ``rules`` match inside one sentence. DSL: ``SENTENCE(a, b, ...)``.

    A ``not_(x)`` operand means ``x`` is absent from that sentence.
    """
    return SameSentence(*rules)


def same_paragraph(*rules: Rule | str) -> SameParagraph:
    """All ``rules`` match inside one paragraph (blocks separated by blank lines). DSL: ``PARAGRAPH(a, b, ...)``."""
    return SameParagraph(*rules)
