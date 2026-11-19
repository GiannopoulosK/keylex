"""The :class:`Keyword` object and the :class:`Term` objects it is made of."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import total_ordering
from typing import Any, Literal

from .text import looks_like_acronym

__all__ = ["TERM_KINDS", "Keyword", "Term", "TermKind"]

TermKind = Literal["keyword", "synonym", "acronym", "variant", "pattern"]
TERM_KINDS: tuple[str, ...] = ("keyword", "synonym", "acronym", "variant", "pattern")
_RELATED_KINDS = ("synonym", "acronym", "variant")


#: Global count of keyword / keyword-list mutations. Matchers compare it to skip the O(n)
#: per-keyword signature check when nothing anywhere has changed.
_MUTATIONS = [0]


def _clean(text: Any, what: str = "term") -> str:
    if not isinstance(text, str):
        raise TypeError(f"{what} must be a str, got {type(text).__name__}: {text!r}")
    if not text.isascii():
        text = unicodedata.normalize("NFC", text)  # "café" typed as e + U+0301 == "café"
    cleaned = " ".join(text.split())
    if not cleaned:
        raise ValueError(f"{what} must be a non-empty string")
    return cleaned


def _flatten(items: tuple[Any, ...]) -> list[Any]:
    """Accept ``f("a", "b")``, ``f(["a", "b"])`` and ``f("a")`` uniformly.

    A bare string is *never* split into characters.
    """
    out: list[Any] = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, (str, Term)):
            out.append(item)
        elif isinstance(item, Iterable):
            out.extend(_flatten(tuple(item)))
        else:
            raise TypeError(f"Expected str, Term or an iterable of them, got {type(item).__name__}")
    return out


def _key(text: str) -> str:
    return text.casefold()


@dataclass(frozen=True)
class Term:
    """One surface form of a keyword.

    Args:
        text: The text to match (or a regular expression when ``kind="pattern"``).
        kind: ``"keyword"``, ``"synonym"``, ``"acronym"``, ``"variant"`` or ``"pattern"``.
        case_sensitive: ``None`` inherits from the keyword / matcher. Acronyms and all-caps
            terms such as ``"NASA"`` are case-sensitive by default.
        fuzzy: Allow fuzzy matching for this term. ``None`` = default (acronyms, patterns and
            very short terms are never fuzzy-matched).
        whole_word: Only match on word boundaries (default). ``False`` allows matches inside
            words, e.g. ``"bio"`` in ``"biotech"``.
    """

    text: str
    kind: TermKind = "keyword"
    case_sensitive: bool | None = None
    fuzzy: bool | None = None
    whole_word: bool = True

    def __post_init__(self) -> None:
        if self.kind not in TERM_KINDS:
            raise ValueError(f"Unknown term kind {self.kind!r}; expected one of {TERM_KINDS}")
        if self.kind == "pattern":
            if not isinstance(self.text, str) or not self.text:
                raise ValueError("pattern must be a non-empty string")
            try:
                re.compile(self.text)
            except re.error as exc:
                raise ValueError(f"Invalid regular expression {self.text!r}: {exc}") from None
        else:
            object.__setattr__(self, "text", _clean(self.text))

    @property
    def is_acronym_like(self) -> bool:
        return self.kind == "acronym" or (self.kind != "pattern" and looks_like_acronym(self.text))

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"text": self.text, "kind": self.kind}
        if self.case_sensitive is not None:
            data["case_sensitive"] = self.case_sensitive
        if self.fuzzy is not None:
            data["fuzzy"] = self.fuzzy
        if not self.whole_word:
            data["whole_word"] = False
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Term:
        return cls(**{k: data[k] for k in ("text", "kind", "case_sensitive", "fuzzy", "whole_word") if k in data})


@total_ordering
class Keyword:
    """A canonical keyword plus everything that should count as a mention of it.

    Example::

        ai = Keyword(
            "artificial intelligence",
            synonyms=["machine intelligence"],
            acronyms=["AI"],                       # case-sensitive: won't match "ai" in "Thai"
            exclude=["Steven Spielberg"],          # reject matches with this nearby
            category="technology",
        )

    Args:
        keyword: The canonical name. Matches always report this name.
        synonyms / acronyms / variants: Alternative surface forms. Accepts a string or a list.
        patterns: Regular expressions that also count as this keyword.
        case_sensitive: Default case sensitivity for this keyword's terms (``None`` = inherit).
        fuzzy: Whether fuzzy matching is allowed for this keyword (``None`` = inherit).
        exclude: Context phrases; a match is rejected if one appears within ``context_window``
            words of it (e.g. ``Keyword("Apple", exclude=["fruit", "pie"])``).
        require: Context phrases; a match is kept only if at least one appears nearby.
        context_window: Number of words on each side inspected for ``exclude`` / ``require``.
        category: Optional category, reported on every match.
        tags: Free-form labels (usable from rules via ``tag("...")``).
        weight: Importance used by ``output="score"`` and rule scoring.
        metadata: Arbitrary JSON-serialisable data carried along (IDs, codes, ...).
    """

    __hash__ = None  # type: ignore[assignment]  # mutable -> unhashable (use .name as a key)

    def __init__(
        self,
        keyword: str,
        synonyms: str | Iterable[str] | None = None,
        acronyms: str | Iterable[str] | None = None,
        *,
        variants: str | Iterable[str] | None = None,
        patterns: str | Iterable[str] | None = None,
        case_sensitive: bool | None = None,
        fuzzy: bool | None = None,
        exclude: str | Iterable[str] | None = None,
        require: str | Iterable[str] | None = None,
        context_window: int = 10,
        category: str | None = None,
        tags: str | Iterable[str] | None = None,
        weight: float = 1.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._name = _clean(keyword, "keyword")
        self._terms: dict[str, Term] = {_key(self._name): Term(self._name, "keyword")}
        self._version = 0
        self._case_sensitive = case_sensitive
        self._fuzzy = fuzzy
        self._exclude: list[str] = []
        self._require: list[str] = []
        self._context_window = 10
        self._category: str | None = None
        self._tags: list[str] = []
        self._weight = 1.0
        self.metadata: dict[str, Any] = dict(metadata or {})

        self.add_synonyms(synonyms)
        self.add_acronyms(acronyms)
        self.add_variants(variants)
        self.add_patterns(patterns)
        self.exclude = exclude
        self.require = require
        self.context_window = context_window
        self.category = category
        self.tags = tags
        self.weight = weight

    # ------------------------------------------------------------------ identity
    @property
    def name(self) -> str:
        """The canonical keyword."""
        return self._name

    @property
    def keyword(self) -> str:
        """Alias of :attr:`name` (read-only; use ``KeywordList.rename`` to rename)."""
        return self._name

    @keyword.setter
    def keyword(self, value: str) -> None:
        raise AttributeError(
            "Keyword names are read-only so that KeywordList indexes stay consistent. "
            "Use KeywordList.rename(old, new) or Keyword.renamed(new)."
        )

    def renamed(self, new_name: str) -> Keyword:
        """Return a copy of this keyword with a different canonical name."""
        return Keyword.from_dict({**self.to_dict(), "keyword": new_name})

    # ------------------------------------------------------------------ terms
    @property
    def terms(self) -> list[Term]:
        """All terms, canonical keyword first, in insertion order."""
        return list(self._terms.values())

    def iter_terms(self, *kinds: str) -> Iterator[Term]:
        for term in self._terms.values():
            if not kinds or term.kind in kinds:
                yield term

    def _texts(self, kind: str) -> list[str]:
        return [t.text for t in self._terms.values() if t.kind == kind]

    def add_terms(self, kind: TermKind, *items: str | Term | Iterable[str | Term] | None) -> None:
        """Add terms of the given kind. Duplicates (case-insensitive) are ignored."""
        if kind == "keyword":
            raise ValueError("The canonical keyword cannot be added; it is set at construction")
        changed = False
        for item in _flatten(items):
            term = item if isinstance(item, Term) else Term(item, kind)
            if term.kind != kind:
                term = Term(term.text, kind, term.case_sensitive, term.fuzzy, term.whole_word)
            key = term.text if term.kind == "pattern" else _key(term.text)
            if term.kind == "pattern":
                key = "re:" + key
            if key not in self._terms:
                self._terms[key] = term
                changed = True
        if changed:
            self._touch()

    def remove_terms(self, kind: TermKind, *items: str | Iterable[str] | None) -> None:
        changed = False
        for item in _flatten(items):
            text = item.text if isinstance(item, Term) else item
            key = ("re:" + text) if kind == "pattern" else _key(" ".join(text.split()))
            term = self._terms.get(key)
            if term is not None and term.kind == kind:
                del self._terms[key]
                changed = True
        if changed:
            self._touch()

    def clear_terms(self, *kinds: str) -> None:
        kinds = kinds or (*_RELATED_KINDS, "pattern")
        before = len(self._terms)
        self._terms = {k: t for k, t in self._terms.items() if t.kind == "keyword" or t.kind not in kinds}
        if len(self._terms) != before:
            self._touch()

    # --- synonyms
    @property
    def synonyms(self) -> list[str]:
        return self._texts("synonym")

    @synonyms.setter
    def synonyms(self, synonyms: str | Iterable[str] | None) -> None:
        self.set_synonyms(synonyms)

    def add_synonyms(self, *synonyms: str | Iterable[str] | None) -> None:
        """Add synonyms: ``add_synonyms("a", "b")`` or ``add_synonyms(["a", "b"])``."""
        self.add_terms("synonym", *synonyms)

    def remove_synonyms(self, *synonyms: str | Iterable[str] | None) -> None:
        self.remove_terms("synonym", *synonyms)

    def set_synonyms(self, *synonyms: str | Iterable[str] | None) -> None:
        self.clear_terms("synonym")
        self.add_synonyms(*synonyms)

    def clear_synonyms(self) -> None:
        self.clear_terms("synonym")

    @property
    def has_synonyms(self) -> bool:
        return any(t.kind == "synonym" for t in self._terms.values())

    # --- acronyms
    @property
    def acronyms(self) -> list[str]:
        return self._texts("acronym")

    @acronyms.setter
    def acronyms(self, acronyms: str | Iterable[str] | None) -> None:
        self.set_acronyms(acronyms)

    def add_acronyms(self, *acronyms: str | Iterable[str] | None) -> None:
        self.add_terms("acronym", *acronyms)

    def remove_acronyms(self, *acronyms: str | Iterable[str] | None) -> None:
        self.remove_terms("acronym", *acronyms)

    def set_acronyms(self, *acronyms: str | Iterable[str] | None) -> None:
        self.clear_terms("acronym")
        self.add_acronyms(*acronyms)

    def clear_acronyms(self) -> None:
        self.clear_terms("acronym")

    @property
    def has_acronyms(self) -> bool:
        return any(t.kind == "acronym" for t in self._terms.values())

    # --- variants (inflections, spellings, ...)
    @property
    def variants(self) -> list[str]:
        return self._texts("variant")

    @variants.setter
    def variants(self, variants: str | Iterable[str] | None) -> None:
        self.clear_terms("variant")
        self.add_variants(variants)

    def add_variants(self, *variants: str | Iterable[str] | None) -> None:
        self.add_terms("variant", *variants)

    def remove_variants(self, *variants: str | Iterable[str] | None) -> None:
        self.remove_terms("variant", *variants)

    # --- regex patterns
    @property
    def patterns(self) -> list[str]:
        return self._texts("pattern")

    @patterns.setter
    def patterns(self, patterns: str | Iterable[str] | None) -> None:
        self.clear_terms("pattern")
        self.add_patterns(patterns)

    def add_patterns(self, *patterns: str | Iterable[str] | None) -> None:
        self.add_terms("pattern", *patterns)

    def remove_patterns(self, *patterns: str | Iterable[str] | None) -> None:
        self.remove_terms("pattern", *patterns)

    # --- related terms (synonyms + acronyms + variants)
    @property
    def related_terms(self) -> list[str]:
        """Synonyms, acronyms and variants (everything except the keyword and patterns)."""
        return [t.text for t in self._terms.values() if t.kind in _RELATED_KINDS]

    @related_terms.setter
    def related_terms(self, terms: Mapping[str, Iterable[str]]) -> None:
        self.clear_terms(*_RELATED_KINDS)
        self.add_related_terms(**dict(terms.items()))

    def add_related_terms(
        self,
        synonyms: str | Iterable[str] | None = None,
        acronyms: str | Iterable[str] | None = None,
        variants: str | Iterable[str] | None = None,
    ) -> None:
        self.add_synonyms(synonyms)
        self.add_acronyms(acronyms)
        self.add_variants(variants)

    def remove_related_terms(
        self,
        synonyms: str | Iterable[str] | None = None,
        acronyms: str | Iterable[str] | None = None,
        variants: str | Iterable[str] | None = None,
    ) -> None:
        self.remove_synonyms(synonyms)
        self.remove_acronyms(acronyms)
        self.remove_variants(variants)

    def clear_related_terms(self) -> None:
        self.clear_terms(*_RELATED_KINDS)

    def has_related_terms(self) -> bool:
        return any(t.kind in _RELATED_KINDS for t in self._terms.values())

    # ------------------------------------------------------------------ matching options
    def _touch(self) -> None:
        self._version += 1
        _MUTATIONS[0] += 1

    @property
    def case_sensitive(self) -> bool | None:
        return self._case_sensitive

    @case_sensitive.setter
    def case_sensitive(self, value: bool | None) -> None:
        self._case_sensitive = value
        self._touch()

    @property
    def fuzzy(self) -> bool | None:
        return self._fuzzy

    @fuzzy.setter
    def fuzzy(self, value: bool | None) -> None:
        self._fuzzy = value
        self._touch()

    @property
    def exclude(self) -> list[str]:
        return list(self._exclude)

    @exclude.setter
    def exclude(self, value: str | Iterable[str] | None) -> None:
        self._exclude = list(dict.fromkeys(_clean(v, "exclude phrase") for v in _flatten((value,))))
        self._touch()

    @property
    def require(self) -> list[str]:
        return list(self._require)

    @require.setter
    def require(self, value: str | Iterable[str] | None) -> None:
        self._require = list(dict.fromkeys(_clean(v, "require phrase") for v in _flatten((value,))))
        self._touch()

    @property
    def context_window(self) -> int:
        return self._context_window

    @context_window.setter
    def context_window(self, value: int) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("context_window must be a non-negative integer")
        self._context_window = value
        self._touch()

    @property
    def category(self) -> str | None:
        return self._category

    @category.setter
    def category(self, value: str | None) -> None:
        self._category = None if value is None else _clean(value, "category")
        self._touch()

    @property
    def tags(self) -> list[str]:
        return list(self._tags)

    @tags.setter
    def tags(self, value: str | Iterable[str] | None) -> None:
        self._tags = list(dict.fromkeys(_clean(v, "tag") for v in _flatten((value,))))
        self._touch()

    @property
    def weight(self) -> float:
        return self._weight

    @weight.setter
    def weight(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("weight must be a number")
        self._weight = float(value)
        self._touch()

    # ------------------------------------------------------------------ utilities
    def copy(self) -> Keyword:
        """Return an independent deep copy."""
        return Keyword.from_dict(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict (inverse of :meth:`from_dict`)."""
        data: dict[str, Any] = {"keyword": self._name}
        for kind, plural in (
            ("synonym", "synonyms"),
            ("acronym", "acronyms"),
            ("variant", "variants"),
            ("pattern", "patterns"),
        ):
            terms = [t for t in self._terms.values() if t.kind == kind]
            if terms or kind in ("synonym", "acronym"):
                data[plural] = [
                    t.text if t.case_sensitive is None and t.fuzzy is None and t.whole_word else t.to_dict()
                    for t in terms
                ]
        canonical = next(iter(self._terms.values()))
        if canonical.whole_word is False or canonical.fuzzy is not None or canonical.case_sensitive is not None:
            data["keyword_term"] = canonical.to_dict()
        optional: dict[str, tuple[Any, Any]] = {
            "case_sensitive": (self._case_sensitive, None),
            "fuzzy": (self._fuzzy, None),
            "exclude": (self._exclude, []),
            "require": (self._require, []),
            "context_window": (self._context_window, 10),
            "category": (self._category, None),
            "tags": (self._tags, []),
            "weight": (self._weight, 1.0),
            "metadata": (self.metadata, {}),
        }
        for name, (value, default) in optional.items():
            if value != default:
                data[name] = list(value) if isinstance(value, list) else value
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Keyword:
        """Build a keyword from :meth:`to_dict` output (unknown keys raise ``ValueError``)."""
        if "keyword" not in data:
            raise ValueError("Keyword dict must contain a 'keyword' key")
        known = {
            "keyword",
            "synonyms",
            "acronyms",
            "variants",
            "patterns",
            "case_sensitive",
            "fuzzy",
            "exclude",
            "require",
            "context_window",
            "category",
            "tags",
            "weight",
            "metadata",
            "keyword_term",
        }
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown keyword field(s): {sorted(unknown)}")
        kw = cls(
            data["keyword"],
            case_sensitive=data.get("case_sensitive"),
            fuzzy=data.get("fuzzy"),
            exclude=data.get("exclude"),
            require=data.get("require"),
            context_window=data.get("context_window", 10),
            category=data.get("category"),
            tags=data.get("tags"),
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata"),
        )
        if "keyword_term" in data:
            term = Term.from_dict({**data["keyword_term"], "text": kw.name, "kind": "keyword"})
            kw._terms[_key(kw.name)] = term
        for kind, plural in (
            ("synonym", "synonyms"),
            ("acronym", "acronyms"),
            ("variant", "variants"),
            ("pattern", "patterns"),
        ):
            values = data.get(plural) or []
            if isinstance(values, str):
                values = [values]
            kw.add_terms(kind, [Term.from_dict({**v, "kind": kind}) if isinstance(v, Mapping) else v for v in values])  # type: ignore[arg-type]
        return kw

    # ------------------------------------------------------------------ dunder
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Keyword):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Keyword):
            return NotImplemented
        return (self._name.casefold(), self._name) < (other._name.casefold(), other._name)

    def __contains__(self, text: object) -> bool:
        """``"AI" in keyword`` -> is this one of the keyword's (non-pattern) terms?"""
        if not isinstance(text, str):
            return False
        term = self._terms.get(_key(" ".join(text.split())))
        return term is not None and term.kind != "pattern"

    def __repr__(self) -> str:
        parts = [repr(self._name)]
        for label in ("synonyms", "acronyms", "variants", "patterns"):
            values = getattr(self, label)
            if values:
                parts.append(f"{label}={values!r}")
        if self._category:
            parts.append(f"category={self._category!r}")
        return f"Keyword({', '.join(parts)})"

    def __str__(self) -> str:
        return self._name
