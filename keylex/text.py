"""Text utilities shared by matchers, rules and extractors.

Everything here is dependency-free and works on plain ``str`` objects. Character offsets always
refer to the *original* text, so matches can be mapped back exactly.
"""

from __future__ import annotations

import functools
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "DASHES",
    "STEMMERS",
    "Token",
    "TokenNormalizer",
    "fold_accents",
    "fold_accents_with_offsets",
    "OffsetMap",
    "light_stem",
    "looks_like_acronym",
    "normalize_text",
    "paragraph_spans",
    "resolve_stemmer",
    "sentence_spans",
    "tokenize",
    "word_tokens",
]

#: Characters treated as hyphens/dashes. They are transparent when matching ("New-York" == "New York").
DASHES = frozenset("-‐‑‒–—―﹘﹣－")

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_WORD_RE = re.compile(r"\w+")
_SENTENCE_END_RE = re.compile(r"[.!?…]+[\"'”’)\]]*\s+|\n\s*\n")
_PREV_WORD_RE = re.compile(r"(\w+(?:\.\w+)*)\.$")
#: Abbreviations after which a period does not end a sentence (compared case-insensitively,
#: without the final period). Single capital letters (initials, "J. Smith") are handled separately.
ABBREVIATIONS = frozenset(
    """mr mrs ms dr prof sr jr st mt ft rev hon gen col lt sgt capt cmdr gov sen rep pres
    vs etc al eg ie cf approx est dept univ assn bros inc ltd co corp plc llc no nos vol fig figs
    pp ch sec art para jan feb mar apr jun jul aug sep sept oct nov dec mon tue wed thu fri sat sun
    e.g i.e u.s u.k u.n e.u u.s.a a.m p.m ph.d b.a m.a b.sc m.sc n.b""".split()
)
_PARAGRAPH_RE = re.compile(r"\n\s*\n")


@dataclass(frozen=True, slots=True)
class Token:
    """A token with character offsets into the original text."""

    text: str
    start: int
    end: int

    @property
    def is_word(self) -> bool:
        return bool(_WORD_RE.fullmatch(self.text))

    @property
    def is_dash(self) -> bool:
        return self.text in DASHES


def tokenize(text: str) -> list[Token]:
    """Split text into word tokens (``\\w+``) and single punctuation tokens."""
    return [Token(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def word_tokens(text: str) -> list[Token]:
    """Return only the word tokens (``\\w+``) of ``text``."""
    return [Token(m.group(), m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def fold_accents(text: str) -> str:
    """Strip diacritics: ``"Café Zürich"`` -> ``"Cafe Zurich"``."""
    if text.isascii():
        return text
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def fold_accents_with_offsets(text: str) -> tuple[str, OffsetMap | None]:
    """:func:`fold_accents` plus a map from the folded string back to ``text`` offsets."""
    if text.isascii():
        return text, None
    out: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for i, ch in enumerate(text):
        folded = ch if ch.isascii() else _fold_char(ch)  # a lone combining mark folds to ""
        out.append(folded)
        if len(folded) == 1:
            starts.append(i)
            ends.append(i + 1)
        elif folded:
            starts.extend([i] * len(folded))
            ends.extend([i + 1] * len(folded))
    return "".join(out), OffsetMap(starts, ends)


@functools.lru_cache(maxsize=4096)
def _fold_char(ch: str) -> str:
    return fold_accents(ch)


#: Words that end in "s" but are not plurals (or whose plural is irregular for the rules below).
_STEM_INVARIANT = frozenset(
    """news series species means does goes has was is this thus yes gas bias alias atlas canvas
    lens chaos christmas physics mathematics economics politics ethics athletics logistics
    linguistics genetics electronics aerobics gymnastics measles diabetes herpes rabies""".split()
)
#: Plurals whose singular ends in "ie" (movies -> movie, not movy) or that need an explicit form.
_STEM_EXCEPTIONS = {
    **{
        w + "s": w
        for w in """movie cookie pie tie lie die calorie zombie hippie rookie selfie
    smoothie brownie goalie hoodie lingerie prairie rotisserie sortie veggie auntie birdie
    budgie collie freebie genie newbie pixie talkie yuppie""".split()
    },
    "buses": "bus",
    "gases": "gas",
    "lenses": "lens",
    "biases": "bias",
    "aliases": "alias",
    "atlases": "atlas",
    "viruses": "virus",
    "bonuses": "bonus",
    "campuses": "campus",
    "statuses": "status",
    "censuses": "census",
    "crises": "crisis",
    "theses": "thesis",
    "analyses": "analysis",
    "hypotheses": "hypothesis",
    "diagnoses": "diagnosis",
    "syntheses": "synthesis",
    "parentheses": "parenthesis",
    "emphases": "emphasis",
}


def light_stem(word: str) -> str:
    """A conservative English plural stemmer (an extended S-stemmer).

    It only removes plural endings, so it is safe to apply to both the text and the keywords::

        banks -> bank, policies -> policy, boxes -> box, classes -> class, analysis -> analysis
        movies -> movie, news -> news, crises -> crisis
    """
    w = word
    if w in _STEM_INVARIANT:
        return w
    special = _STEM_EXCEPTIONS.get(w)
    if special is not None:
        return special
    n = len(w)
    if n > 4 and w.endswith("ies") and not w.endswith(("eies", "aies")):
        return w[:-3] + "y"
    if n > 4 and w.endswith(("sses", "xes", "ches", "shes", "zzes")):
        return w[:-2]
    if n > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is", "'s")):
        return w[:-1]
    return w


STEMMERS: dict[str, Callable[[str], str]] = {"light": light_stem, "plural": light_stem}


def resolve_stemmer(stemmer: str | Callable[[str], str] | None) -> Callable[[str], str] | None:
    """Turn a stemmer name (``"light"``) or callable into a callable (or ``None``)."""
    if stemmer is None or callable(stemmer):
        return stemmer
    try:
        return STEMMERS[stemmer]
    except KeyError:
        raise ValueError(
            f"Unknown stemmer {stemmer!r}. Use one of {sorted(STEMMERS)} or pass a callable "
            "(e.g. nltk.stem.PorterStemmer().stem)."
        ) from None


def looks_like_acronym(text: str) -> bool:
    """``True`` for short all-caps tokens such as ``"US"``, ``"NASA"``, ``"G7"``."""
    # cheap checks first: this runs for every term at compile time
    return 2 <= len(text) <= 6 and text.isupper() and text.isalnum()


class OffsetMap:
    """Maps character offsets in a normalised string back to the original string."""

    __slots__ = ("_ends", "_starts")

    def __init__(self, starts: list[int], ends: list[int]) -> None:
        self._starts = starts
        self._ends = ends

    def span(self, start: int, end: int) -> tuple[int, int]:
        """Original ``(start, end)`` for a span of the normalised text."""
        if end <= start:
            pos = self._starts[start] if start < len(self._starts) else (self._ends[-1] if self._ends else 0)
            return pos, pos
        return self._starts[start], self._ends[end - 1]


#: Invisible word-internal characters removed before matching: soft hyphen, zero-width
#: (non-)joiner, word joiner, BOM and variation selectors, so that an emoji with or without U+FE0F
#: and words with soft hyphens inside them match.
_IGNORABLE = "\u00ad\u200c\u200d\u2060\ufeff" + "".join(map(chr, range(0xFE00, 0xFE10)))
#: Typographic apostrophes are folded to "'" (so "Don\u2019t" matches "don't"), and the zero-width
#: space, a word *boundary*, becomes a real space. Both keep the text length.
_SAME_LENGTH = {"\u2018": "'", "\u2019": "'", "\u02bc": "'", "\u200b": " "}
_PREP_TABLE = str.maketrans({**dict.fromkeys(_IGNORABLE), **_SAME_LENGTH})
_NEEDS_PREP_RE = re.compile(f"[{_IGNORABLE}{''.join(_SAME_LENGTH)}]")
_IGNORABLE_RE = re.compile(f"[{_IGNORABLE}]")
#: Characters with combining class 0 that still compose with the preceding character (Hangul
#: vowel/trailing jamo, some Indic, Tibetan and Myanmar vowel signs). Computed from the Unicode
#: canonical decompositions; only these need the (slow) composition check.
_COMPOSING_STARTERS = frozenset(
    map(
        chr,
        [
            *range(0x1161, 0x1176),
            *range(0x11A8, 0x11C3),
            *(0x9BE, 0x9D7, 0xB3E, 0xB56, 0xB57, 0xBBE, 0xBD7, 0xCC2, 0xCD5, 0xCD6, 0xD3E, 0xD57),
            *(0xDCF, 0xDDF, 0xFB5, 0xFB7, 0x102E, 0x1B35, 0x11127, 0x1133E, 0x11357, 0x114B0),
            *(0x114BA, 0x114BD, 0x115AF, 0x11930),
        ],
    )
)


def _composes(text: str, i: int, j: int, form: str) -> bool:
    """Does ``text[j]`` (a non-combining character) merge with the cluster ``text[i:j]`` under
    ``form``? True for Hangul conjoining jamo and some Indic vowel signs."""
    ch = text[j]
    if ch not in _COMPOSING_STARTERS:
        return False
    norm = unicodedata.normalize
    return norm(form, text[i : j + 1]) != norm(form, text[i:j]) + norm(form, ch)  # type: ignore[arg-type]


def _strip_ignorables(text: str) -> tuple[str, OffsetMap]:
    """Delete :data:`_IGNORABLE` characters (and apply the same-length replacements) from an
    already-normalised text, mapping offsets without a per-character Python loop."""
    starts: list[int] = []
    ends: list[int] = []
    pos = 0
    for m in _IGNORABLE_RE.finditer(text):
        starts.extend(range(pos, m.start()))
        ends.extend(range(pos + 1, m.start() + 1))
        pos = m.end()
    starts.extend(range(pos, len(text)))
    ends.extend(range(pos + 1, len(text) + 1))
    return text.translate(_PREP_TABLE), OffsetMap(starts, ends)


def normalize_text(text: str, form: str | None = "NFKC") -> tuple[str, OffsetMap | None]:
    """Unicode-normalise ``text`` and return a map back to the original offsets.

    Besides ``form`` normalisation, invisible format characters (soft hyphens, joiners, variation
    selectors) are removed, zero-width spaces become spaces and typographic apostrophes become ``'``.

    Returns ``(text, None)`` when nothing changes (the common case, checked cheaply). Characters
    are normalised in clusters (a character plus everything that combines with it), so every
    cluster maps to exactly one span of the original text.
    """
    if form is None or text.isascii():
        return text, None
    if unicodedata.is_normalized(form, text):  # type: ignore[arg-type]
        if not _NEEDS_PREP_RE.search(text):
            return text, None
        if not _IGNORABLE_RE.search(text):
            return text.translate(_PREP_TABLE), None  # same-length replacements only: same offsets
        return _strip_ignorables(text)
    out: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    n = len(text)
    i = 0
    while i < n:
        j = i + 1
        while j < n and (unicodedata.combining(text[j]) or _composes(text, i, j, form)):
            j += 1
        chunk = unicodedata.normalize(form, text[i:j]).translate(_PREP_TABLE)  # type: ignore[arg-type]
        out.append(chunk)
        starts.extend([i] * len(chunk))
        ends.extend([j] * len(chunk))
        i = j
    return "".join(out), OffsetMap(starts, ends)


class TokenNormalizer:
    """Normalises a single token for case-insensitive comparison.

    Applies ``casefold`` and, optionally, accent folding and a stemmer.
    """

    __slots__ = ("_cache", "ignore_accents", "stemmer")

    def __init__(self, ignore_accents: bool = False, stemmer: Callable[[str], str] | None = None) -> None:
        self.ignore_accents = ignore_accents
        self.stemmer = stemmer
        self._cache: dict[str, str] = {}

    def __call__(self, token: str) -> str:
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        out = token.casefold()
        if "̇" in out:  # Turkish dotted capital I: "İ".casefold() == "i̇"
            out = out.replace("i̇", "i")
        if self.ignore_accents:
            out = fold_accents(out)
        if self.stemmer is not None and out.isalpha():
            out = self.stemmer(out)
        if len(self._cache) < 200_000:
            self._cache[token] = out
        return out

    def exact(self, token: str) -> str:
        """Case-preserving normalisation (used for case-sensitive comparisons)."""
        return fold_accents(token) if self.ignore_accents else token


def _split_spans(text: str, pattern: re.Pattern[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in pattern.finditer(text):
        if text[pos : m.start()].strip():
            spans.append((pos, m.start()))
        pos = m.end()
    if text[pos:].strip():
        spans.append((pos, len(text)))
    return spans


def _is_false_boundary(text: str, m: re.Match[str]) -> bool:
    """``True`` when a period-based break is not a sentence end (abbreviation, initial, ...)."""
    punct = m.group().rstrip()
    if not punct.startswith(".") or punct.startswith(".."):
        return False  # "!", "?", ellipses and blank lines always end a sentence
    prev = _PREV_WORD_RE.search(text, max(0, m.start() - 20), m.start() + 1)
    if prev is None:
        return False
    word = prev.group(1)
    if len(word) == 1 and word.isalpha() and word.isupper():
        return True  # an initial: "J. Smith"
    return word.casefold() in ABBREVIATIONS


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Approximate sentence boundaries as ``(start, end)`` character spans.

    Periods after common abbreviations ("Dr.", "U.S.", "e.g.") and initials ("J. Smith") do not end a sentence.
    Blank lines always end a sentence.
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in _SENTENCE_END_RE.finditer(text):
        if _is_false_boundary(text, m):
            continue
        cut = m.start() + len(m.group().rstrip())
        if text[pos:cut].strip():
            spans.append((pos, cut))
        pos = m.end()
    if text[pos:].strip():
        spans.append((pos, len(text)))
    return spans


def paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Paragraph boundaries (blocks separated by blank lines) as ``(start, end)`` spans."""
    return _split_spans(text, _PARAGRAPH_RE)
