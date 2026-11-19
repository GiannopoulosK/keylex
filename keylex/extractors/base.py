"""Shared machinery for unsupervised keyword extraction."""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from bisect import bisect_left
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from rapidfuzz import fuzz

from ..keyword import Keyword
from ..keyword_list import KeywordList
from ..stopwords import get_stopwords
from ..text import light_stem, sentence_spans

__all__ = ["BaseExtractor", "ExtractedKeyword", "PreparedText", "Word"]

_MARKS = "̀-ͯ"  # combining accents (decomposed text) stay inside a word
_WORD_RE = re.compile(
    rf"(?:[^\W\d_]\.){{2,}}(?![^\W_])"  # abbreviations: U.S., e.g.
    rf"|(?<![\w.])\.[^\W\d_][\w{_MARKS}]*"  # .NET, .js
    rf"|\d+(?:[.,]\d+)*(?![\w&+#{_MARKS}])"  # plain numbers: 2023, 5.3, 1,000
    # words, incl. inner - ' & . (state-of-the-art, don't, AT&T, Node.js), digits (5G, COVID-19)
    # and a trailing ++ or # (C++, C#)
    rf"|[^\W_][\w{_MARKS}]*(?:[-'’&.][^\W_][\w{_MARKS}]*)*(?:\+\+|#(?!\w))?"
    r"|(?<=\s)&(?=\s)"  # a free-standing "&" ("Research & Development") is a connector token
)

#: Phrases made only of capitalised words ("ECB President Christine Lagarde") may exceed ``max_ngram``
#: up to this many words.
MAX_NAME_LENGTH = 6

#: Part-of-speech tags (Universal POS) allowed in a phrase when a POS tagger is supplied.
_NP_TAGS = frozenset({"NOUN", "PROPN", "ADJ", "NUM", "X", "SYM"})
_NP_HEADS = frozenset({"NOUN", "PROPN", "X", "SYM"})

# --- connectors: stopwords allowed *inside* a phrase -------------------------------------------------
#: "of"-like words: "Department of Defense", "state of the art".
_OF_WORDS: dict[str, frozenset[str]] = {
    "en": frozenset({"of"}),
    "fr": frozenset({"de", "du", "des", "d"}),
    "es": frozenset({"de", "del"}),
    "it": frozenset({"di", "del", "della", "dei", "degli", "delle"}),
    "de": frozenset({"der", "des", "für", "von"}),
    "el": frozenset({"του", "της", "των"}),
}
#: Articles allowed right after an "of"-like word ("Department of the Treasury").
_ARTICLES = frozenset({"the", "la", "le", "les", "l", "el", "los", "las", "il", "lo", "der", "die", "das", "den"})
#: Coordinators allowed only between capitalised words ("Johnson and Johnson", "Licensor and Licensee").
_AND_WORDS = frozenset({"and", "for", "und", "et", "y", "e", "και"})
#: Name particles allowed between capitalised words ("Charles de Gaulle", "Ludwig van Beethoven").
_PARTICLES = frozenset({"de", "la", "le", "du", "des", "von", "van", "der", "den", "da", "di", "del", "dos", "das"})


@dataclass(frozen=True)
class ExtractedKeyword:
    """A keyword found by an extractor. Higher ``score`` is better (normalised to ``(0, 1]``)."""

    phrase: str
    score: float
    count: int = 1
    positions: tuple[int, ...] = field(default=(), repr=False, compare=False)

    def to_keyword(self, **kwargs: object) -> Keyword:
        return Keyword(self.phrase, **kwargs)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {"phrase": self.phrase, "score": round(self.score, 4), "count": self.count}


@dataclass(frozen=True)
class Word:
    surface: str
    lower: str
    start: int
    end: int
    sentence: int
    chunk: int  # punctuation-delimited chunk id; phrases never cross chunks
    is_stop: bool
    is_number: bool
    sentence_start: bool
    weak_end: bool  # past-tense verbs / adverbs: a keyphrase should not end with one
    possessive: bool = False  # "Apple's": a phrase may end here but not continue past it
    acronym: bool = False  # all-caps "US", "WHO", "AT&T": never a stopword, keeps its case as key
    pos: str | None = None  # Universal POS tag when a tagger was supplied

    @property
    def key(self) -> str:
        """Identity used for statistics: lowercase, except acronyms ("US" != "us")."""
        return self.surface if self.acronym else self.lower

    @property
    def capitalised(self) -> bool:
        return self.surface[:1].isupper()

    @property
    def bad_end(self) -> bool:
        return self.is_stop or self.is_number or self.weak_end or (self.pos is not None and self.pos not in _NP_HEADS)


def _is_weak_end(lower: str) -> bool:
    """Heuristic stand-in for POS tagging: "raised", "quickly" rarely end a keyphrase."""
    if len(lower) <= 4:
        return False
    return (lower.endswith("ed") and not lower.endswith("eed")) or lower.endswith("ly")


def leading_modifier_ok(chunk: Sequence[Word], i: int) -> bool:
    """Can the "-ed" word ``chunk[i]`` lead a phrase? It is an adjective after a determiner or
    preposition ("a distributed store", "of containerized apps") but a verb after a noun ("banks
    raised rates") or before a name ("praised Apple")."""
    if i + 1 >= len(chunk) or chunk[i + 1].capitalised:
        return False
    return i > 0 and chunk[i - 1].is_stop and not chunk[i - 1].acronym


def _is_acronym(surface: str) -> bool:
    letters = [c for c in surface if c.isalpha()]
    return len(letters) >= 2 and surface.isupper() and len(surface) <= 8


class PreparedText:
    """Tokenised view of a text used by all extractors."""

    def __init__(
        self,
        text: str,
        stopwords: frozenset[str],
        min_word_length: int,
        *,
        language: str = "en",
        pos_tags: Sequence[tuple[int, int, str]] | None = None,
    ) -> None:
        self.text = text
        self.language = language
        self.stopwords = stopwords
        self.words: list[Word] = []
        spans = sentence_spans(text) or [(0, len(text))]
        self.n_sentences = len(spans)
        self._build(text, spans, stopwords, min_word_length)
        self._fix_sentence_start_case()
        if pos_tags is not None:
            self._apply_pos(pos_tags)
        self._of = _OF_WORDS.get(language, frozenset())

    def _build(self, text: str, spans: list[tuple[int, int]], stopwords: frozenset[str], min_word_length: int) -> None:
        english = self.language == "en"
        chunk = 0
        for s_idx, (s, e) in enumerate(spans):
            chunk += 1
            matches = list(_WORD_RE.finditer(text, s, e))
            # in an all-caps sentence ("BREAKING NEWS: ...") capitals don't signal acronyms
            alpha = [m.group() for m in matches if any(c.isalpha() for c in m.group())]
            shouting = len(alpha) >= 3 and sum(t.isupper() for t in alpha) > len(alpha) / 2
            prev_end: int | None = None
            first = True
            for m in matches:
                if prev_end is not None and text[prev_end : m.start()].strip():
                    chunk += 1  # punctuation between words breaks a phrase
                surface = m.group()
                lower = surface.lower().replace("’", "'")
                possessive = False
                if len(lower) > 2 and lower.endswith("'s"):
                    lower, surface, possessive = lower[:-2], surface[:-2], True
                is_number = not any(c.isalpha() for c in surface) and surface != "&"
                acronym = not shouting and _is_acronym(surface)
                self.words.append(
                    Word(
                        surface=surface,
                        lower=lower,
                        start=m.start(),
                        end=m.start() + len(surface),
                        sentence=s_idx,
                        chunk=chunk,
                        is_stop=not acronym
                        and (lower in stopwords or len(lower) < min_word_length or not any(c.isalnum() for c in lower)),
                        is_number=is_number,
                        sentence_start=first,
                        weak_end=english and not acronym and _is_weak_end(lower),
                        possessive=possessive,
                        acronym=acronym,
                    )
                )
                first = False
                prev_end = m.end()

    def _fix_sentence_start_case(self) -> None:
        """Lowercase a sentence-initial capital only if the word is seen lowercase elsewhere
        ("Central banks ..." -> "central banks", but "Fed officials ..." keeps "Fed")."""
        seen_lower = {w.lower for w in self.words if w.surface[:1].islower()}
        self.words = [
            replace(w, surface=w.surface[0].lower() + w.surface[1:])
            if w.sentence_start and w.lower in seen_lower and not w.surface.isupper()
            else w
            for w in self.words
        ]

    def _apply_pos(self, tags: Sequence[tuple[int, int, str]]) -> None:
        """Attach the tag of the last tagger token inside each word (the head of "AT&T", "Node.js")."""
        tags = sorted(tags)
        starts = [t[0] for t in tags]
        out = []
        for w in self.words:
            i = bisect_left(starts, w.end) - 1
            pos = None
            while i >= 0 and tags[i][1] > w.start:
                if tags[i][0] < w.end:
                    pos = tags[i][2]
                    break
                i -= 1
            out.append(replace(w, pos=pos))
        self.words = out

    def chunks(self) -> Iterator[list[Word]]:
        """Runs of words not separated by punctuation."""
        current: list[Word] = []
        for w in self.words:
            if current and w.chunk != current[-1].chunk:
                yield current
                current = []
            current.append(w)
        if current:
            yield current

    # -- phrase shape -----------------------------------------------------------------------------
    def connector_ok(self, chunk: Sequence[Word], k: int) -> bool:
        """May the stopword ``chunk[k]`` appear *inside* a phrase (it has neighbours on both sides)?"""
        w, prev, nxt = chunk[k], chunk[k - 1], chunk[k + 1]
        low = w.lower
        if low == "&":
            return not prev.is_stop and not nxt.is_stop
        if low in self._of:
            return not nxt.is_stop or nxt.lower in _ARTICLES
        if low in _ARTICLES and prev.lower in self._of:
            return not nxt.is_stop and not nxt.is_number
        if low in _AND_WORDS:
            return (
                prev.capitalised
                and nxt.capitalised
                and prev.surface.isalpha()
                and nxt.surface.isalpha()
                and not prev.acronym
                and not nxt.acronym
                and not prev.is_stop
                and not nxt.is_stop
            )
        if low in _PARTICLES:
            j = k + 1
            while j < len(chunk) - 1 and chunk[j].lower in _PARTICLES:
                j += 1
            return (prev.capitalised or prev.lower in _PARTICLES) and chunk[j].capitalised and not chunk[j].is_stop
        return False

    def valid_phrase(self, chunk: Sequence[Word], i: int, j: int) -> bool:
        """Is ``chunk[i:j]`` (j exclusive) a well-formed candidate phrase?"""
        first, last = chunk[i], chunk[j - 1]
        if first.is_stop or first.is_number or last.bad_end:
            return False
        if first.pos is not None and first.pos not in _NP_TAGS:
            return False
        for k in range(i, j - 1):
            w = chunk[k]
            if w.possessive:
                return False  # "Apple's iPhone": the possessive ends a phrase
            if k > i:
                if w.is_stop and not self.connector_ok(chunk, k):
                    return False
                if w.weak_end:
                    return False  # "banks raised interest": verbs only allowed as a leading modifier
                if w.pos is not None and not w.is_stop and w.pos not in _NP_TAGS:
                    return False
        if j - i > 1 and last.weak_end:
            return False
        if j - i != 3 and any(w.lower in _AND_WORDS for w in chunk[i + 1 : j - 1]):
            return False  # "Licensor and Licensee", but not "Samsung Electronics and Google"
        return not (j - i > 1 and first.weak_end and not leading_modifier_ok(chunk, i))

    def candidates(self, max_n: int, *, names: bool = True) -> Iterator[tuple[list[Word], int]]:
        """All well-formed n-grams (1..max_n) inside chunks, plus whole capitalised runs up to
        :data:`MAX_NAME_LENGTH` words when ``names`` is true. Yields ``(words, index_of_first_word)``.
        """
        index = 0
        for chunk in self.chunks():
            size = len(chunk)
            run = self._name_runs(chunk) if names else [0] * size
            run_len = Counter(run)
            for i in range(size):
                for n in range(1, max_n + 1):
                    j = i + n
                    if j > size:
                        break
                    if names and (self._cuts_name(run, run_len, i, j) or self._cuts_connector_phrase(chunk, i, j)):
                        continue
                    if self.valid_phrase(chunk, i, j):
                        yield chunk[i:j], index + i
                if names and run[i] and (i == 0 or run[i - 1] != run[i]):
                    end = i + 1
                    while end < size and run[end] == run[i]:
                        end += 1
                    if max_n < end - i <= MAX_NAME_LENGTH and self.valid_phrase(chunk, i, end):
                        yield chunk[i:end], index + i
            index += size

    @staticmethod
    def _is_name_word(w: Word) -> bool:
        return w.capitalised and not w.is_stop and not w.is_number

    def _name_runs(self, chunk: Sequence[Word]) -> list[int]:
        """Run id (> 0) of every word in a run of capitalised words, which may contain connectors
        ("State of New York", "Charles de Gaulle"); ``0`` elsewhere. A possessive ends a run."""
        run = [0] * len(chunk)
        current = 0
        k = 0
        while k < len(chunk):
            w = chunk[k]
            if not self._is_name_word(w):
                k += 1
                continue
            current += 1
            start = k
            while k < len(chunk):
                if self._is_name_word(chunk[k]):
                    run[k] = current
                    k += 1
                    if chunk[k - 1].possessive:
                        break
                    continue
                # connectors inside a name, followed by another name word
                j = k
                while (
                    j < len(chunk) - 1
                    and chunk[j].is_stop
                    and chunk[j].lower not in _AND_WORDS
                    and self.connector_ok(chunk, j)
                ):
                    j += 1
                if j > k and self._is_name_word(chunk[j]) and k > start:
                    for m in range(k, j):
                        run[m] = current
                    k = j
                    continue
                break
        return run

    @staticmethod
    def _cuts_name(run: list[int], run_len: Counter[int], i: int, j: int) -> bool:
        """``[i, j)`` starts or ends in the middle of a (not overly long) name: "ECB President Christine"."""
        rid = run[i] or run[j - 1]
        if not rid:
            return False
        if run_len[rid] > MAX_NAME_LENGTH:
            return False
        before = i > 0 and run[i] and run[i - 1] == run[i]
        after = j < len(run) and run[j - 1] and run[j] == run[j - 1]
        return bool(before or after)

    @staticmethod
    def _cuts_connector_phrase(chunk: Sequence[Word], i: int, j: int) -> bool:
        """A phrase with an inner connector must be complete: "set of worker" (nodes) is cut off."""
        if j - i < 3 or j >= len(chunk):
            return False
        nxt = chunk[j]
        if chunk[j - 1].capitalised:
            return False  # names are handled by _cuts_name ("Research & Development budgets" is fine)
        return any(w.is_stop for w in chunk[i + 1 : j - 1]) and not nxt.is_stop and not nxt.is_number


def _norm(key: tuple[str, ...]) -> str:
    """Spelling-insensitive identity: "e-mail" == "email", "neural networks" == "neural network"."""
    return " ".join(light_stem(t.lower().replace("-", "")) for t in key).replace(" ", "")


def _similarity(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


class _Collector:
    """Aggregates occurrences of candidate phrases (key = tuple of :attr:`Word.key`)."""

    def __init__(self) -> None:
        self.count: Counter[tuple[str, ...]] = Counter()
        self.surfaces: dict[tuple[str, ...], Counter[str]] = {}
        self.positions: dict[tuple[str, ...], list[int]] = {}

    def add(self, words: Sequence[Word]) -> tuple[str, ...]:
        key = tuple(w.key for w in words)
        self.count[key] += 1
        surface = " ".join(w.surface for w in words)
        self.surfaces.setdefault(key, Counter())[surface] += 1
        self.positions.setdefault(key, []).append(words[0].start)
        return key

    def surface(self, key: tuple[str, ...]) -> str:
        return self.surfaces[key].most_common(1)[0][0]


def _check_top_n(value: object, name: str = "top_n") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer (or None for the default), got {value!r}")
    return value


class BaseExtractor(ABC):
    """Base class for extractors.

    Args:
        top_n: How many keywords :meth:`extract` returns by default (a positive integer).
        max_ngram: Longest phrase (in words) to consider. Runs of capitalised words (names such as
            "ECB President Christine Lagarde") may be longer, up to :data:`MAX_NAME_LENGTH` words.
        stopwords: Iterable of stopwords; overrides ``language``.
        language: Built-in stopword list to use (``"en"`` default, ``"de"``, ``"fr"``, ``"es"``,
            ``"it"``, ``"el"``). English also enables the verb heuristics.
        min_word_length: Shorter words are treated as stopwords (all-caps acronyms excepted).
        deduplicate: Drop near-duplicates of better-ranked phrases (plurals, "e-mail"/"email", typos,
            shorter phrases that only occur inside a longer one).
        dedup_threshold: Similarity (0-100) above which two phrases are duplicates.
        diversity: Maximal Marginal Relevance trade-off in ``[0, 1]``: ``0`` returns the plain ranking,
            higher values prefer phrases that share fewer words with those already selected
            ("Apple iPhone" vs "Apple iPhone sales"). Results are still returned best score first.
        pos_filter: Optional spaCy ``Language`` with a tagger (e.g. ``spacy.load("en_core_web_sm")``)
            or any callable ``text -> tokens`` with ``.idx``, ``.text`` and ``.pos_``. Only
            noun-phrase-shaped candidates (adjectives/nouns ending in a noun) are kept.
    """

    name = "base"

    def __init__(
        self,
        top_n: int = 10,
        *,
        max_ngram: int = 3,
        stopwords: Iterable[str] | None = None,
        language: str = "en",
        min_word_length: int = 2,
        deduplicate: bool = True,
        dedup_threshold: float = 90,
        diversity: float = 0.3,
        pos_filter: Callable[[str], Any] | None = None,
    ) -> None:
        self.top_n = _check_top_n(top_n)
        if isinstance(max_ngram, bool) or not isinstance(max_ngram, int) or max_ngram < 1:
            raise ValueError("max_ngram must be >= 1")
        if not 0 <= diversity <= 1:
            raise ValueError("diversity must be between 0 and 1")
        if pos_filter is not None and not callable(pos_filter):
            raise TypeError("pos_filter must be a spaCy Language (or a callable text -> tokens)")
        self.max_ngram = max_ngram
        self.language = language.lower().replace("_", "-").split("-")[0]
        if isinstance(stopwords, str):
            raise TypeError("stopwords must be an iterable of words, not a string")
        self.stopwords = (
            frozenset(s.lower() for s in stopwords) if stopwords is not None else get_stopwords(self.language)
        )
        self.min_word_length = min_word_length
        self.deduplicate = deduplicate
        self.dedup_threshold = dedup_threshold
        self.diversity = diversity
        self.pos_filter = pos_filter

    def _pos_tags(self, text: str) -> list[tuple[int, int, str]] | None:
        if self.pos_filter is None:
            return None
        doc = self.pos_filter(text)
        has_annotation = getattr(doc, "has_annotation", None)
        if callable(has_annotation) and not has_annotation("POS"):
            raise ValueError(
                "pos_filter needs a pipeline with a part-of-speech tagger "
                "(e.g. spacy.load('en_core_web_sm')); spacy.blank() has none"
            )
        return [(t.idx, t.idx + len(t.text), t.pos_) for t in doc]

    def prepare(self, text: str) -> PreparedText:
        return PreparedText(
            text, self.stopwords, self.min_word_length, language=self.language, pos_tags=self._pos_tags(text)
        )

    @abstractmethod
    def _score(self, prepared: PreparedText) -> tuple[dict[tuple[str, ...], float], _Collector]:
        """Return ``{phrase key: score (higher is better)}`` and the occurrence collector."""

    def extract(self, text: str, top_n: int | None = None) -> list[ExtractedKeyword]:
        """Return the ``top_n`` best keywords/keyphrases of ``text``, best first.

        ``top_n=None`` uses the extractor's default; anything but a positive integer raises ``ValueError``.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a str")
        n = self.top_n if top_n is None else _check_top_n(top_n)
        prepared = self.prepare(text)
        if not prepared.words:
            return []
        scores, collector = self._score(prepared)
        if not scores:
            return []
        first_pos = {k: min(collector.positions[k]) for k in scores}
        ranked = sorted(scores, key=lambda k: (-scores[k], -collector.count[k], first_pos[k], k))
        best = scores[ranked[0]] or 1.0
        pool_size = n if self.diversity <= 0 else max(3 * n, n + 20)
        pool = self._dedupe(ranked, scores, collector, pool_size)
        chosen = self._diversify(pool, n) if self.diversity > 0 else pool[:n]
        out = [
            ExtractedKeyword(
                collector.surface(key), score / best, collector.count[key], tuple(collector.positions[key])
            )
            for key, score in chosen
        ]
        return sorted(out, key=lambda k: -k.score)

    def _dedupe(
        self, ranked: list[tuple[str, ...]], scores: dict[tuple[str, ...], float], collector: _Collector, limit: int
    ) -> list[tuple[tuple[str, ...], float]]:
        """Best-first candidates without near-duplicates; a shorter phrase that never occurs outside a
        longer, lower-ranked one is replaced by it (keeping the better score)."""
        if not self.deduplicate:
            return [(k, scores[k]) for k in ranked[:limit]]
        count = collector.count
        out: list[tuple[tuple[str, ...], float]] = []
        seen: list[tuple[str, str]] = []  # (normalised, lower)
        for key in ranked:
            if any(_is_subsequence(key, c) and count[key] <= count[c] for c, _ in out):
                continue  # never occurs outside a better-ranked phrase
            lower = " ".join(key).lower()
            norm = _norm(key)
            if any(norm == s or fuzz.ratio(lower, low) >= self.dedup_threshold for s, low in seen):
                continue
            absorbed = [i for i, (c, _) in enumerate(out) if _is_subsequence(c, key) and count[c] <= count[key]]
            if absorbed:
                first = absorbed[0]
                out[first] = (key, out[first][1])
                seen[first] = (norm, lower)
                for i in reversed(absorbed[1:]):
                    del out[i], seen[i]
                continue
            seen.append((norm, lower))
            out.append((key, scores[key]))
            if len(out) >= limit:
                break
        return out

    def _diversify(self, pool: list[tuple[tuple[str, ...], float]], n: int) -> list[tuple[tuple[str, ...], float]]:
        """Maximal Marginal Relevance over the rank-ordered pool (rank-based relevance keeps it
        independent of each algorithm's score scale)."""
        if len(pool) <= 1:
            return pool[:n]
        words = [frozenset(_norm((t,)) for t in key) for key, _ in pool]
        size = len(pool)
        relevance = [1.0 - i / size for i in range(size)]
        selected = [0]
        max_sim = [_similarity(words[0], words[i]) for i in range(size)]
        remaining = set(range(1, size))
        lam = self.diversity
        while remaining and len(selected) < n:
            best = max(remaining, key=lambda i: ((1 - lam) * relevance[i] - lam * max_sim[i], -i))
            selected.append(best)
            remaining.discard(best)
            for i in remaining:
                max_sim[i] = max(max_sim[i], _similarity(words[best], words[i]))
        return [pool[i] for i in selected]

    def extract_many(self, texts: Iterable[str], top_n: int | None = None) -> Iterator[list[ExtractedKeyword]]:
        for text in texts:
            yield self.extract(text, top_n)

    def to_keyword_list(self, text: str, top_n: int | None = None) -> KeywordList:
        """Extract and wrap the results in a :class:`KeywordList` ready for matching.

        Every phrase is a surface form of ``text``, so ``KeywordMatcher(kl).find(text)`` finds it again.
        """
        return KeywordList(
            [Keyword(k.phrase, metadata={"score": round(k.score, 4)}) for k in self.extract(text, top_n)]
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(top_n={self.top_n}, max_ngram={self.max_ngram})"


def _is_subsequence(short: tuple[str, ...], long: tuple[str, ...]) -> bool:
    n = len(short)
    return n < len(long) and any(long[i : i + n] == short for i in range(len(long) - n + 1))
