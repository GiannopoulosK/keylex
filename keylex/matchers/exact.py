"""Exact matching engine: a token trie (fast, boundary-aware) plus a regex path for patterns."""

from __future__ import annotations

import re
from typing import Any

from ..match import Match
from ..text import DASHES, Token, TokenNormalizer, fold_accents, fold_accents_with_offsets, tokenize
from ._compile import CompiledTerm

_END = ""  # tokens are never empty, so "" is a safe terminal key
_DASH_CHARS = "".join(re.escape(d) for d in sorted(DASHES))
_SEP_CLASS = "[\\s" + _DASH_CHARS + "]+"
_OPT_SEP_CLASS = "[\\s" + _DASH_CHARS + "]*"
_SPLIT_RE = re.compile(f"(\\s+|[{_DASH_CHARS}]+)")


def substring_pattern(text: str) -> str:
    """Regex for a ``whole_word=False`` term: whitespace matches any run of spaces/dashes and a
    dash is optional, exactly like the token matcher (``"e-mail"`` = ``"e mail"`` = ``"email"``)."""
    out = []
    for i, part in enumerate(_SPLIT_RE.split(text)):
        if i % 2 == 0:
            out.append(re.escape(part))
        else:
            out.append(_SEP_CLASS if part.strip() == "" else _OPT_SEP_CLASS)
    return "".join(out)


def _same_case(token: str, ref: str) -> bool:
    """Case-sensitive comparison that tolerates stemmed endings ("Apples" vs "Apple"): the part
    the two spellings share must have exactly the same case."""
    if token == ref:
        return True
    n = 0
    for a, b in zip(token, ref, strict=False):
        if a.casefold() != b.casefold():
            break
        n += 1
    return n > 0 and token[:n] == ref[:n]


def term_token_variants(text: str, acronym_like: bool) -> list[tuple[str, ...]]:
    """All token sequences under which a term is indexed.

    ``"e-mail"`` -> ``("e", "mail")`` and ``("email",)``; acronym ``"US"`` -> ``("US",)`` and
    ``("U", ".", "S", ".")`` so that "U.S." matches too.
    """
    toks = [t.text for t in tokenize(text)]
    variants: list[tuple[str, ...]] = []
    base = tuple(t for t in toks if t not in DASHES)
    if base:
        variants.append(base)
    if any(t in DASHES for t in toks):
        joined: list[str] = []
        glue = False
        for t in toks:
            if t in DASHES:
                glue = bool(joined)
                continue
            if glue:
                joined[-1] += t
            else:
                joined.append(t)
            glue = False
        if tuple(joined) not in variants:
            variants.append(tuple(joined))
    if acronym_like and text.isalpha() and 2 <= len(text) <= 6:
        dotted: list[str] = []
        for ch in text:
            dotted.extend((ch, "."))
        variants.append(tuple(dotted))
    return variants


class ExactEngine:
    def __init__(self, terms: list[CompiledTerm], normalizer: TokenNormalizer) -> None:
        self.normalizer = normalizer
        self.trie: dict[str, Any] = {}
        self.regex_terms: list[tuple[CompiledTerm, re.Pattern[str]]] = []
        #: whole_word=False terms; matched on accent-folded text when ignore_accents is on
        self.substring_terms: list[tuple[CompiledTerm, re.Pattern[str]]] = []
        for ct in terms:
            if ct.term.kind == "pattern":
                flags = 0 if ct.case_sensitive else re.IGNORECASE
                self.regex_terms.append((ct, re.compile(ct.term.text, flags)))
            elif not ct.term.whole_word:
                flags = 0 if ct.case_sensitive else re.IGNORECASE
                text = ct.text or ct.term.text
                if normalizer.ignore_accents:
                    text = fold_accents(text)
                self.substring_terms.append((ct, re.compile(substring_pattern(text), flags)))
            else:
                for raw in term_token_variants(ct.text or ct.term.text, ct.term.is_acronym_like):
                    node = self.trie
                    for tok in raw:
                        node = node.setdefault(normalizer(tok), {})
                    exact = tuple(normalizer.exact(t) for t in raw)
                    hits = node.setdefault(_END, [])
                    hits.append((ct, exact))
                    if ct.title_ok and len(raw) == 1:
                        hits.append((ct, (exact[0].capitalize(),)))

    def find(self, text: str, tokens: list[Token] | None = None) -> list[Match]:
        if tokens is None:
            tokens = tokenize(text)
        toks = [t for t in tokens if not t.is_dash]
        norm = self.normalizer
        stemmed = norm.stemmer is not None
        keys = [norm(t.text) for t in toks]
        matches: list[Match] = []
        trie = self.trie
        n = len(toks)
        if trie:
            for i in range(n):
                node = trie.get(keys[i])
                j = i + 1
                while node is not None:
                    hits = node.get(_END)
                    if hits:
                        start, end = toks[i].start, toks[j - 1].end
                        surface: tuple[str, ...] | None = None
                        for ct, exact in hits:
                            if ct.case_sensitive:
                                if surface is None:
                                    surface = tuple(norm.exact(t.text) for t in toks[i:j])
                                if surface != exact and not (stemmed and all(map(_same_case, surface, exact))):
                                    continue
                            matches.append(self._make(ct, text, start, end, "exact"))
                    if j >= n:
                        break
                    node = node.get(keys[j])
                    j += 1
        for ct, pattern in self.regex_terms:
            for m in pattern.finditer(text):
                if m.end() > m.start():
                    matches.append(self._make(ct, text, m.start(), m.end(), "pattern"))
        if self.substring_terms:
            haystack, offsets = text, None
            if norm.ignore_accents:
                haystack, offsets = fold_accents_with_offsets(text)
            for ct, pattern in self.substring_terms:
                for m in pattern.finditer(haystack):
                    if m.end() > m.start():
                        start, end = m.span() if offsets is None else offsets.span(*m.span())
                        matches.append(self._make(ct, text, start, end, "exact"))
        return matches

    @staticmethod
    def _make(ct: CompiledTerm, text: str, start: int, end: int, method: str) -> Match:
        kw = ct.keyword
        return Match(
            keyword=kw.name,
            start=start,
            end=end,
            text=text[start:end],
            term=ct.term.text,
            kind=ct.term.kind,
            method=method,
            score=100.0,
            category=kw.category,
            keyword_obj=kw,
        )
