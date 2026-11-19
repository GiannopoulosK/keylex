"""Exact matching engine: a token trie (fast, boundary-aware) plus a regex path for patterns."""

from __future__ import annotations

import re
from typing import Any

from ..match import Match
from ..text import DASHES, Token, TokenNormalizer, tokenize
from ._compile import CompiledTerm

_END = ""  # tokens are never empty, so "" is a safe terminal key
_SEP_CLASS = "[\\s" + "".join(re.escape(d) for d in sorted(DASHES)) + "]+"


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
        for ct in terms:
            if ct.term.kind == "pattern":
                flags = 0 if ct.case_sensitive else re.IGNORECASE
                self.regex_terms.append((ct, re.compile(ct.term.text, flags)))
            elif not ct.term.whole_word:
                flags = 0 if ct.case_sensitive else re.IGNORECASE
                pattern = _SEP_CLASS.join(re.escape(p) for p in (ct.text or ct.term.text).split())
                self.regex_terms.append((ct, re.compile(pattern, flags)))
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
                                if surface != exact:
                                    continue
                            matches.append(self._make(ct, text, start, end, "exact"))
                    if j >= n:
                        break
                    node = node.get(keys[j])
                    j += 1
        for ct, pattern in self.regex_terms:
            method = "pattern" if ct.term.kind == "pattern" else "exact"
            for m in pattern.finditer(text):
                if m.end() > m.start():
                    matches.append(self._make(ct, text, m.start(), m.end(), method))
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
