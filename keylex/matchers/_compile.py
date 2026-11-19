"""Resolution of per-term matching options."""

from __future__ import annotations

from dataclasses import dataclass

from ..keyword import Keyword, Term
from ..keyword_list import KeywordList
from ..text import normalize_text


@dataclass(frozen=True, slots=True)
class CompiledTerm:
    keyword: Keyword
    term: Term
    case_sensitive: bool
    fuzzy: bool
    #: The term text after Unicode normalisation (what the engines index).
    text: str = ""
    #: Acronym that may also appear in Title case ("Nato" for "NATO").
    title_ok: bool = False


def resolve_terms(
    keywords: KeywordList,
    *,
    case_sensitive: bool,
    min_fuzzy_length: int,
    unicode_form: str | None = "NFKC",
    acronym_case: str = "strict",
) -> list[CompiledTerm]:
    """Apply the precedence ``Term > Keyword > acronym heuristic > matcher default``.

    ``acronym_case`` controls acronym-like terms that don't set ``case_sensitive`` explicitly:
    ``"strict"`` (only the exact spelling), ``"title"`` (also ``Nato`` for ``NATO``) or
    ``"ignore"`` (any case, like ordinary terms).
    """
    out: list[CompiledTerm] = []
    for kw in keywords:
        for term in kw.iter_terms():
            title_ok = False
            if term.case_sensitive is not None:
                cs = term.case_sensitive
            elif kw.case_sensitive is not None:
                cs = kw.case_sensitive
            elif term.is_acronym_like and acronym_case != "ignore":
                cs = True
                title_ok = acronym_case == "title" and term.text.isalpha()
            else:
                cs = case_sensitive

            if term.kind == "pattern":
                fz = False
            elif term.fuzzy is not None:
                fz = term.fuzzy
            elif kw.fuzzy is not None:
                fz = kw.fuzzy and not term.is_acronym_like and len(term.text) >= min_fuzzy_length
            else:
                fz = not term.is_acronym_like and len(term.text) >= min_fuzzy_length
            text = term.text
            if term.kind != "pattern":
                # the same normalisation as the text (NFKC, invisible characters, apostrophes)
                text = normalize_text(text, unicode_form)[0]
            out.append(CompiledTerm(kw, term, cs, fz, text, title_ok))
    return out
