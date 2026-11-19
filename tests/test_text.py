from __future__ import annotations

import pytest

from keylex.text import (
    fold_accents,
    light_stem,
    looks_like_acronym,
    paragraph_spans,
    resolve_stemmer,
    sentence_spans,
    tokenize,
)


@pytest.mark.parametrize(
    ("word", "stem"),
    [
        ("banks", "bank"),
        ("policies", "policy"),
        ("boxes", "box"),
        ("classes", "class"),
        ("churches", "church"),
        ("analysis", "analysis"),
        ("status", "status"),
        ("glass", "glass"),
        ("houses", "house"),
        ("is", "is"),
    ],
)
def test_light_stem(word: str, stem: str):
    assert light_stem(word) == stem


def test_tokenize_offsets():
    text = "C++ & U.S. (e-mail)"
    toks = tokenize(text)
    assert [t.text for t in toks] == ["C", "+", "+", "&", "U", ".", "S", ".", "(", "e", "-", "mail", ")"]
    assert all(text[t.start : t.end] == t.text for t in toks)
    assert toks[10].is_dash and toks[0].is_word and not toks[1].is_word


def test_fold_and_acronyms():
    assert fold_accents("Café Zürich") == "Cafe Zurich"
    assert looks_like_acronym("NASA") and looks_like_acronym("G7") and not looks_like_acronym("A")
    assert not looks_like_acronym("Nasa") and not looks_like_acronym("ABCDEFGH")


def test_sentences_and_paragraphs():
    text = 'One. "Two!" Three?\n\nFour'
    assert [text[s:e] for s, e in sentence_spans(text)] == ["One.", '"Two!"', "Three?", "Four"]
    assert [text[s:e] for s, e in paragraph_spans(text)] == ['One. "Two!" Three?', "Four"]


def test_resolve_stemmer():
    assert resolve_stemmer(None) is None
    assert resolve_stemmer("light") is light_stem
    assert resolve_stemmer(str.lower) is str.lower
