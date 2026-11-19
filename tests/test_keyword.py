from __future__ import annotations

import pytest

from keylex import Keyword, KeywordList, Term


def test_construction_and_terms():
    kw = Keyword("artificial intelligence", "machine intelligence", ["AI"], variants="A.I.", category="tech")
    assert kw.name == kw.keyword == "artificial intelligence"
    assert [t.kind for t in kw.terms] == ["keyword", "synonym", "acronym", "variant"]
    assert kw.related_terms == ["machine intelligence", "AI", "A.I."]
    assert kw.has_synonyms and kw.has_acronyms and kw.has_related_terms()
    assert "ai" in kw and "Machine  Intelligence" in kw and "robot" not in kw


def test_whitespace_normalised_and_dedup_case_insensitive():
    kw = Keyword("  New   York ", synonyms=["new york", "NYC", "nyc"])
    assert kw.name == "New York"
    assert kw.synonyms == ["NYC"]


def test_varargs_and_lists():
    kw = Keyword("x")
    kw.add_synonyms("a", "b")
    kw.add_synonyms(["c", "d"])
    kw.remove_synonyms("a", ["c"])
    assert kw.synonyms == ["b", "d"]
    kw.set_synonyms(["z"])
    assert kw.synonyms == ["z"]
    kw.clear_synonyms()
    assert not kw.has_synonyms


def test_related_terms_management():
    kw = Keyword("x")
    kw.add_related_terms(synonyms=["s"], acronyms=["XX"], variants=["xs"])
    assert kw.related_terms == ["s", "XX", "xs"]
    kw.remove_related_terms(acronyms="XX")
    assert kw.acronyms == []
    kw.related_terms = {"synonyms": ["q"]}
    assert kw.related_terms == ["q"]
    kw.clear_related_terms()
    assert not kw.has_related_terms()


def test_patterns_validated():
    kw = Keyword("order id", patterns=[r"ORD-\d{4}"])
    assert kw.patterns == [r"ORD-\d{4}"]
    with pytest.raises(ValueError):
        Keyword("bad", patterns=["("])


def test_term_objects_and_options():
    kw = Keyword("Apple", synonyms=[Term("AAPL", case_sensitive=True, fuzzy=False)])
    term = kw.terms[1]
    assert term.kind == "synonym" and term.case_sensitive is True and term.fuzzy is False
    with pytest.raises(ValueError):
        Term("x", kind="nope")  # type: ignore[arg-type]
    assert Term("NASA").is_acronym_like and not Term("Nasa").is_acronym_like


def test_round_trip_with_all_fields():
    kw = Keyword(
        "Apple",
        synonyms=[Term("apple inc", fuzzy=False)],
        acronyms="AAPL",
        variants=["Apple's"],
        patterns=[r"\bAAPL\.O\b"],
        case_sensitive=True,
        fuzzy=False,
        exclude=["fruit"],
        require=["stock", "shares"],
        context_window=5,
        category="company",
        tags=["tech", "nasdaq"],
        weight=2.5,
        metadata={"ticker": "AAPL"},
    )
    data = kw.to_dict()
    assert Keyword.from_dict(data) == kw
    copy = kw.copy()
    assert copy == kw and copy is not kw
    copy.add_synonyms("other")
    assert copy != kw


def test_from_dict_rejects_unknown_fields():
    with pytest.raises(ValueError):
        Keyword.from_dict({"keyword": "x", "synonym": ["typo in field name"]})
    with pytest.raises(ValueError):
        Keyword.from_dict({"synonyms": []})


def test_option_validation():
    with pytest.raises(ValueError):
        Keyword("x", context_window=-1)
    with pytest.raises(TypeError):
        Keyword("x", weight="heavy")  # type: ignore[arg-type]


def test_renamed_copy():
    kw = Keyword("old", synonyms=["s"])
    new = kw.renamed("new")
    assert new.name == "new" and new.synonyms == ["s"] and kw.name == "old"


def test_version_changes_on_mutation():
    kw = Keyword("x")
    kl = KeywordList([kw])
    sig = kl.signature()
    kw.add_synonyms("y")
    assert kl.signature() != sig
    sig = kl.signature()
    kw.add_synonyms("y")  # no-op
    assert kl.signature() == sig


def test_repr_and_str():
    kw = Keyword("x", synonyms=["y"], category="c")
    assert repr(kw) == "Keyword('x', synonyms=['y'], category='c')"
    assert str(kw) == "x"
