"""Regression tests for every bug found in the 0.0.2 diagnostic (B1-B20)."""

from __future__ import annotations

import json

import pytest

from keylex import Keyword, KeywordList, KeywordMatcher, OutputFormat


def test_b1_serialisation_round_trip():
    kl = KeywordList([Keyword("artificial intelligence", synonyms=["machine intelligence"], acronyms=["AI"])])
    for restored in (KeywordList.from_dict(kl.to_dict()), KeywordList.from_json(kl.to_json())):
        kw = restored["artificial intelligence"]
        assert kw.synonyms == ["machine intelligence"]
        assert kw.acronyms == ["AI"]
        assert restored == kl
    # the JSON is versioned and readable
    assert json.loads(kl.to_json())["version"] == 1


def test_b2_fuzzy_matches_typos_and_every_occurrence():
    m = KeywordMatcher(["environment"], "fuzzy")
    assert [x.text for x in m.find("the enviroment is changing")] == ["enviroment"]
    assert len(m.find("environment and environment")) == 2
    long = KeywordMatcher(["environment"], "fuzzy", scorer="ratio")
    assert long.find("the environment is changing quickly today")


def test_b3_count_counts_matches_not_substrings():
    m = KeywordMatcher(["AI", "AI ethics", "ethics"])
    assert m.match("AI ethics matters", "count") == {"AI": 0, "AI ethics": 1, "ethics": 0}
    m_all = KeywordMatcher(["AI", "AI ethics", "ethics"], overlaps="all")
    assert m_all.match("AI ethics matters", "count") == {"AI": 1, "AI ethics": 1, "ethics": 1}


def test_b4_acronyms_are_case_sensitive():
    kw = Keyword("United States", acronyms=["US"])
    m = KeywordMatcher([kw])
    assert m.match("Let us go", "keywords") == []
    assert m.match("The US economy", "keywords") == ["United States"]


def test_b5_empty_keywords_rejected():
    for bad in ("", "   "):
        with pytest.raises(ValueError):
            Keyword(bad)
    with pytest.raises(ValueError):
        Keyword("x", synonyms=[""])


def test_b6_strings_are_not_split_into_characters():
    k = Keyword("x")
    k.add_synonyms("cat")
    k.add_acronyms("CT")
    assert k.synonyms == ["cat"]
    assert k.acronyms == ["CT"]
    k.synonyms = "dog"
    assert k.synonyms == ["dog"]


def test_b7_terms_with_symbols():
    m = KeywordMatcher(["C++", ".NET", "C#"])
    assert m.match("I like C++, C# and .NET", "boolean") == {"C++": True, ".NET": True, "C#": True}
    assert KeywordMatcher(["C"]).find("C++")[0].text != "C++"


def test_b8_remove_by_name():
    kl = KeywordList(["finance"])
    kl.remove(Keyword("finance", synonyms=["money"]))
    assert len(kl) == 0


def test_b9_names_are_read_only_and_rename_is_consistent():
    kl = KeywordList(["old"])
    with pytest.raises(AttributeError):
        kl[0].name = "new"  # type: ignore[misc]
    kl.rename("old", "new")
    assert "old" not in kl and "new" in kl
    kl.add("new")
    assert kl.names == ["new"]


def test_b10_no_duplicate_matches():
    kw = Keyword("AI", synonyms=["ai"])
    assert len(KeywordMatcher([kw]).find("AI rocks")) == 1


def test_b11_overlaps_resolved_by_default():
    m = KeywordMatcher(["AI", "AI ethics", "ethics"])
    assert [x.keyword for x in m.find("AI ethics matters")] == ["AI ethics"]


def test_b12_keywords_are_unhashable():
    with pytest.raises(TypeError):
        {Keyword("z")}


def test_b13_enum_output_formats():
    m = KeywordMatcher(["AI"])
    assert m.match("AI", OutputFormat.COUNT) == {"AI": 1}
    assert m.match("AI", "COUNT") == {"AI": 1}


def test_b14_options_validated_at_construction():
    with pytest.raises(ValueError):
        KeywordMatcher(["a"], match_type="regex")
    with pytest.raises(ValueError):
        KeywordMatcher(["a"], threshold=500)
    with pytest.raises(ValueError):
        KeywordMatcher(["a"], overlaps="sometimes")
    with pytest.raises(ValueError):
        KeywordMatcher(["a"], scorer="nope")


def test_b15_duplicates_merge_and_merge_copies():
    kl = KeywordList(["bank"])
    kl.add(Keyword("bank", synonyms=["lender"]))
    assert kl["bank"].synonyms == ["lender"]
    a, b = KeywordList(), KeywordList(["bank"])
    a.merge(b)
    a[0].add_synonyms("lender")
    assert b[0].synonyms == []


def test_b16_deterministic_order():
    k = Keyword("x", synonyms=["gamma", "alpha", "beta"])
    assert k.synonyms == ["gamma", "alpha", "beta"]
    m = KeywordMatcher(["d", "a", "b", "c"])
    assert m.match("a b c d", "keywords") == ["a", "b", "c", "d"]


def test_b17_equality_and_containment():
    assert KeywordList(["bank"]) != KeywordList([Keyword("bank", synonyms=["lender"])])
    kl = KeywordList([Keyword("bank", synonyms=["lender"])])
    assert "Bank" in kl and "LENDER" in kl and "river" not in kl


def test_b18_type_validation():
    with pytest.raises(TypeError):
        Keyword(123)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        KeywordList([42])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        KeywordMatcher(["a"]).find(None)  # type: ignore[arg-type]


def test_b19_full_ordering():
    a, b = Keyword("a"), Keyword("B")
    assert a < b and a <= b and b > a and b >= a
    assert sorted([b, a]) == [a, b]


def test_b20_whitespace_and_hyphen_variation():
    m = KeywordMatcher(["New York"])
    assert len(m.find("New  York, New\nYork and New-York")) == 3


def test_from_dict_shorthand():
    assert KeywordList.from_dict({"AI": ["machine intelligence", {"acronyms": ["AI"]}]})["AI"].synonyms == [
        "machine intelligence"
    ]
