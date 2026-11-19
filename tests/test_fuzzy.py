from __future__ import annotations

from keylex import Keyword, KeywordMatcher, Term


def test_typos_multiword_and_spans():
    m = KeywordMatcher(["machine learning", "environment"], "fuzzy")
    text = "Machne lerning helps the enviroment."
    found = m.find(text)
    assert [(x.keyword, x.text, x.method) for x in found] == [
        ("machine learning", "Machne lerning", "fuzzy"),
        ("environment", "enviroment", "fuzzy"),
    ]
    for x in found:
        assert text[x.start : x.end] == x.text and 85 <= x.score < 100


def test_exact_wins_over_fuzzy_and_no_shadow_duplicates():
    m = KeywordMatcher(["environment"], "fuzzy")
    found = m.find("environment is key")
    assert [(x.text, x.method, x.score) for x in found] == [("environment", "exact", 100.0)]


def test_split_and_joined_words():
    m = KeywordMatcher(["database"], "fuzzy", threshold=85)
    assert [x.text for x in m.find("a data base")] == ["data base"]


def test_short_terms_and_acronyms_never_fuzzy():
    m = KeywordMatcher([Keyword("United States", acronyms="USA"), "cat"], "fuzzy", threshold=60)
    assert m.find("USB cut") == []


def test_threshold_and_scorer():
    strict = KeywordMatcher(["environment"], "fuzzy", threshold=99)
    assert strict.find("enviroment") == []
    partial = KeywordMatcher(["environment"], "fuzzy", scorer="partial_ratio", threshold=90)
    assert partial.find("environmental")


def test_per_term_and_per_keyword_fuzzy_switches():
    m = KeywordMatcher([Keyword("environment", fuzzy=False)], "fuzzy")
    assert m.find("enviroment") == []
    m2 = KeywordMatcher([Keyword("env", synonyms=[Term("ecosystem", fuzzy=True)])], "fuzzy")
    assert [x.term for x in m2.find("ecosytem")] == ["ecosystem"]
    m3 = KeywordMatcher([Keyword("feline", synonyms=[Term("cat", fuzzy=True)])], "fuzzy", threshold=60)
    assert [x.text for x in m3.find("cats")] == ["cats"]  # explicit fuzzy=True overrides the length guard


def test_callable_scorer():
    m = KeywordMatcher(["hello"], "fuzzy", scorer=lambda a, b, **_: 100.0 if a[0] == b[0] else 0.0)
    assert [x.text for x in m.find("hat")] == ["hat"]


def test_repeated_windows_scored_once_and_all_reported():
    m = KeywordMatcher(["enviroment"], "fuzzy")
    assert len(m.find(" ".join(["environment"] * 200))) == 200
