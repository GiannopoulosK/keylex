from __future__ import annotations

from collections.abc import Callable

import pytest

from keylex import ExactMatcher, FuzzyMatcher, Keyword, KeywordList, KeywordMatcher, Match, Term, highlight


def texts(matches: list[Match]) -> list[str]:
    return [m.text for m in matches]


def test_match_object_fields():
    kw = Keyword("United States", acronyms="US", category="country")
    (m,) = KeywordMatcher([kw]).find("Visit the U.S. today")
    assert (m.keyword, m.text, m.term, m.kind, m.method, m.category) == (
        "United States",
        "U.S.",
        "US",
        "acronym",
        "exact",
        "country",
    )
    assert m.span == (10, 14) and len(m) == 4 and m.keyword_obj is kw and m.score == 100
    assert m.to_dict()["start"] == 10
    assert hash(m) == hash(Match("United States", 10, 14, "U.S.", "US", "acronym", "exact", 100.0, "country"))


def test_accepts_many_input_types():
    assert KeywordMatcher("python").contains("I love Python")
    assert KeywordMatcher({"bank": ["lender"]}).contains("a lender")
    assert KeywordMatcher([{"keyword": "bank", "synonyms": ["lender"]}]).contains("a lender")
    with pytest.raises(TypeError):
        KeywordMatcher(42)  # type: ignore[arg-type]


def test_word_boundaries():
    m = KeywordMatcher(["cat"])
    assert texts(m.find("cat category concat cat.")) == ["cat", "cat"]
    sub = KeywordMatcher([Keyword("biology", synonyms=[Term("bio", whole_word=False)])])
    assert texts(sub.find("biotech")) == ["bio"]


def test_case_sensitivity_resolution():
    assert KeywordMatcher(["NASA"]).find("nasa") == []  # acronym-looking -> case-sensitive
    assert KeywordMatcher([Keyword("NASA", case_sensitive=False)]).find("nasa")
    assert KeywordMatcher(["Python"], case_sensitive=True).find("python") == []
    assert KeywordMatcher([Keyword("Python", synonyms=[Term("py", case_sensitive=True)])]).find("PY") == []


def test_possessives_and_punctuation():
    m = KeywordMatcher(["Apple"])
    assert texts(m.find("Apple's results (Apple) Apple-based")) == ["Apple", "Apple", "Apple"]


def test_hyphen_variants():
    m = KeywordMatcher(["e-mail", "state-of-the-art"])
    found = texts(m.find("e-mail, email, e mail, stateoftheart and state of the art"))
    assert found == ["e-mail", "email", "e mail", "stateoftheart", "state of the art"]


def test_regex_patterns():
    kw = Keyword("order", patterns=[r"ORD-\d{4}"])
    (m,) = KeywordMatcher([kw]).find("ref ord-1234")
    assert m.method == "pattern" and m.kind == "pattern" and m.text == "ord-1234"


def test_overlap_strategies():
    kl = KeywordList(["New York", "York", "New York City", "City Hall"])
    text = "New York City Hall"
    assert [m.keyword for m in KeywordMatcher(kl).find(text)] == ["New York City"]
    assert [m.keyword for m in KeywordMatcher(kl, overlaps="leftmost").find(text)] == ["New York City"]
    assert len(KeywordMatcher(kl, overlaps="all").find(text)) == 4


def test_identical_spans_from_different_keywords_both_kept():
    kl = KeywordList([Keyword("Apple Inc", synonyms="apple"), Keyword("apple fruit", synonyms="apple")])
    assert sorted(m.keyword for m in KeywordMatcher(kl).find("an apple")) == ["Apple Inc", "apple fruit"]


def test_context_exclude_and_require():
    company = Keyword("Apple", exclude=["pie", "fruit"], context_window=3)
    m = KeywordMatcher([company])
    assert m.find("I baked an Apple pie") == []
    assert len(m.find("Apple shares rose. Later that week I ate fruit salad with friends")) == 1
    river = Keyword("bank", require=["river", "water"], context_window=4)
    m2 = KeywordMatcher([river])
    assert texts(m2.find("We walked along the bank of the river. Later on, the downtown bank closed early.")) == [
        "bank"
    ]


def test_stemmer_and_accents():
    m = KeywordMatcher(["policy", "box"], stemmer="light")
    assert texts(m.find("Policies and boxes")) == ["Policies", "boxes"]
    m2 = KeywordMatcher(["cafe"], ignore_accents=True)
    assert texts(m2.find("Café au lait")) == ["Café"]
    assert KeywordMatcher(["cafe"]).find("Café") == []
    m3 = KeywordMatcher(["run"], stemmer=lambda w: w[:3])
    assert texts(m3.find("running runs")) == ["running", "runs"]
    with pytest.raises(ValueError):
        KeywordMatcher(["x"], stemmer="porter-ish")


def test_output_formats():
    kl = KeywordList([Keyword("bank", category="finance", weight=2), Keyword("river", category="nature"), "unused"])
    m = KeywordMatcher(kl)
    text = "The bank by the river. Another bank."
    assert len(m.match(text)) == 3
    assert m.match(text, "detailed")[0]["keyword"] == "bank"
    assert m.match(text, "keywords") == ["bank", "river"]
    assert m.match(text, "boolean") == {"bank": True, "river": True, "unused": False}
    assert m.count(text) == {"bank": 2, "river": 1, "unused": 0}
    assert m.match(text, "categories") == {"finance": 2, "nature": 1}
    assert m.score(text) == 5.0
    with pytest.raises(ValueError):
        m.match(text, "xml")


def test_match_many_and_call():
    m = KeywordMatcher(["a"])
    assert list(m.match_many(["a", "b", "a a"], "count")) == [{"a": 1}, {"a": 0}, {"a": 2}]
    assert m("a") == m.find("a")


def test_live_keyword_list_updates():
    kl = KeywordList(["alpha"])
    m = KeywordMatcher(kl)
    assert m.match("alpha beta", "keywords") == ["alpha"]
    kl.add("beta")
    assert m.match("alpha beta", "keywords") == ["alpha", "beta"]
    kl["beta"].add_synonyms("gamma")
    assert m.match("gamma", "keywords") == ["beta"]
    kl.remove("alpha")
    assert m.match("alpha", "keywords") == []


def test_highlight_styles():
    m = KeywordMatcher(["bank", "river"])
    text = "bank & <river>"
    assert m.highlight(text) == "[bank] & <[river]>"
    assert (
        m.highlight(text, "html")
        == '<mark data-keyword="bank">bank</mark> &amp; &lt;<mark data-keyword="river">river</mark>&gt;'
    )
    assert m.highlight(text, "{text}:{keyword}") == "bank:bank & <river:river>"
    assert highlight(text, m.find(text), lambda x: x.text.upper()) == "BANK & <RIVER>"
    with pytest.raises(ValueError):
        m.highlight(text, "neon")


def test_exact_and_fuzzy_subclasses():
    assert ExactMatcher(["x"]).match_type == "exact"
    assert FuzzyMatcher(["x"], threshold=70).threshold == 70
    assert "KeywordMatcher(1 keywords" in repr(ExactMatcher(["x"]))


def test_large_vocabulary_is_fast(time_budget: Callable[[float, float], None]):
    import time

    words = [f"term{i}" for i in range(5000)]
    text = " ".join(words[::7] * 20)
    m = KeywordMatcher(words)
    t = time.perf_counter()
    found = m.find(text)
    assert len(found) == len(words[::7]) * 20
    time_budget(t, 5)
