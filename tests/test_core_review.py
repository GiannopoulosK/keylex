"""Regression tests for the core-matching issues found in the 0.1.0 review."""

from __future__ import annotations

import json
import time
from collections.abc import Callable

import pytest

from keylex import Keyword, KeywordList, KeywordMatcher
from keylex.matchers import fuzzy as fuzzy_mod
from keylex.text import light_stem, normalize_text, sentence_spans


def spans(matches):
    return [(m.keyword, m.text, m.start, m.end) for m in matches]


# --------------------------------------------------------------------------- unicode
def test_decomposed_text_matches_and_offsets_point_into_original():
    text = "a café here"
    (m,) = KeywordMatcher(["café"]).find(text)
    assert (m.start, m.end) == (2, 7)
    assert m.text == text[2:7] == "café"


def test_decomposed_keyword_name_is_canonicalised():
    assert Keyword("café").name == "café"


def test_fullwidth_and_compat_forms_match_with_nfkc():
    assert spans(KeywordMatcher(["ABC"]).find("x ＡＢＣ y")) == [("ABC", "ＡＢＣ", 2, 5)]
    assert KeywordMatcher(["ABC"], unicode_form=None).find("ＡＢＣ") == []


def test_turkish_dotted_capital_i():
    found = KeywordMatcher(["İstanbul"]).find("istanbul ISTANBUL İstanbul")
    assert [m.text for m in found] == ["istanbul", "ISTANBUL", "İstanbul"]


def test_normalize_text_maps_offsets():
    norm, offsets = normalize_text("éx")
    assert norm == "éx" and offsets is not None
    assert offsets.span(0, 1) == (0, 2) and offsets.span(1, 2) == (2, 3)
    assert normalize_text("plain ascii") == ("plain ascii", None)


def test_invalid_unicode_form():
    with pytest.raises(ValueError, match="unicode_form"):
        KeywordMatcher(["a"], unicode_form="NFD")


# --------------------------------------------------------------------------- stemmer
@pytest.mark.parametrize(
    ("word", "stem"),
    [
        ("movies", "movie"),
        ("cookies", "cookie"),
        ("policies", "policy"),
        ("news", "news"),
        ("does", "does"),
        ("series", "series"),
        ("crises", "crisis"),
        ("buses", "bus"),
        ("banks", "bank"),
        ("analysis", "analysis"),
    ],
)
def test_light_stem(word, stem):
    assert light_stem(word) == stem


def test_stemmer_no_false_positives_and_ie_plurals():
    assert KeywordMatcher(["news"], stemmer="light").find("a new policy") == []
    assert KeywordMatcher(["doe"], stemmer="light").find("she does it") == []
    assert [m.text for m in KeywordMatcher(["movie"], stemmer="light").find("I love movies")] == ["movies"]


# --------------------------------------------------------------------------- sentences & context
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Smith met Dr. Jones at the hospital. Then he left.",
            ["Smith met Dr. Jones at the hospital.", "Then he left."],
        ),
        ("J. K. Rowling wrote it. Fine.", ["J. K. Rowling wrote it.", "Fine."]),
        ("Use e.g. this one. U.S. Treasury rose.", ["Use e.g. this one.", "U.S. Treasury rose."]),
        ("Costs approx. 5 dollars.", ["Costs approx. 5 dollars."]),
        ('He said "Go." She went!', ['He said "Go."', "She went!"]),
        ("Wait... What?", ["Wait...", "What?"]),
        ("One\n\nTwo", ["One", "Two"]),
    ],
)
def test_sentence_spans(text, expected):
    assert [text[a:b] for a, b in sentence_spans(text)] == expected


def test_context_window_does_not_cross_sentences():
    kw = Keyword("Apple", exclude=["pie"])
    text = "apple pie. Apple shares rose."
    assert [m.start for m in KeywordMatcher([kw]).find(text)] == [11]
    assert KeywordMatcher([kw], context_scope="text").find(text) == []


def test_require_context_is_sentence_scoped():
    kw = Keyword("Apple", require=["shares"])
    assert KeywordMatcher([kw]).find("Apple is red. Shares rose.") == []
    assert len(KeywordMatcher([kw]).find("Apple shares rose.")) == 1


# --------------------------------------------------------------------------- acronyms
def test_acronym_case_policies():
    text = "nato Nato NATO NaTo"
    assert [m.text for m in KeywordMatcher(["NATO"]).find(text)] == ["NATO"]
    assert [m.text for m in KeywordMatcher(["NATO"], acronym_case="title").find(text)] == ["Nato", "NATO"]
    assert len(KeywordMatcher(["NATO"], acronym_case="ignore").find(text)) == 4
    # an explicit case_sensitive setting still wins
    strict = Keyword("NATO", case_sensitive=True)
    assert [m.text for m in KeywordMatcher([strict], acronym_case="ignore").find(text)] == ["NATO"]
    with pytest.raises(ValueError, match="acronym_case"):
        KeywordMatcher(["NATO"], acronym_case="lower")


# --------------------------------------------------------------------------- KeywordList IO
def test_csv_round_trip_is_lossless(tmp_path):
    from keylex.keyword import Term

    kw = Keyword(
        'Smith, "Jr"',
        synonyms=["a,b", "semi;colon"],
        acronyms=["SJ"],
        patterns=[r"\b(foo|bar)\b"],
        category="per,son",
        tags=["t1", "t|2"],
        weight=2.5,
        exclude=["not this"],
        require=["yes"],
        context_window=3,
        case_sensitive=True,
        fuzzy=False,
        metadata={"id": 7},
    )
    kw.add_terms("synonym", Term("special", case_sensitive=True, fuzzy=False))
    kl = KeywordList([kw, Keyword("single;item")])
    assert KeywordList.from_csv(kl.to_csv()).to_dict() == kl.to_dict()
    path = tmp_path / "kw.csv"
    kl.save(path)
    assert KeywordList.load(path).to_dict() == kl.to_dict()


def test_load_strips_utf8_bom(tmp_path):
    path = tmp_path / "kw.json"
    path.write_text(KeywordList(["café"]).to_json(), encoding="utf-8-sig")
    assert KeywordList.load(path).names == ["café"]


def test_find_conflicts():
    kl = KeywordList(["Bank", "bank", Keyword("ML", synonyms="machine learning"), "machine learning", "solo"])
    assert kl.find_conflicts() == {"bank": ["Bank", "bank"], "machine learning": ["ML", "machine learning"]}


# --------------------------------------------------------------------------- live recompilation
def test_live_recompile_fast_path_still_sees_changes():
    kl = KeywordList(["alpha"])
    matcher = KeywordMatcher(kl)
    assert len(matcher.find("alpha beta")) == 1
    KeywordList(["unrelated"]).add("other")  # mutation elsewhere: must not break anything
    assert len(matcher.find("alpha beta")) == 1
    kl["alpha"].add_synonyms("beta")
    assert [m.text for m in matcher.find("alpha beta")] == ["alpha", "beta"]
    kl.add("gamma")
    assert len(matcher.find("gamma")) == 1
    matcher.keyword_list = KeywordList(["delta"])
    assert [m.keyword for m in matcher.find("alpha delta")] == ["delta"]


def test_many_short_texts_with_large_vocabulary_is_fast(time_budget: Callable[[float, float], None]):
    matcher = KeywordMatcher([f"term{i}" for i in range(20_000)]).compile()
    texts = [f"hello term{i}" for i in range(2_000)]
    t = time.perf_counter()
    assert sum(len(r) for r in matcher.match_many(texts)) == 2_000
    time_budget(t, 2.0)


# --------------------------------------------------------------------------- fuzzy batching
def test_batched_fuzzy_scoring_equals_unbatched(monkeypatch):
    pytest.importorskip("numpy")
    words = ["inflation", "interest rate", "central bank", "monetary policy", "unemployment"] * 30
    kws = [f"{w} {i}" if i else w for i, w in enumerate(words)]
    text = "The centrl bank kept intrest rates; inflaton and unemploymant rose. monetery policy " * 20
    batched = KeywordMatcher(kws, match_type="fuzzy").find(text)
    monkeypatch.setattr(fuzzy_mod, "_np", None)
    plain = KeywordMatcher(kws, match_type="fuzzy").find(text)
    assert spans(batched) == spans(plain)
    assert [round(m.score, 6) for m in batched] == [round(m.score, 6) for m in plain]
    assert any(m.method == "fuzzy" for m in batched)
    assert len({m.text for m in batched}) >= 4  # typos of several different keywords were found


# --------------------------------------------------------------------------- cross-module follow-ups
def test_matcher_highlight_escapes_html_templates():
    text = "<b>x</b> & y"
    out = KeywordMatcher(["x"]).highlight(text, "<mark>{text}</mark>")
    assert out == "&lt;b&gt;<mark>x</mark>&lt;/b&gt; &amp; y"
    assert KeywordMatcher(["x"]).highlight(text, "<mark>{text}</mark>", escape=False) == "<b><mark>x</mark></b> & y"


def test_cli_extract_validates_options(tmp_path, capsys):
    from keylex.cli import main

    doc = tmp_path / "doc.txt"
    doc.write_text("Central banks raised interest rates. Interest rates matter.", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["extract", "-n", "0", str(doc)])
    assert exc.value.code == 2
    capsys.readouterr()
    assert main(["extract", "-n", "2", "--language", "en", "--diversity", "0", "--json", str(doc)]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 2
