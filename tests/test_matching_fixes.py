"""Regression tests for the matcher / keyword / keyword-list fixes of the 2026-11 release review."""

from __future__ import annotations

import pickle
import threading
import time
import unicodedata
from pathlib import Path

import pytest

from keylex import Keyword, KeywordList, KeywordMatcher, Term, highlight
from keylex.match import Match
from keylex.matchers._postprocess import drop_fuzzy_shadowed
from keylex.text import normalize_text


def spans(matcher: KeywordMatcher, text: str) -> list[tuple[str, str]]:
    return [(m.keyword, m.text) for m in matcher.find(text)]


# ---------------------------------------------------------------- keyword files
def test_save_txt_writes_the_text_format(tmp_path: Path) -> None:
    kl = KeywordList([Keyword("C#", synonyms=["csharp"]), "F#", Keyword("python", synonyms=["py3", "py"])])
    path = tmp_path / "kw.txt"
    kl.save(path)
    assert path.read_text(encoding="utf-8") == "C# | csharp\nF#\npython | py3 | py\n"
    assert KeywordList.load(path) == kl


def test_save_txt_refuses_to_lose_data(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="acronyms"):
        KeywordList([Keyword("United States", acronyms=["US"])]).save(tmp_path / "kw.txt")
    assert not (tmp_path / "kw.txt").exists()


def test_save_and_load_yaml_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    kl = KeywordList([Keyword("Apple", acronyms=["AAPL"], exclude=["pie"], weight=2, metadata={"id": 1}), "pear"])
    kl.save(tmp_path / "kw.yaml")
    assert KeywordList.load(tmp_path / "kw.yaml") == kl


def test_unknown_suffix_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        KeywordList(["a"]).save(tmp_path / "kw.xml")
    (tmp_path / "kw.xml").write_text("a", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        KeywordList.load(tmp_path / "kw.xml")


def test_hash_inside_keywords_is_not_a_comment() -> None:
    kl = KeywordList.from_text("C# | csharp\nF#  # functional\n# a comment\n#hashtag-free line\nGo #lang")
    assert kl.names == ["C#", "F#", "Go"]
    assert kl["C#"].synonyms == ["csharp"]


@pytest.mark.parametrize(("bom", "codec"), [("\ufeff", "utf-16"), ("\ufeff", "utf-8"), ("", "utf-8")])
def test_load_detects_byte_order_marks(tmp_path: Path, bom: str, codec: str) -> None:
    path = tmp_path / "kw.txt"
    path.write_bytes((bom + "café | coffee\n").encode(codec) if codec == "utf-8" else "café | coffee\n".encode(codec))
    assert KeywordList.load(path).names == ["café"]


def test_load_decode_error_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "kw.txt"
    path.write_bytes("café".encode("latin-1"))
    with pytest.raises(ValueError, match=r"kw.txt"):
        KeywordList.load(path)


def test_csv_round_trip_keeps_whitespace_in_patterns() -> None:
    kl = KeywordList([Keyword("x", patterns=[r"foo\s "]), Keyword("y", patterns=[" bar"])])
    back = KeywordList.from_csv(kl.to_csv())
    assert back["x"].patterns == [r"foo\s "]
    assert back["y"].patterns == [" bar"]
    assert back == kl


def test_yaml_style_non_string_names_get_a_clear_error() -> None:
    with pytest.raises(TypeError, match="quote"):
        KeywordList.from_dict({True: ["affirmative"]})  # type: ignore[dict-item]


def test_a_keyword_named_keywords_is_not_a_document() -> None:
    kl = KeywordList.from_dict({"keywords": ["kw", "key words"]})
    assert kl.names == ["keywords"]
    assert kl["keywords"].synonyms == ["kw", "key words"]
    assert KeywordList.from_json(kl.to_json()) == kl


def test_save_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "kw.txt"
    KeywordList(["old"]).save(path)
    with pytest.raises(ValueError):
        KeywordList([Keyword("x", acronyms=["X1"])]).save(path)
    assert path.read_text(encoding="utf-8") == "old\n"
    assert [p.name for p in tmp_path.iterdir()] == ["kw.txt"]  # no temp files left behind


# ---------------------------------------------------------------- Keyword
def test_synonym_starting_with_re_colon_does_not_hide_a_pattern() -> None:
    kw = Keyword("x", synonyms=["re:foo"], patterns=["foo"])
    assert kw.synonyms == ["re:foo"]
    assert kw.patterns == ["foo"]
    assert Keyword.from_dict(kw.to_dict()) == kw


@pytest.mark.parametrize("phrase", ["!!", "++", "--"])
def test_context_phrases_need_a_word(phrase: str) -> None:
    with pytest.raises(ValueError, match="never be found"):
        Keyword("Apple", exclude=[phrase])
    with pytest.raises(ValueError, match="never be found"):
        Keyword("Apple", require=phrase)


@pytest.mark.parametrize("term", ["-", "—", "- -"])
def test_dash_only_terms_are_rejected(term: str) -> None:
    with pytest.raises(ValueError, match="dashes"):
        Keyword("dash", synonyms=[term])


def test_ordering_is_total_and_by_name() -> None:
    a, b = Keyword("apple"), Keyword("apple", synonyms=["pomme"])
    assert a != b
    assert a <= b and a >= b and not a < b and not a > b
    assert sorted([Keyword("b"), Keyword("A"), Keyword("a")]) == [Keyword("A"), Keyword("a"), Keyword("b")]


# ---------------------------------------------------------------- matcher options
def test_changing_options_after_construction_takes_effect() -> None:
    m = KeywordMatcher(["machine learning"], "fuzzy", threshold=95)
    assert m.find("machin lerning") == []
    m.threshold = 70
    assert [x.keyword for x in m.find("machin lerning")] == ["machine learning"]

    m2 = KeywordMatcher(["Apple"])
    assert m2.find("apple")
    m2.case_sensitive = True
    assert m2.find("apple") == []
    m2.configure(case_sensitive=False, stemmer="light")
    assert [x.text for x in m2.find("apples")] == ["apples"]


def test_configure_validates_and_rolls_back() -> None:
    m = KeywordMatcher(["a b"], threshold=80)
    with pytest.raises(ValueError):
        m.configure(threshold=150, overlaps="all")
    assert m.threshold == 80 and m.overlaps == "longest"
    with pytest.raises(TypeError, match="Unknown option"):
        m.configure(treshold=3)
    assert KeywordMatcher(["a b"], **m.options()).options() == m.options()


def test_invalid_attribute_assignment_is_reported_on_next_call() -> None:
    m = KeywordMatcher(["a"])
    m.find("a")
    m.overlaps = "widest"
    with pytest.raises(ValueError, match="overlaps"):
        m.find("a")


def test_matcher_pickles() -> None:
    m = KeywordMatcher([Keyword("United States", acronyms=["US"])], stemmer="light")
    m.find("U.S.")
    clone = pickle.loads(pickle.dumps(m))
    assert spans(clone, "the U.S. and US") == spans(m, "the U.S. and US")


def test_concurrent_find_while_keywords_change() -> None:
    kl = KeywordList(["alpha"])
    matcher = KeywordMatcher(kl)
    errors: list[BaseException] = []
    stop = time.monotonic() + 1.0

    def reader() -> None:
        try:
            while time.monotonic() < stop:
                assert matcher.find("alpha beta")[0].keyword == "alpha"
        except BaseException as exc:  # pragma: no cover - only on failure
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    i = 0
    while time.monotonic() < stop:
        kl.add(f"kw{i}")
        kl["alpha"].add_synonyms(f"syn{i}")
        i += 1
    for t in threads:
        t.join()
    assert errors == []


# ---------------------------------------------------------------- exact matching
def test_case_sensitive_terms_follow_the_stemmer() -> None:
    m = KeywordMatcher([Keyword("Apple", case_sensitive=True)], stemmer="light")
    assert [x.text for x in m.find("Apples are tasty; apples too")] == ["Apples"]
    policy = KeywordMatcher([Keyword("Policy", case_sensitive=True)], stemmer="light")
    assert [x.text for x in policy.find("Policies, POLICIES, policies")] == ["Policies"]


def test_substring_terms_handle_dashes_and_accents() -> None:
    kw = Keyword("coffee", synonyms=[Term("cafe", whole_word=False), Term("a-b", whole_word=False)])
    m = KeywordMatcher([kw], ignore_accents=True)
    assert [x.text for x in m.find("supercafé")] == ["café"]
    assert [x.text for x in m.find("xa bx xa–bx xabx")] == ["a b", "a–b", "ab"]


@pytest.mark.parametrize(
    ("text", "surface"),
    [
        ("I ❤\ufe0f NY", "❤\ufe0f NY"),  # emoji variation selector
        ("co\u00adoperation", "co\u00adoperation"),  # soft hyphen
        ("co\u200boperation", "co\u200boperation"),  # zero-width space
    ],
)
def test_invisible_characters_are_transparent(text: str, surface: str) -> None:
    m = KeywordMatcher(["❤ NY", "cooperation"])
    assert [x.text for x in m.find(text)] == [surface]


def test_typographic_apostrophes_match() -> None:
    m = KeywordMatcher(["don't panic", "O'Brien"])
    assert [x.text for x in m.find("Don\u2019t panic, said O\u2019Brien")] == ["Don\u2019t panic", "O\u2019Brien"]


def test_hangul_jamo_compose_like_real_nfkc() -> None:
    text = unicodedata.normalize("NFD", "한국 경제")
    normalized, offsets = normalize_text(text)
    assert normalized == "한국 경제"
    assert offsets is not None
    m = KeywordMatcher(["한국"])
    found = m.find(text)
    assert [unicodedata.normalize("NFC", x.text) for x in found] == ["한국"]
    assert found[0].text == text[found[0].start : found[0].end]


# ---------------------------------------------------------------- fuzzy matching
@pytest.mark.parametrize(
    "text",
    ["I fixed the machine. Learning is fun.", "the machine, learning", "machine (learning)"],
)
def test_fuzzy_windows_do_not_cross_boundaries(text: str) -> None:
    assert KeywordMatcher(["machine learning"], "fuzzy").find(text) == []


def test_fuzzy_windows_still_cross_abbreviations_and_matching_punctuation() -> None:
    m = KeywordMatcher(["Dr. Smithson", "Smith, Johnson"], "fuzzy")
    assert {x.keyword for x in m.find("Dr. Smithsen met Smith, Jonson")} == {"Dr. Smithson", "Smith, Johnson"}


def test_drop_fuzzy_shadowed_is_not_quadratic() -> None:
    exact = [Match("k", i * 10, i * 10 + 5, "x", "x") for i in range(20_000)]
    fuzzy = [Match("k", i * 10 + 6, i * 10 + 9, "y", "y", method="fuzzy") for i in range(20_000)]
    overlapping = Match("k", 3, 7, "z", "z", method="fuzzy")
    t0 = time.perf_counter()
    kept = drop_fuzzy_shadowed([*exact, *fuzzy, overlapping])
    assert time.perf_counter() - t0 < 2
    assert len(kept) == 40_000 and overlapping not in kept


def test_fuzzy_on_repetitive_text_is_fast() -> None:
    m = KeywordMatcher(["machine learning"], "fuzzy")
    t0 = time.perf_counter()
    assert len(m.find("machine learning is fun. " * 8000)) == 8000
    assert time.perf_counter() - t0 < 5


# ---------------------------------------------------------------- highlight
def test_bad_highlight_template_lists_the_fields() -> None:
    m = KeywordMatcher(["bank"])
    with pytest.raises(ValueError, match="available fields"):
        m.highlight("a bank", "<{foo}>")
    assert highlight("a bank", m.find("a bank"), "[{kind}:{text}]") == "a [keyword:bank]"
