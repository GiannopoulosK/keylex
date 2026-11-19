"""Regression tests for the extraction / spaCy / highlight review (items 1-13)."""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from keylex import Keyword, KeywordList, KeywordMatcher, Term, extract_keywords, get_extractor, highlight
from keylex.extractors import EXTRACTORS, RakeExtractor, TfidfExtractor, YakeExtractor
from keylex.match import Match
from keylex.stopwords import EXTRACTION_STOPWORDS, GREEK_STOPWORDS, get_stopwords

ARTICLE = (Path(__file__).resolve().parents[1] / "examples" / "article.txt").read_text(encoding="utf-8")
NEWS = (
    "The European Central Bank raised interest rates by a quarter point on Thursday, its tenth consecutive "
    "increase. ECB President Christine Lagarde said the bank was not yet done. Inflation in the euro zone "
    "slowed to 5.3% in July, still well above the ECB's 2% target. Analysts at Deutsche Bank said further "
    "rate hikes would depend on incoming data. Lagarde warned that wage growth could keep inflation elevated."
)
TECH = (
    "Kubernetes is an open-source container orchestration platform. A Kubernetes cluster consists of a "
    "control plane and a set of worker nodes. The control plane runs the API server and etcd, a distributed "
    "key-value store. Services expose pods through a stable virtual IP address and DNS name. The economy "
    "slows while the control plane runs."
)
POSSESSIVE = "Apple's iPhone sales rose. Analysts praised Apple's iPhone lineup. The company's CEO's plan was bold."
TECHNICAL = (
    "AT&T rolled out 5G. AT&T says 5G and R&D in C++ and C# and Node.js and .NET and 3D printing matter. "
    "5G again. Node.js rocks. 3D printing grows. e-mail and email marketing. E-mail marketing."
)
ACRONYMS = "The WHO warned. US Treasury agree on rules. IT budgets grow. The WHO and the US Treasury met."
TEXTS = [ARTICLE, NEWS, TECH, POSSESSIVE, TECHNICAL, ACRONYMS]


def phrases(text: str, method: str = "yake", n: int = 10, **kwargs: object) -> list[str]:
    return [k.phrase for k in extract_keywords(text, n, method, **kwargs)]


# 1. every extracted phrase is a real surface form -------------------------------------------------


@pytest.mark.parametrize("method", sorted(EXTRACTORS))
@pytest.mark.parametrize("text", TEXTS, ids=["article", "news", "tech", "possessive", "technical", "acronyms"])
def test_extracted_phrases_are_found_again_by_the_matcher(method: str, text: str):
    kl = get_extractor(method, top_n=15).to_keyword_list(text)
    assert len(kl)
    found = {m.keyword for m in KeywordMatcher(kl, overlaps="all").find(text)}
    assert found == set(kl.names)


def test_possessives_end_a_phrase():
    for method in EXTRACTORS:
        out = phrases(POSSESSIVE, method, 15)
        assert not any("Apple iPhone" in p or "company CEO" in p or "CEO plan" in p for p in out), (method, out)
        assert "Apple" in out and "iPhone sales" in out


# 2. acronyms are not stopwords ----------------------------------------------------------------------


@pytest.mark.parametrize("method", sorted(EXTRACTORS))
def test_all_caps_acronyms_are_not_stopwords(method: str):
    out = phrases(ACRONYMS, method, 15)
    assert "WHO" in out or any("WHO" in p.split() for p in out)
    assert any("US" in p.split() for p in out)
    assert any("IT" in p.split() for p in out)
    assert "Treasury agree" not in out
    # lowercase "us" / "it" are still stopwords
    assert not any(
        p.lower() in ("us", "it") and p.islower() for p in phrases("Tell us about it. Give it to us.", method)
    )


def test_all_caps_sentence_is_not_a_run_of_acronyms():
    out = phrases("BREAKING NEWS ABOUT THE ECONOMY TODAY. The economy grows.", "yake")
    assert not any(p.split()[0].lower() in ("about", "the") for p in out)


# 3. RAKE keeps long runs ------------------------------------------------------------------------------


def test_rake_splits_long_runs_instead_of_dropping_them():
    text = "Machine learning model training pipeline optimization framework."
    out = RakeExtractor(max_ngram=4).extract(text)
    assert out
    assert all(len(k.phrase.split()) <= 4 for k in out)
    assert " ".join(k.phrase for k in sorted(out, key=lambda k: k.positions[0])) == text[:-1]


# 4. technical terms are single tokens -------------------------------------------------------------------


def test_technical_terms_survive_tokenisation():
    out = set(phrases(TECHNICAL, "yake", 20))
    for term in ("AT&T", "5G", "R&D", "C++", "C#", "Node.js", ".NET", "3D printing"):
        assert term in out or any(term in p for p in out), term
    assert "js" not in out and "NET" not in out and "Node" not in out


def test_ampersand_connector():
    out = phrases("Research & Development budgets grew. Research & Development matters.", "yake", 5)
    assert "Research & Development" in out


# 5. top_n validation, TF-IDF -----------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, True, 2.5, "3"])
def test_top_n_must_be_a_positive_integer(bad: object):
    with pytest.raises(ValueError):
        YakeExtractor().extract("alpha beta gamma", top_n=bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        YakeExtractor(top_n=bad)  # type: ignore[arg-type]
    assert len(YakeExtractor(top_n=2).extract("alpha beta. gamma delta. epsilon zeta.", top_n=None)) == 2


def test_tfidf_rejects_string_corpus():
    with pytest.raises(TypeError):
        TfidfExtractor(corpus="hello world")
    with pytest.raises(TypeError):
        TfidfExtractor().fit("hello world")
    with pytest.raises(TypeError):
        TfidfExtractor().fit([b"bytes"])  # type: ignore[list-item]


def test_tfidf_idf_demotes_corpus_wide_terms():
    text = "Data quality matters. Data pipelines move data. Data teams clean data. Quantum annealing is new."
    corpus = [f"Data report number {i} with data tables and more data." for i in range(6)]
    unfitted = [k.phrase.lower() for k in TfidfExtractor(top_n=5).extract(text)]
    fitted = [k.phrase.lower() for k in TfidfExtractor(top_n=5, corpus=corpus).extract(text)]
    assert unfitted[0] == "data"
    assert "data" not in fitted and "quantum annealing" in fitted


def test_tfidf_ties_are_broken_by_position_not_alphabet():
    out = phrases("Zebra crossing. Apple orchard.", "tfidf", 2)
    assert out == ["Zebra crossing", "Apple orchard"]


# 6. quality heuristics ------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", sorted(EXTRACTORS))
def test_verbs_do_not_leak_into_phrases(method: str):
    for text in (TECH, POSSESSIVE, NEWS):
        for p in phrases(text, method, 15):
            words = p.lower().split()
            assert not {"consists", "runs", "slows", "praised", "expose", "warned"} & set(words), (method, p)
            assert words[-1] not in EXTRACTION_STOPWORDS


@pytest.mark.parametrize("method", sorted(EXTRACTORS))
def test_inner_connectors(method: str):
    for text in (ARTICLE, TECH, TECHNICAL):
        for p in phrases(text, method, 20):
            assert p not in ("enjoyed the warm", "address and DNS", "e-mail and email", "C++ and C#"), (method, p)
            assert " the " not in f" {p} " or " of the " in f" {p} "
    assert "Department of Defense" in phrases("The Department of Defense said so. Department of Defense staff.")
    assert "Licensor and Licensee" in phrases("Between Licensor and Licensee. Licensor and Licensee agree.", n=5)


def test_names_are_not_cut_at_the_ngram_limit():
    for method in ("yake", "tfidf", "textrank"):
        out = phrases(NEWS, method, 15)
        assert not any(p.endswith("Christine") or p == "President Christine Lagarde" for p in out), (method, out)
        assert "ECB President Christine Lagarde" in out, (method, out)
    assert "State of New York" in phrases("This shall be governed by the laws of the State of New York.", n=5)


def test_mmr_diversity_reduces_redundancy():
    text = (
        "Apple iPhone sales grew. The Apple iPhone lineup expands. Apple iPhone users upgrade. Apple iPhone "
        "prices climb. Samsung Galaxy phones compete. Google Pixel cameras impress. Battery life matters. "
        "Display quality improves. Wireless charging spreads."
    )

    def redundancy(results: list[str]) -> int:
        return sum(1 for p in results if "iPhone" in p)

    plain = phrases(text, "yake", 6, diversity=0.0)
    diverse = phrases(text, "yake", 6, diversity=0.7)
    assert redundancy(diverse) < redundancy(plain)
    with pytest.raises(ValueError):
        YakeExtractor(diversity=2)


def test_spelling_variants_are_duplicates():
    for method in EXTRACTORS:
        out = [p.lower() for p in phrases(TECHNICAL, method, 20)]
        assert not ("e-mail marketing" in out and "email marketing" in out), method


def test_textrank_does_not_rank_fragments_first():
    top = [p.lower() for p in phrases(ARTICLE, "textrank", 10)]
    assert "fall" not in top and "need" not in top
    news = phrases(NEWS, "textrank", 5)
    assert news[0] != "Bank" and "Bank" not in news


# 7. optional POS filter --------------------------------------------------------------------------------


def _fake_tagger(tags: dict[str, str]):
    def nlp(text: str) -> list[SimpleNamespace]:
        import re

        return [
            SimpleNamespace(idx=m.start(), text=m.group(), pos_=tags.get(m.group().lower(), "NOUN"))
            for m in re.finditer(r"\w+|[^\w\s]", text)
        ]

    return nlp


def test_pos_filter_keeps_noun_phrases_only():
    text = "Fast cars race quickly. Fast cars race often. Bright students study hard."
    tags = {
        "fast": "ADJ",
        "bright": "ADJ",
        "race": "VERB",
        "study": "VERB",
        "quickly": "ADV",
        "often": "ADV",
        "hard": "ADV",
    }
    for method in EXTRACTORS:
        out = phrases(text, method, 10, pos_filter=_fake_tagger(tags))
        assert out and all(p.split()[-1].lower() in ("cars", "students") for p in out), (method, out)
        without = phrases(text, method, 10)
        assert any(p.split()[-1].lower() not in ("cars", "students") for p in without), method


def test_pos_filter_requires_a_tagger():
    spacy = pytest.importorskip("spacy")
    with pytest.raises(ValueError, match="tagger"):
        YakeExtractor(pos_filter=spacy.blank("en")).extract("Some text here.")
    with pytest.raises(TypeError):
        YakeExtractor(pos_filter="en_core_web_sm")  # type: ignore[arg-type]


# 8. languages -------------------------------------------------------------------------------------------


def test_language_stopwords():
    greek = "Ελληνικά κείμενα για μηχανική μάθηση. Μηχανική μάθηση παντού."
    for method in EXTRACTORS:
        out = phrases(greek, method, 10, language="el")
        assert out and not any("για" in p.split() for p in out), (method, out)
    assert "για" in GREEK_STOPWORDS
    german = phrases(
        "Das Gesundheitsministerium kündigte neue Maßnahmen an. Das Gesundheitsministerium prüft.", language="de"
    )
    assert not any(p.split()[0].lower() == "das" for p in german)
    for lang in ("de", "fr", "es", "it", "el", "en", "EN", "fr-FR", "de_DE"):
        assert get_stopwords(lang)
    with pytest.raises(ValueError):
        get_extractor("yake", language="xx")
    # explicit stopwords override the language list
    out = phrases("alpha beta gamma. alpha beta gamma.", "yake", 5, stopwords=["beta"], language="de")
    assert "beta" not in [p for o in out for p in o.split()]
    with pytest.raises(TypeError):
        YakeExtractor(stopwords="the")  # type: ignore[arg-type]


# 9-12. spaCy -------------------------------------------------------------------------------------------


@pytest.fixture
def ks():
    pytest.importorskip("spacy")
    from keylex.integrations import spacy as module

    return module


def test_spacy_component_round_trips_to_disk_and_bytes(ks, tmp_path):
    import spacy

    nlp = spacy.blank("en")
    kl = KeywordList([Keyword("machine learning", category="ML"), Keyword("NASA")])
    ks.add_to_pipeline(nlp, kl, set_ents=True, label="keyword", case_sensitive=False)
    assert [e.text for e in nlp("I like machine learning at NASA").ents] == ["machine learning", "NASA"]
    nlp.to_disk(tmp_path / "model")
    loaded = spacy.load(tmp_path / "model")
    doc = loaded("I like machine learning at NASA")
    assert [(e.text, e.label_) for e in doc.ents] == [("machine learning", "machine learning"), ("NASA", "NASA")]
    assert loaded.get_pipe("keylex").keyword_list["machine learning"].category == "ML"

    other = spacy.blank("en")
    other.add_pipe("keylex")
    other.from_bytes(nlp.to_bytes())
    assert [e.text for e in other("machine learning and NASA").ents] == ["machine learning", "NASA"]


def test_spacy_identical_spans_keep_every_keyword(ks):
    import spacy

    kl = KeywordList(
        [
            Keyword("Apple Inc", synonyms=["apple"], category="ORG"),
            Keyword("apple fruit", synonyms=["apple"], category="FOOD"),
        ]
    )
    nlp = spacy.blank("en")
    nlp.add_pipe("keylex", config={"keywords": kl.to_list()})
    doc = nlp("I ate an apple.")
    assert sorted(m.keyword for m in doc._.keylex_matches) == ["Apple Inc", "apple fruit"]
    spans = list(doc.spans["keylex"])
    assert [s.label_ for s in spans] == ["ORG", "FOOD"]
    for s in spans:
        assert sorted(m.keyword for m in s._.keylex_matches) == ["Apple Inc", "apple fruit"]
    assert {s.label_: s._.keylex_match.keyword for s in spans} == {"ORG": "Apple Inc", "FOOD": "apple fruit"}


def test_entity_ruler_patterns_respect_case(ks):
    import spacy

    kl = KeywordList(
        [
            Keyword("NASA"),
            Keyword("Python", case_sensitive=True),
            Keyword("deep learning", patterns=[r"deep\s+nets?"], acronyms=["DL"]),
        ]
    )
    with pytest.warns(UserWarning, match="regex"):
        patterns = ks.to_entity_ruler_patterns(kl)
    nlp = spacy.blank("en")
    nlp.add_pipe("entity_ruler").add_patterns(patterns)
    text = "nasa and python and dl; NASA Python DL and Deep Learning"
    assert [e.text for e in nlp(text).ents] == [m.text for m in KeywordMatcher(kl).find(text) if m.kind != "pattern"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ks.to_entity_ruler_patterns(["plain"])


def test_set_ents_keeps_existing_entities_by_default(ks):
    import spacy

    def build(**config: object):
        nlp = spacy.blank("en")
        nlp.add_pipe("entity_ruler").add_patterns([{"label": "GPE", "pattern": "New York"}])
        nlp.add_pipe("keylex", config={"keywords": ["New York City", "Boston"], "set_ents": True, **config})
        return nlp("I love New York City and Boston.")

    assert [(e.text, e.label_) for e in build().ents] == [("New York", "GPE"), ("Boston", "KEYWORD")]
    assert [(e.text, e.label_) for e in build(overwrite_ents=True).ents] == [
        ("New York City", "KEYWORD"),
        ("Boston", "KEYWORD"),
    ]


def test_alignment_mode_strict_skips_intra_word_matches(ks):
    import spacy

    kl = KeywordList([Keyword("learn", variants=[Term("learnab", "variant", whole_word=False)])])
    nlp = spacy.blank("en")
    ks.add_to_pipeline(nlp, kl)
    doc = nlp("this is unlearnable")
    assert len(doc._.keylex_matches) == 1 and len(doc.spans["keylex"]) == 0
    nlp2 = spacy.blank("en")
    ks.add_to_pipeline(nlp2, kl, alignment_mode="expand")
    doc2 = nlp2("this is unlearnable")
    assert [s.text for s in doc2.spans["keylex"]] == ["unlearnable"]
    with pytest.raises(ValueError):
        ks.match_spans(nlp("x"), KeywordMatcher(kl), alignment_mode="sloppy")


# 13. highlight escaping ----------------------------------------------------------------------------------


@pytest.fixture
def xss() -> tuple[str, list[Match]]:
    text = "<script>alert(1)</script> machine learning & more"
    return text, KeywordMatcher(["machine learning"]).find(text)


def test_highlight_escapes_html_templates(xss):
    text, ms = xss
    out = highlight(text, ms, "<mark>{text}</mark>")
    assert "<script>" not in out and "&lt;script&gt;" in out and "<mark>machine learning</mark>" in out
    assert "&amp; more" in out
    assert "<script>" not in highlight(text, ms, "html")
    assert highlight(text, ms, "brackets").startswith("<script>")  # plain-text styles are left alone
    assert highlight(text, ms, "<mark>{text}</mark>", escape=False).startswith("<script>")
    assert "&lt;script&gt;" in highlight(text, ms, "[{text}]", escape=True)


def test_highlight_callable_escaping(xss):
    text, _ = xss
    nasty = "<b>x</b>"
    kl = KeywordList([Keyword("y", synonyms=[nasty])])
    t2 = f"{text} {nasty}"
    ms2 = KeywordMatcher(kl).find(t2)

    def style(m: Match) -> str:
        return f"<i>{m.text}</i>"

    assert "<script>" in highlight(t2, ms2, style)  # callables are not escaped by default
    out = highlight(t2, ms2, style, escape=True)
    assert "<script>" not in out and "<i>&lt;b&gt;x&lt;/b&gt;</i>" in out
