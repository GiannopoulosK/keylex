from __future__ import annotations

import pytest

from keylex import KeywordMatcher, extract_keywords, get_extractor
from keylex.extractors import EXTRACTORS, RakeExtractor, TfidfExtractor, YakeExtractor

TEXT = """
Central banks raised interest rates again this quarter. The Federal Reserve said inflation remains
high, and the European Central Bank signalled further rate hikes. Analysts expect interest rates to
stay elevated while inflation cools. Mortgage rates climbed to a 20-year high. The Federal Reserve
chair said more data is needed before interest rates fall.
"""


@pytest.mark.parametrize("method", sorted(EXTRACTORS))
def test_every_method_returns_sensible_keywords(method: str):
    results = extract_keywords(TEXT, top_n=8, method=method)
    assert 1 <= len(results) <= 8
    phrases = [r.phrase.lower() for r in results]
    assert any("interest rate" in p or "federal reserve" in p or "inflation" in p for p in phrases)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True) and scores[0] == pytest.approx(1.0)
    for p in phrases:
        words = p.split()
        assert words[0] not in ("the", "and", "of") and words[-1] not in ("the", "and", "of")
        assert not words[-1].endswith("ed") or words[-1].endswith("eed")


def test_yake_prefers_proper_nouns_and_frequent_terms():
    phrases = [r.phrase for r in YakeExtractor(top_n=5).extract(TEXT)]
    assert "Federal Reserve" in phrases


def test_deduplication():
    text = "Neural networks. A neural network. Neural networks again, and the neural network."
    phrases = [r.phrase.lower() for r in extract_keywords(text, 5)]
    assert sum(p.startswith("neural network") for p in phrases) == 1
    raw = [r.phrase.lower() for r in extract_keywords(text, 5, deduplicate=False)]
    assert len(raw) >= len(phrases)


def test_rake_keeps_long_phrases():
    phrases = [r.phrase.lower() for r in RakeExtractor(top_n=5).extract(TEXT)]
    assert any(len(p.split()) >= 2 for p in phrases)


def test_tfidf_with_corpus():
    corpus = [
        "Interest rates and inflation dominate the news.",
        "Interest rates are discussed in every report.",
        "The mortgage market reacted.",
    ]
    tfidf = TfidfExtractor(top_n=5, corpus=corpus)
    assert tfidf.is_fitted
    phrases = [r.phrase.lower() for r in tfidf.extract(TEXT)]
    assert "federal reserve" in phrases
    assert not TfidfExtractor().is_fitted


def test_custom_stopwords_and_ngram():
    out = extract_keywords("alpha beta gamma. alpha beta gamma.", 5, stopwords=["beta"], max_ngram=1)
    assert all(len(r.phrase.split()) == 1 for r in out)
    assert "beta" not in [r.phrase for r in out]


def test_to_keyword_list_feeds_matcher():
    extractor = get_extractor("yake", top_n=5)
    kl = extractor.to_keyword_list(TEXT)
    assert len(kl) == 5 and all("score" in kw.metadata for kw in kl)
    assert KeywordMatcher(kl).find(TEXT)


def test_edge_cases():
    assert extract_keywords("", 5) == []
    assert extract_keywords("the and of", 5) == []
    assert extract_keywords("12 34 56", 5) == []
    with pytest.raises(ValueError):
        get_extractor("magic")
    with pytest.raises(ValueError):
        YakeExtractor(top_n=0)
    with pytest.raises(TypeError):
        YakeExtractor().extract(None)  # type: ignore[arg-type]
    assert list(YakeExtractor(top_n=2).extract_many(["alpha beta", "gamma"]))[1][0].phrase == "gamma"
    assert extract_keywords("Machine learning.", 3)[0].to_keyword().name.lower() == "machine learning"
