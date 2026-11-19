"""Regression tests for the extraction / spaCy review fixes (2026-11)."""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

from keylex import Keyword, KeywordList, extract_keywords
from keylex.extractors import EXTRACTORS, TfidfExtractor, get_extractor
from keylex.stopwords import EXTRACTION_STOPWORDS

# --- extraction -------------------------------------------------------------------------------------

DNN = "We train a deep neural network. The deep neural network is fast. Deep neural networks are popular."


@pytest.mark.parametrize("method", sorted(EXTRACTORS))
def test_spelling_variants_are_pooled_into_the_kept_phrase(method: str):
    out = extract_keywords(DNN, method=method)
    dnn = [k for k in out if "neural" in k.phrase.lower()]
    assert [k.phrase.lower() for k in dnn] == ["deep neural network"], (method, out)
    assert dnn[0].count == 3
    assert len(dnn[0].positions) == 3
    for p in dnn[0].positions:
        assert DNN[p : p + len("deep neural network")].lower() == "deep neural network"


def test_without_dedup_variants_stay_separate():
    out = extract_keywords(DNN, method="rake", deduplicate=False, diversity=0)
    assert {k.phrase.lower() for k in out} >= {"deep neural network", "deep neural networks"}


AMBIGUOUS = (
    "We minimise the mean squared error of the estimator. The mean squared error drops by half. "
    "Patients with an enlarged left ventricle were studied; the left ventricle wall thickens. "
    "Results in number theory and free fall experiments. Our talk show guests discussed the light rail project."
)


@pytest.mark.parametrize("method", sorted(EXTRACTORS))
def test_ambiguous_verb_nouns_can_start_or_end_phrases(method: str):
    out = " | ".join(k.phrase.lower() for k in extract_keywords(AMBIGUOUS, top_n=15, method=method))
    for phrase in ("left ventricle", "number theory", "free fall", "talk show guests"):
        assert phrase in out, (method, phrase, out)
    if method != "rake":  # RAKE keeps the whole run "mean squared error drops"
        assert "mean squared error" in out, (method, out)


@pytest.mark.parametrize("method", sorted(EXTRACTORS))
def test_ambiguous_words_read_as_verbs_after_subjects(method: str):
    text = (
        "Prices fall when investors hope for cuts. The patient left early. Analysts need better data. "
        "Gold prices fall again. Investors hope again."
    )
    for k in extract_keywords(text, top_n=15, method=method):
        words = k.phrase.lower().split()
        assert not {"fall", "hope", "need", "left"} & set(words), (method, k.phrase)
        assert words[-1] not in EXTRACTION_STOPWORDS


def test_weak_end_words_never_end_a_phrase():
    text = "A number of patients were found. The number grows. Lost luggage was found."
    for method in sorted(EXTRACTORS):
        for k in extract_keywords(text, top_n=10, method=method):
            assert k.phrase.lower().split()[-1] not in ("number", "found", "lost"), (method, k.phrase)


@dataclass
class _Tok:
    idx: int
    text: str
    pos_: str


def _fake_tagger(tags: dict[str, str]):
    def tag(text: str) -> list[_Tok]:
        return [_Tok(m.start(), m.group(), tags.get(m.group().lower(), "NOUN")) for m in re.finditer(r"\w+", text)]

    return tag


def test_pos_filter_replaces_verb_heuristics():
    text = "That was a great find for the team. A great find indeed. Distributed systems are hard."
    tags = {"that": "PRON", "was": "AUX", "a": "DET", "great": "ADJ", "for": "ADP", "the": "DET", "are": "AUX"}
    tags.update({"indeed": "ADV", "hard": "ADJ", "distributed": "ADJ", "is": "AUX"})
    tagger = _fake_tagger(tags)
    plain = [k.phrase.lower() for k in extract_keywords(text, top_n=10)]
    tagged = [k.phrase.lower() for k in extract_keywords(text, top_n=10, pos_filter=tagger)]
    assert "great find" not in plain  # "find" is a verb stopword for the heuristics ...
    assert "great find" in tagged  # ... but the tagger says NOUN
    assert "distributed systems" in tagged
    # verbs are still rejected when the tagger says so
    tags["find"] = "VERB"
    assert "great find" not in [k.phrase.lower() for k in extract_keywords(text, top_n=10, pos_filter=tagger)]


TWEET = (
    "Visit https://example.com/path?q=1 or mail info@example.com now. Follow @jfk_airport and #travel "
    "for news on the new terminal; short link t.co/abc123. The new terminal opens www.jfk.example.org soon."
)


@pytest.mark.parametrize("method", sorted(EXTRACTORS))
def test_urls_emails_handles_are_not_phrases(method: str):
    out = [k.phrase for k in extract_keywords(TWEET, top_n=20, method=method)]
    junk = re.compile(r"https?|example|info|t\.co|abc123|jfk|travel|www|\.org|[@#]")
    for p in out:
        assert not junk.search(p.lower()), (method, p)
    assert any("new terminal" in p.lower() for p in out), (method, out)


def test_tfidf_fit_empty_corpus_raises():
    with pytest.raises(ValueError, match="empty"):
        TfidfExtractor().fit([])
    with pytest.raises(ValueError, match="empty"):
        TfidfExtractor(corpus=iter(()))


def test_top_n_none_means_default():
    assert get_extractor("yake", top_n=None).top_n == 10
    assert len(extract_keywords(" ".join(f"word{i} alpha{i}." for i in range(30)), top_n=None)) == 10
    with pytest.raises(ValueError, match="positive integer"):
        get_extractor("yake", top_n=0)


# --- spaCy --------------------------------------------------------------------------------------------

spacy = pytest.importorskip("spacy")
from spacy.tokens import Doc, DocBin  # noqa: E402

from keylex.integrations import spacy as ks  # noqa: E402

KL = KeywordList([Keyword("United States", acronyms=["US"], category="GPE"), "machine learning"])
TEXT = "Machine learning in the U.S. and US"


def _nlp():
    nlp = spacy.blank("en")
    ks.add_to_pipeline(nlp, KL, set_ents=True)
    return nlp


def _summary(doc: Doc):
    return (
        [(m.keyword, m.start, m.end, m.text, m.kind, m.category) for m in doc._.keylex_matches],
        [(s.text, s.label_, s._.keylex_match.keyword) for s in doc.spans["keylex"]],
        [[m.keyword for m in s._.keylex_matches] for s in doc.spans["keylex"]],
    )


def test_processed_doc_to_bytes_round_trip():
    nlp = _nlp()
    doc = nlp(TEXT)
    assert doc._.keylex_matches[0].keyword_obj is KL["machine learning"]  # live objects in-process
    data = doc.to_bytes()
    restored = Doc(nlp.vocab).from_bytes(data)
    assert _summary(restored) == _summary(doc)
    assert restored._.keylex_matches == doc._.keylex_matches
    # keyword objects are found again while the component is alive
    assert restored._.keylex_matches[1].keyword_obj is KL["United States"]


def test_processed_doc_docbin_round_trip():
    nlp = _nlp()
    doc = nlp(TEXT)
    bin_ = DocBin(store_user_data=True)
    bin_.add(doc)
    restored = next(iter(DocBin().from_bytes(bin_.to_bytes()).get_docs(nlp.vocab)))
    assert _summary(restored) == _summary(doc)
    assert restored.spans["keylex"][1]._.keylex_match.category == "GPE"


def test_span_match_setter_and_doc_setter_still_work():
    nlp = _nlp()
    doc = nlp(TEXT)
    span = doc.spans["keylex"][0]
    other = doc._.keylex_matches[1]
    span._.keylex_match = other
    assert span._.keylex_match == other
    assert len(span._.keylex_matches) == 2
    doc._.keylex_matches = None
    assert doc._.keylex_matches is None
    restored = Doc(nlp.vocab).from_bytes(doc.to_bytes())
    assert restored.spans["keylex"][0]._.keylex_match == other
    assert spacy.blank("en")("x")._.keylex_matches is None


def test_n_process_pipe(tmp_path: Path):
    script = tmp_path / "mp.py"
    script.write_text(
        textwrap.dedent(
            """
            import spacy
            from keylex import Keyword, KeywordList
            from keylex.integrations.spacy import add_to_pipeline

            if __name__ == "__main__":
                nlp = spacy.blank("en")
                kl = KeywordList([Keyword("United States", acronyms=["US"]), "machine learning"])
                add_to_pipeline(nlp, kl)
                docs = list(nlp.pipe(["US machine learning"] * 4, n_process=2, batch_size=2))
                out = [[m.keyword for m in d._.keylex_matches] for d in docs]
                assert out == [["United States", "machine learning"]] * 4, out
                assert docs[0]._.keylex_matches[0].keyword_obj is kl["United States"]
                assert [s._.keylex_match.keyword for s in docs[0].spans["keylex"]] == out[0]
                print("OK")
            """
        ),
        encoding="utf-8",
    )
    res = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=300)
    assert res.returncode == 0 and "OK" in res.stdout, res.stderr[-2000:]


def test_spacy_load_without_original_keyword_file(tmp_path: Path):
    path = tmp_path / "kw.json"
    KeywordList([Keyword("machine learning"), Keyword("NASA")]).save(path)
    nlp = spacy.blank("en")
    nlp.add_pipe("keylex", config={"keywords": str(path)})
    nlp.to_disk(tmp_path / "model")
    path.unlink()
    with pytest.warns(UserWarning, match="not found"):
        loaded = spacy.load(tmp_path / "model")
    doc = loaded("machine learning at NASA")
    assert [s.text for s in doc.spans["keylex"]] == ["machine learning", "NASA"]


def test_two_components_accumulate_matches():
    nlp = spacy.blank("en")
    ks.add_to_pipeline(nlp, KL)
    ks.add_to_pipeline(nlp, ["machine"], name="k2", spans_key="k2")
    doc = nlp("machine learning in the US")
    assert [m.keyword for m in doc._.keylex_matches] == ["machine learning", "United States", "machine"]
    assert [s.text for s in doc.spans["keylex"]] == ["machine learning", "US"]
    assert [s.text for s in doc.spans["k2"]] == ["machine"]


def test_entity_ruler_patterns_fixed_label_and_dotted_acronyms():
    fixed = ks.to_entity_ruler_patterns(KL, label="fixed", default_label="KW")
    assert {p["label"] for p in fixed} == {"KW"}
    assert {p["label"] for p in ks.to_entity_ruler_patterns(KL, label="ORG")} == {"ORG"}
    patterns = ks.to_entity_ruler_patterns(KL)
    assert {"label": "GPE", "pattern": [{"ORTH": "U.S."}], "id": "United States"} in patterns
    nlp = spacy.blank("en")
    nlp.add_pipe("entity_ruler").add_patterns(patterns)
    text = "Machine learning in the U.S. and US and us"
    assert [e.text for e in nlp(text).ents] == [m.text for m in ks.KeywordMatcher(KL).find(text)]
