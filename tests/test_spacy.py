from __future__ import annotations

import pytest

spacy = pytest.importorskip("spacy")

from keylex import Keyword, KeywordList  # noqa: E402
from keylex.integrations import spacy as ks  # noqa: E402


@pytest.fixture
def kl() -> KeywordList:
    return KeywordList([Keyword("United States", acronyms="US", category="COUNTRY"), "machine learning"])


def test_factory_with_config(kl: KeywordList):
    nlp = spacy.blank("en")
    nlp.add_pipe("keylex", config={"keywords": kl.to_list(), "set_ents": True})
    doc = nlp("Machine learning in the U.S. grows.")
    assert [(s.text, s.label_) for s in doc.spans["keylex"]] == [("Machine learning", "KEYWORD"), ("U.S.", "COUNTRY")]
    assert [e.text for e in doc.ents] == ["Machine learning", "U.S."]
    assert doc.spans["keylex"][1]._.keylex_match.keyword == "United States"
    assert len(doc._.keylex_matches) == 2
    assert nlp.config["components"]["keylex"]["keywords"][0]["keyword"] == "United States"


def test_add_to_pipeline_is_live(kl: KeywordList):
    nlp = spacy.blank("en")
    ks.add_to_pipeline(nlp, kl, label="keyword", match_type="fuzzy")
    doc = nlp("US and machne learning")
    assert [(s.text, s.label_) for s in doc.spans["keylex"]] == [
        ("US", "United States"),
        ("machne learning", "machine learning"),
    ]
    kl.add("and")
    doc = nlp("US and machine learning")
    assert len(doc.spans["keylex"]) == 3
    docs = list(nlp.pipe(["US", "nothing"]))
    assert [len(d.spans["keylex"]) for d in docs] == [1, 0]


def test_factory_loads_file(tmp_path, kl: KeywordList):
    path = tmp_path / "kw.json"
    kl.save(path)
    nlp = spacy.blank("en")
    nlp.add_pipe("keylex", config={"keywords": str(path), "label": "fixed", "default_label": "KW"})
    doc = nlp("the US")
    assert [s.label_ for s in doc.spans["keylex"]] == ["KW"]


def test_entity_ruler_patterns(kl: KeywordList):
    patterns = ks.to_entity_ruler_patterns(kl)
    assert {"label": "COUNTRY", "pattern": [{"ORTH": "US"}], "id": "United States"} in patterns
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler", config={"phrase_matcher_attr": "LOWER"})
    ruler.add_patterns(patterns)
    doc = nlp("machine learning in the us")
    assert next((e.text, e.ent_id_) for e in doc.ents) == ("machine learning", "machine learning")
    assert ks.to_entity_ruler_patterns(kl, label="keyword")[0]["label"] == "United States"
    assert ks.to_entity_ruler_patterns(kl, label=lambda k: k.name.upper())[0]["label"] == "UNITED STATES"


def test_bad_label_mode(kl: KeywordList):
    nlp = spacy.blank("en")
    with pytest.raises(ValueError):
        ks.match_spans(nlp("x"), ks.KeywordMatcher(kl), label="colour")
