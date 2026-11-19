from __future__ import annotations

import pytest

from keylex import Keyword, KeywordList, KeywordMatcher
from keylex.rules import (
    Document,
    RuleEngine,
    RuleSyntaxError,
    all_of,
    any_of,
    at_least,
    before,
    category,
    kw,
    near,
    none_of,
    not_,
    parse_rule,
    same_paragraph,
    same_sentence,
    tag,
)

TEXT = (
    "The central bank raised the interest rate. Inflation remains high and inflation expectations rose.\n\n"
    "We had a picnic on the river bank."
)


@pytest.fixture
def doc() -> Document:
    kl = KeywordList(
        [
            Keyword("central bank", category="finance", tags=["institution"]),
            Keyword("interest rate", category="finance"),
            Keyword("inflation", category="finance"),
            Keyword("bank", category="finance"),
            Keyword("river", category="nature"),
            Keyword("picnic"),
        ]
    )
    return Document(TEXT, KeywordMatcher(kl, overlaps="all").find(TEXT))


def test_keyword_and_boolean_ops(doc: Document):
    assert kw("inflation").evaluate(doc)
    assert len(kw("inflation").evaluate(doc).matches) == 2
    assert not kw("unicorn").evaluate(doc)
    assert (kw("bank") & kw("river")).evaluate(doc)
    assert (kw("unicorn") | "river").evaluate(doc)
    assert not (~kw("river")).evaluate(doc)
    assert ("river" & kw("picnic")).evaluate(doc)
    assert kw("unicorn", "picnic").evaluate(doc)
    assert not_("unicorn").evaluate(doc) and none_of("unicorn", "dragon").evaluate(doc)
    assert all_of("bank", "river").evaluate(doc) and not all_of("bank", "unicorn").evaluate(doc)


def test_counts_and_any_of(doc: Document):
    assert at_least(2, "inflation").evaluate(doc)
    assert not kw("inflation").at_least(3).evaluate(doc)
    assert any_of("bank", "river", "unicorn", min_count=2).evaluate(doc)
    assert not any_of("bank", "unicorn", "dragon", min_count=2).evaluate(doc)
    with pytest.raises(ValueError):
        at_least(0, "x")


def test_proximity(doc: Document):
    assert near("river", "picnic", within=4).evaluate(doc)
    assert not near("river", "picnic", within=1).evaluate(doc)
    assert before("picnic", "river", within=5).evaluate(doc)
    assert not before("river", "picnic", within=5).evaluate(doc)
    assert near("bank", "river", within=0).evaluate(doc)  # "river bank": adjacent


def test_scopes(doc: Document):
    assert same_sentence("central bank", "interest rate").evaluate(doc)
    assert not same_sentence("central bank", "river").evaluate(doc)
    assert same_paragraph("central bank", "inflation").evaluate(doc)
    assert not same_paragraph("inflation", "picnic").evaluate(doc)
    with pytest.raises(ValueError):
        same_sentence("one")


def test_category_and_tag(doc: Document):
    assert category("finance").at_least(5).evaluate(doc)
    assert category("nature").evaluate(doc) and not category("sports").evaluate(doc)
    assert tag("institution").evaluate(doc) and not tag("nope").evaluate(doc)


def test_disambiguation_example(doc: Document):
    financial_bank = kw("bank") & ~near("bank", "river", within=2)
    result = financial_bank.evaluate(doc)
    assert not result  # the river bank is close to "river"
    finance_story = (kw("central bank") | category("finance").at_least(3)) & ~kw("picnic")
    assert not finance_story.evaluate(doc)


def test_explain_and_keyword_names(doc: Document):
    rule = (kw("inflation") & near("bank", "river", 1)) | ~kw("picnic")
    explanation = rule.evaluate(doc).explain()
    assert "[+]" in explanation and "inflation" in explanation and "pair(s)" in explanation
    assert rule.keyword_names() == {"inflation", "bank", "river", "picnic"}


DSL_CASES = [
    "bank",
    '"central bank" AND NOT river',
    "central bank & !river",
    "interest rate OR inflation",
    "NEAR(bank, river, 3)",
    "BEFORE(picnic, river)",
    "ATLEAST(2, inflation)",
    "ANY(2, bank, river, unicorn)",
    "ALL(bank, river)",
    "NONE(unicorn, dragon)",
    "SENTENCE(\"central bank\", 'interest rate')",
    "PARAGRAPH(bank, inflation)",
    "CATEGORY(finance) AND TAG(institution)",
    'KW("AND")',
    "(bank OR river) AND NOT (unicorn OR dragon)",
    "near(bank, river, 3) and not unicorn",
]


@pytest.mark.parametrize("source", DSL_CASES)
def test_dsl_round_trip(source: str, doc: Document):
    rule = parse_rule(source)
    again = parse_rule(rule.to_dsl())
    assert again == rule
    assert again.evaluate(doc).matched == rule.evaluate(doc).matched


def test_dsl_semantics(doc: Document):
    assert parse_rule("ATLEAST(2, inflation) AND NEAR(river, picnic, 4)").evaluate(doc)
    assert not parse_rule("bank AND NOT river").evaluate(doc)
    assert parse_rule("interest rate").keyword_names() == {"interest rate"}
    assert parse_rule("a OR b AND c").to_dsl() == "(a OR (b AND c))"  # AND binds tighter


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "AND bank",
        "bank AND",
        "(bank",
        "NEAR(bank)",
        "ATLEAST(bank, 2)",
        "ANY()",
        '"unterminated',
        "bank )",
        "ALL(2, a)",
    ],
)
def test_dsl_errors(bad: str):
    with pytest.raises(RuleSyntaxError):
        parse_rule(bad)


def test_engine_auto_adds_keywords_and_scores():
    engine = RuleEngine(
        {"Federal Reserve": {"acronyms": ["Fed"]}},
        rules={
            "monetary": {"rule": '"Federal Reserve" AND "interest rate"', "weight": 3},
            "inflation": "ATLEAST(2, inflation)",
            "nature": "river",
        },
        stemmer="light",
    )
    result = engine.evaluate("The Fed raised interest rates as inflation and more inflation bit.")
    assert result.fired == ["monetary", "inflation"]
    assert result.top == "monetary" and result.score == 4.0
    assert "monetary" in result and "nature" not in result
    assert result["monetary"].keywords == ["Federal Reserve", "interest rate"]
    assert "[monetary] weight=3" in result.explain()
    assert result.to_dict()["fired"] == ["monetary", "inflation"]
    assert engine.classify("nothing here") == []
    assert next(iter(engine.evaluate_many(["river", "x"]))).fired == ["nature"]


def test_engine_sees_keyword_changes_and_rule_changes():
    kl = KeywordList([Keyword("bank")])
    engine = RuleEngine(kl, {"b": "bank"})
    assert engine.classify("lender") == []
    kl["bank"].add_synonyms("lender")
    assert engine.classify("lender") == ["b"]
    engine.add_rule("new", "unicorn", weight=2)
    assert engine.classify("a unicorn and a bank") == ["new", "b"]
    engine.remove_rule("new")
    assert engine.classify("unicorn") == []


def test_engine_persistence(tmp_path):
    engine = RuleEngine(
        [Keyword("bank", synonyms="lender")],
        {"finance": {"rule": "bank AND NOT river", "weight": 2, "description": "money"}},
        match_type="fuzzy",
    )
    path = tmp_path / "rules.json"
    engine.save(path)
    loaded = RuleEngine.load(path)
    assert loaded.rules["finance"].weight == 2 and loaded.rules["finance"].description == "money"
    assert loaded.matcher.match_type == "fuzzy"
    assert loaded.classify("a lendr") == ["finance"]
    yaml = pytest.importorskip("yaml")
    ypath = tmp_path / "rules.yaml"
    ypath.write_text(yaml.safe_dump(engine.to_dict()), encoding="utf-8")
    assert RuleEngine.load(ypath).classify("lender") == ["finance"]


def test_engine_validation():
    with pytest.raises(ValueError):
        RuleEngine().add_rule("", "x")
    with pytest.raises(TypeError):
        RuleEngine().add_rule("x", 42)  # type: ignore[arg-type]
    with pytest.raises(RuleSyntaxError):
        RuleEngine(rules={"bad": "AND"})


def test_rule_test_helper():
    assert kw("bank").test("the bank", KeywordMatcher(["bank"]))
