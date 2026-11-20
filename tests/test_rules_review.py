"""Regression tests for the rules / CLI review findings (items 1-16)."""

from __future__ import annotations

import io
import json
import math
import random
import sys
import time
import warnings
from collections.abc import Callable
from typing import Any

import pytest

from keylex import Keyword, KeywordList, KeywordMatcher
from keylex.cli import main
from keylex.match import Match
from keylex.rules import (
    AllOf,
    AnyOf,
    AtLeast,
    CategoryRule,
    Document,
    KeywordRule,
    Near,
    Not,
    Rule,
    RuleEngine,
    RuleResult,
    RuleSyntaxError,
    SameParagraph,
    SameSentence,
    TagRule,
    at_least,
    before,
    category,
    kw,
    near,
    parse_rule,
    same_sentence,
    tag,
)


def make_doc(text: str, *keywords: Keyword | str) -> Document:
    return Document(text, KeywordMatcher(list(keywords), overlaps="all").find(text))


# ---------------------------------------------------------------------- 1. NEAR performance
def _synthetic(n_words: int, vocab: int = 20) -> Document:
    words = [f"k{i % vocab}" for i in range(n_words)]
    text = " ".join(words)
    matches, pos = [], 0
    for w in words:
        matches.append(Match(w, pos, pos + len(w), w, w, category="c"))
        pos += len(w) + 1
    return Document(text, matches)


def test_near_is_fast_on_large_documents(time_budget: Callable[[float, float], None]):
    doc = _synthetic(20_000)
    rule = parse_rule("NEAR(CATEGORY(c), CATEGORY(c), 5)")
    t0 = time.perf_counter()
    result = doc.evaluate(rule)
    time_budget(t0, 2.0)
    assert result.matched and result.count == 10_000  # disjoint neighbour pairs
    t0 = time.perf_counter()
    assert doc.evaluate(parse_rule("BEFORE(k1, k2, 3)")).count == 1000
    time_budget(t0, 2.0)


def _brute_force(doc: Document, rule: Near, a: list[Match], b: list[Match]) -> set[Match]:
    """The original O(n*m) definition of which matches take part in a qualifying pair."""
    hits: set[Match] = set()
    for ma in a:
        for mb in b:
            if rule._gap_ok(ma, doc.word_span(ma), mb, doc.word_span(mb)):
                hits |= {ma, mb}
    return hits


@pytest.mark.parametrize("seed", range(25))
def test_near_evidence_matches_brute_force(seed: int):
    rng = random.Random(seed)
    words = [rng.choice(["a", "b", "c", "x", "y"]) for _ in range(rng.randint(1, 60))]
    text = " ".join(words)
    kws = [Keyword("a"), Keyword("b"), Keyword("a b"), Keyword("b c x")]
    doc = make_doc(text, *kws)
    for within in (0, 1, 3):
        for ordered in (False, True):
            for sa, sb in (("a", "b"), ("a b", "b"), ("b", "b c x"), ("a", "a")):
                rule = Near(sa, sb, within, ordered)
                res = doc.evaluate(rule)
                a, b = doc.keyword_matches(sa), doc.keyword_matches(sb)
                expected = _brute_force(doc, rule, a, b)
                assert set(res.matches) == expected
                assert res.matched == bool(expected)
                assert res.count <= len(expected)


# ---------------------------------------------------------------------- 2. ATLEAST counts occurrences
def test_atleast_counts_occurrences_not_spans():
    doc = make_doc("I like apple pie.", "apple", "pie")
    assert parse_rule("NEAR(apple, pie, 3)").evaluate(doc)
    assert not parse_rule("ATLEAST(2, NEAR(apple, pie, 3))").evaluate(doc)
    doc = make_doc("apple and pie", "apple", "pie")
    assert not parse_rule("ATLEAST(2, apple AND pie)").evaluate(doc)
    assert parse_rule("ATLEAST(2, apple OR pie)").evaluate(doc)  # OR sums
    doc = make_doc("apple pie. Later apple pie again. Apple.", "apple", "pie")
    assert parse_rule("ATLEAST(2, NEAR(apple, pie, 1))").evaluate(doc)
    assert not parse_rule("ATLEAST(3, NEAR(apple, pie, 1))").evaluate(doc)
    assert parse_rule("ATLEAST(2, apple AND pie)").evaluate(doc)  # min(3, 2)
    assert not parse_rule("ATLEAST(3, apple AND pie)").evaluate(doc)
    assert parse_rule("ATLEAST(2, SENTENCE(apple, pie))").evaluate(doc)
    assert not parse_rule("ATLEAST(3, SENTENCE(apple, pie))").evaluate(doc)
    # ANY(n): the n-th largest operand count
    doc = make_doc("a a a b b c", "a", "b", "c")
    assert doc.evaluate(parse_rule("ANY(2, a, b, c)")).count == 2
    assert doc.evaluate(parse_rule("ANY(3, a, b, c)")).count == 1


def test_occurrences_fallback_for_custom_rules():
    m = Match("x", 0, 1, "x", "x")
    assert RuleResult(KeywordRule("x"), True, (m, m)).occurrences == 1
    assert RuleResult(KeywordRule("x"), False).occurrences == 0


# ---------------------------------------------------------------------- 3. NOT in scoped / proximity rules
def test_not_inside_sentence_and_paragraph():
    doc = make_doc("The bank by the river. The bank raised rates.\n\nA river.", "bank", "river")
    assert parse_rule("SENTENCE(bank, NOT river)").evaluate(doc)
    assert same_sentence("bank", ~kw("river")).evaluate(doc).count == 1
    assert not parse_rule("PARAGRAPH(bank, NOT river)").evaluate(doc)
    assert parse_rule("PARAGRAPH(river, NOT bank)").evaluate(doc)
    doc2 = make_doc("The bank by the river.", "bank", "river")
    assert not parse_rule("SENTENCE(bank, NOT river)").evaluate(doc2)
    with pytest.raises(ValueError):
        SameSentence(Not("a"), Not("b"))
    with pytest.raises(RuleSyntaxError):
        parse_rule("SENTENCE(NOT a, NOT b)")


@pytest.mark.parametrize(
    "source", ["NEAR(bank, NOT river, 3)", "BEFORE(NOT a, b)", "ATLEAST(2, NOT a)", "NEAR(NONE(a, b), c)"]
)
def test_not_operand_rejected_where_meaningless(source: str):
    with pytest.raises(RuleSyntaxError, match="NOT"):
        parse_rule(source)


def test_not_operand_rejected_by_constructors():
    with pytest.raises(ValueError, match="NOT"):
        near("a", ~kw("b"))
    with pytest.raises(ValueError, match="NOT"):
        before(~kw("a"), "b")
    with pytest.raises(ValueError, match="NOT"):
        at_least(2, ~kw("a"))
    assert near("a", ~kw("b") | "c")  # only a top-level NOT is meaningless


# ---------------------------------------------------------------------- 4. composite operands in one unit
def test_sentence_operands_evaluated_within_the_unit():
    doc = make_doc("Apple here. Banana and cherry there.", "apple", "banana", "cherry")
    assert not parse_rule("SENTENCE(apple AND banana, cherry)").evaluate(doc)
    assert parse_rule("SENTENCE(apple OR banana, cherry)").evaluate(doc)
    assert parse_rule("SENTENCE(NEAR(banana, cherry, 1), cherry)").evaluate(doc)
    doc = make_doc("Apple and banana. Cherry.\n\nApple, banana, cherry.", "apple", "banana", "cherry")
    res = parse_rule("SENTENCE(apple AND banana, cherry)").evaluate(doc)
    assert res and res.count == 1 and all(m.start > doc.text.index("\n\n") for m in res.matches)
    assert parse_rule("PARAGRAPH(SENTENCE(apple, banana), cherry)").evaluate(doc).count == 2


# ---------------------------------------------------------------------- 5. DSL lexing
@pytest.mark.parametrize(
    ("source", "name"),
    [
        (r"'it\'s'", "it's"),
        (r'"say \"hi\""', 'say "hi"'),
        (r"'a\\b'", "a\\b"),
        (r'"tab\tx"', "tab x"),  # whitespace is normalised in keyword names
        (r'"été"', "été"),
        ("don't", "don't"),
        ("rock'n'roll", "rock'n'roll"),
        ("O'Brien", "O'Brien"),
        ("x²", "x²"),
    ],
)
def test_dsl_strings_and_words(source: str, name: str):
    rule = parse_rule(source)
    assert isinstance(rule, KeywordRule) and rule.name == name
    assert parse_rule(rule.to_dsl()) == rule


@pytest.mark.parametrize(
    "bad",
    [r'"a\q"', '"line\nbreak"', "'open", "NEAR(a, b, ²)", "ATLEAST(², a)", r'"\u12"', '"trailing\\'],
)
def test_dsl_malformed_always_rule_syntax_error(bad: str):
    with pytest.raises(RuleSyntaxError) as exc:
        parse_rule(bad)
    assert 0 <= exc.value.pos <= len(bad)


def test_dsl_apostrophe_words_in_functions():
    rule = parse_rule("NEAR(don't, O'Brien, 2) AND CATEGORY(rock'n'roll)")
    assert rule.keyword_names() == {"don't", "O'Brien"}
    assert parse_rule(rule.to_dsl()) == rule


# ---------------------------------------------------------------------- 6. constructor validation + round trip
@pytest.mark.parametrize(
    "factory",
    [
        lambda: Near("a", "b", True),
        lambda: Near("a", "b", -1),
        lambda: Near("a", "b", 2.5),
        lambda: Near("a", "b", 3, ordered="yes"),
        lambda: AnyOf("a", "b", min_count=True),
        lambda: AnyOf("a", "b", min_count=0),
        lambda: AnyOf("a", "b", min_count=3),
        lambda: AtLeast("a", True),
        lambda: AtLeast("a", 0),
        lambda: AtLeast("a", 1.0),
        lambda: CategoryRule(""),
        lambda: CategoryRule(None),
        lambda: TagRule("   "),
        lambda: TagRule(5),
    ],
)
def test_constructor_validation(factory: Any):
    with pytest.raises((ValueError, TypeError)):
        factory()


def test_parser_rejects_impossible_any():
    with pytest.raises(RuleSyntaxError):
        parse_rule("ANY(3, a, b)")


@pytest.mark.parametrize("weight", [math.nan, math.inf, -math.inf, "abc", None, True])
def test_weights_must_be_finite_numbers(weight: Any):
    with pytest.raises(ValueError, match="weight"):
        RuleEngine(rules={"r": {"rule": "a", "weight": weight}})


def test_negative_weights_are_penalties():
    engine = RuleEngine(rules={"good": {"rule": "a", "weight": 2}, "bad": {"rule": "b", "weight": -1.5}})
    assert engine.score("a b") == 0.5


NAMES = ["bank", "x²", "²", "42", "don't", 'a"b', "a\\b", "AND", "near", "café au lait", "O'Brien", "line\x01ctl"]


def _unwrap_not(rule: Rule) -> Rule:
    while isinstance(rule, Not):
        rule = rule.rule
    return rule


def _random_rule(rng: random.Random, depth: int = 0) -> Rule:
    leaf = depth > 3 or rng.random() < 0.3
    if leaf:
        kind = rng.randrange(3)
        name = rng.choice(NAMES)
        return KeywordRule(name) if kind == 0 else CategoryRule(name) if kind == 1 else TagRule(name)
    op = rng.randrange(8)
    sub = [_random_rule(rng, depth + 1) for _ in range(rng.randint(2, 3))]
    pos = [_unwrap_not(r) for r in sub]  # operands that may not be a NOT
    if op == 0:
        return AllOf(*sub)
    if op == 1:
        return AnyOf(*sub, min_count=rng.randint(1, len(sub)))
    if op == 2:
        return Not(sub[0])
    if op == 3:
        return AtLeast(pos[0], rng.randint(1, 4))
    if op == 4:
        return Near(pos[0], pos[1], rng.randint(0, 9), rng.random() < 0.5)
    if op == 5:
        return SameSentence(pos[0], *sub[1:-1], Not(sub[-1]))
    if op == 6:
        return SameParagraph(*sub)
    return AllOf(sub[0], Not(sub[1]))


@pytest.mark.parametrize("seed", range(200))
def test_to_dsl_always_round_trips(seed: int):
    rule = _random_rule(random.Random(seed))
    again = parse_rule(rule.to_dsl())
    assert again.to_dsl() == rule.to_dsl()


# ---------------------------------------------------------------------- 7. case-insensitive CATEGORY / TAG
def test_category_and_tag_are_case_insensitive():
    doc = make_doc("the bank", Keyword("bank", category="finance", tags=["Hot"]))
    assert category("Finance").evaluate(doc) and parse_rule("CATEGORY(FINANCE)").evaluate(doc)
    assert tag("hot").evaluate(doc) and parse_rule("TAG(HOT)").evaluate(doc)
    assert not category("sports").evaluate(doc)


# ---------------------------------------------------------------------- 8. BEFORE requires no overlap
def test_before_requires_a_to_end_before_b():
    doc = make_doc("the interest rate rose", "rate", "interest rate")
    assert not parse_rule('BEFORE(rate, "interest rate")').evaluate(doc)
    assert not parse_rule('BEFORE("interest rate", rate)').evaluate(doc)
    assert parse_rule('NEAR(rate, "interest rate", 0)').evaluate(doc)  # NEAR still accepts overlap
    doc = make_doc("interest rate and a rate", "rate", "interest rate")
    assert parse_rule('BEFORE("interest rate", rate, 2)').evaluate(doc)


# ---------------------------------------------------------------------- 9. rule terms resolve to keywords
def test_rule_term_resolves_to_existing_keyword():
    kl = KeywordList([Keyword("machine learning", acronyms=["ML"], synonyms=["statistical learning"])])
    engine = RuleEngine(kl, {"ml": "ML", "syn": '"statistical learning"', "plain": "deep learning"})
    assert "ML" not in engine.matcher.keyword_list.names
    assert "deep learning" in engine.matcher.keyword_list.names
    result = engine.evaluate("We use machine learning daily.")
    assert result.fired == ["ml", "syn"] or set(result.fired) == {"ml", "syn"}
    assert result["ml"].keywords == ["machine learning"]
    assert engine.validate() == [
        "rule 'plain': keyword 'deep learning' is not in the keyword list (it will be added as a plain keyword)"
    ]


def test_ambiguous_rule_term_warns_and_keeps_separate_keyword():
    kl = KeywordList([Keyword("machine learning", acronyms=["ML"]), Keyword("markup language", acronyms=["ML"])])
    engine = RuleEngine(kl, {"r": "ML"})
    with pytest.warns(UserWarning, match="several keywords"):
        names = engine.matcher.keyword_list.names
    assert "ML" in names
    assert any("several keywords" in p for p in engine.validate())


def test_exact_keyword_name_wins_over_term():
    kl = KeywordList([Keyword("AI"), Keyword("artificial intelligence", acronyms=["AI"])])
    engine = RuleEngine(kl, {"r": "AI"})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert engine.evaluate("AI rocks")["r"].keywords == ["AI"]


# ---------------------------------------------------------------------- 10. malformed rules files
@pytest.mark.parametrize(
    ("suffix", "content", "message"),
    [
        (".json", "", "empty"),
        (".json", "[1, 2]", "top level"),
        (".json", '{"rules": {"r": {"weight": 2}}}', "missing the 'rule' key"),
        (".json", '{"rules": ["a"]}', '"rules" must be an object'),
        (".json", "{not json", "invalid JSON"),
        (".yaml", "", "empty"),
        (".yaml", "- a\n- b\n", "top level"),
        (".yaml", "rules: [unclosed", "invalid YAML"),
    ],
)
def test_malformed_rules_files(tmp_path, suffix: str, content: str, message: str):
    if suffix == ".yaml":
        pytest.importorskip("yaml")
    path = tmp_path / f"rules{suffix}"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message) as exc:
        RuleEngine.load(path)
    assert str(path) in str(exc.value)


def test_rule_syntax_errors_name_the_rule(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text(json.dumps({"rules": {"good": "a", "broken": "a AND"}}), encoding="utf-8")
    with pytest.raises(RuleSyntaxError, match="rule 'broken'") as exc:
        RuleEngine.load(path)
    assert exc.value.rule_name == "broken" and exc.value.pos == 5
    with pytest.raises(RuleSyntaxError, match="rule 'x'"):
        RuleEngine().add_rule("x", "NEAR(")


def test_load_strips_bom(tmp_path):
    path = tmp_path / "rules.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"rules": {"r": "bank"}}).encode())
    assert RuleEngine.load(path).classify("the bank") == ["r"]


def test_missing_pyyaml_gives_friendly_error(tmp_path, monkeypatch):
    path = tmp_path / "rules.yaml"
    path.write_text("rules: {r: a}\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(ImportError, match=r"keylex\[yaml\]"):
        RuleEngine.load(path)


# ---------------------------------------------------------------------- 11. validate()
def test_validate_reports_problems():
    kl = KeywordList([Keyword("bank", category="finance", tags=["money"])])
    engine = RuleEngine(kl, {"ok": "bank AND CATEGORY(Finance) AND TAG(MONEY)", "bad": "CATEGORY(sports) OR TAG(x)"})
    problems = engine.validate()
    assert len(problems) == 2
    assert all("rule 'bad'" in p for p in problems)
    assert RuleEngine(rules={"r": "a AND b"}).validate() == []  # no keyword list: auto-creation is intended
    assert RuleEngine(kl, {"ok": "bank"}).validate() == []


# ---------------------------------------------------------------------- 12. caching + concise explain
class CountingRule(Rule):
    calls = 0

    def evaluate(self, doc: Document) -> RuleResult:
        CountingRule.calls += 1
        return doc.evaluate(KeywordRule("bank"))

    def to_dsl(self) -> str:
        return "bank"


def test_shared_sub_rules_are_evaluated_once_per_document():
    CountingRule.calls = 0
    shared = CountingRule()
    engine = RuleEngine(rules={"a": AllOf(shared, "x"), "b": AnyOf(shared, "y"), "c": Not(shared)})
    engine.evaluate("bank x")
    assert CountingRule.calls == 1
    engine.evaluate("bank y")
    assert CountingRule.calls == 2


def test_explain_grows_linearly():
    rule: Rule = KeywordRule("k0")
    for i in range(1, 150):
        rule = AllOf(rule, KeywordRule(f"k{i}")) if i % 2 else AnyOf(rule, KeywordRule(f"k{i}"))
    doc = make_doc("k0 k1 k2", "k0", "k1", "k2")
    text = doc.evaluate(rule).explain()
    lines = text.splitlines()
    assert lines[0].endswith(rule.to_dsl()) or rule.to_dsl() in lines[0]  # top level: full rule
    assert all(len(line) < 400 for line in lines[1:])
    assert text.isascii()
    nested = RuleEngine(rules={"r": "bank AND NOT NEAR(bank, river, 5)"}).evaluate("the bank").explain()
    assert "[+] (bank AND NOT NEAR(bank, river, 5))" in nested


# ---------------------------------------------------------------------- CLI (13-16)
def run(capsys, *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.fixture
def rules_file(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text(
        json.dumps(
            {
                "rules": {
                    "money": {"rule": "bank AND NOT river", "weight": 2},
                    "nature": "river",
                    "secret": "password",
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_cli_stdin_is_utf8(capsys, monkeypatch):
    raw = io.BytesIO("﻿café crème".encode())
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(raw, encoding="cp1252"))
    code, out, _ = run(capsys, "match", "-w", "café", "--json")
    assert code == 0 and json.loads(out)[0] == {**json.loads(out)[0], "text": "café", "start": 0}


def test_cli_file_encoding_bom_and_decode_errors(capsys, tmp_path):
    bom = tmp_path / "bom.txt"
    bom.write_bytes(b"\xef\xbb\xbfbank here")
    code, out, _ = run(capsys, "match", "-w", "bank", "--json", str(bom))
    assert code == 0 and json.loads(out)[0]["start"] == 0
    latin = tmp_path / "latin.txt"
    latin.write_bytes("café bank".encode("latin-1"))
    code, out, err = run(capsys, "match", "-w", "café", "--json", str(latin))
    assert code == 1 and "not valid utf-8-sig" in err
    code, out, err = run(capsys, "match", "-w", "café", "--json", "--encoding", "latin-1", str(latin))
    assert code == 0 and err == ""
    code, _, err = run(capsys, "match", "-w", "x", "--encoding", "nope", str(latin))
    assert code == 2 and "unknown encoding" in err


def test_cli_exit_codes(capsys, tmp_path, rules_file):
    text = tmp_path / "t.txt"
    text.write_text("the bank", encoding="utf-8")
    assert run(capsys, "match", "-w", "bank", str(text))[0] == 0
    assert run(capsys, "match", "-w", "lake", str(text))[0] == 1
    assert run(capsys, "match", str(text))[0] == 2
    assert run(capsys, "match", "-w", "bank", str(tmp_path / "missing.txt"))[0] == 2
    assert run(capsys, "match", "-w", "bank", str(tmp_path))[0] == 2
    assert run(capsys, "rules", str(rules_file), str(text))[0] == 0
    assert run(capsys, "rules", str(rules_file), str(tmp_path / "missing.txt"))[0] == 2
    assert run(capsys, "rules", str(tmp_path / "missing.json"), str(text))[0] == 2
    assert run(capsys, "extract", str(text))[0] == 0
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    assert run(capsys, "extract", str(empty))[0] == 1


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "empty"),
        ("[]", "top level"),
        ('{"rules": {"r": {"weight": 1}}}', "missing the 'rule' key"),
        ('{"rules": {"broken": "a AND"}}', "rule 'broken'"),
    ],
)
def test_cli_malformed_rules_file_is_clean(capsys, tmp_path, content: str, message: str):
    path = tmp_path / "r.json"
    path.write_text(content, encoding="utf-8")
    code, _, err = run(capsys, "rules", str(path), "-")
    assert code == 2 and message in err and "Traceback" not in err


def test_cli_highlight_json_and_explain_json(capsys, tmp_path, rules_file):
    text = tmp_path / "t.txt"
    text.write_text("the bank", encoding="utf-8")
    code, out, _ = run(capsys, "match", "-w", "bank", "-o", "highlight", "--json", str(text))
    assert code == 0 and json.loads(out) == "the [bank]"
    code, out, _ = run(capsys, "rules", str(rules_file), str(text), "--json", "--explain")
    data = json.loads(out)
    assert data["fired"] == ["money"] and "[+]" in data["explanation"]


def test_cli_jsonl(capsys, tmp_path, rules_file):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("the bank", encoding="utf-8")
    b.write_text("the river", encoding="utf-8")
    _, out, _ = run(capsys, "rules", str(rules_file), str(a), str(b), "--format", "jsonl")
    records = [json.loads(line) for line in out.splitlines()]
    assert [r["file"] for r in records] == [str(a), str(b)]
    assert [r["fired"] for r in records] == [["money"], ["nature"]]
    assert all(set(r) == {"file", "fired", "score", "rules"} for r in records)
    _, out, _ = run(capsys, "match", "-w", "bank", str(a), "--format", "jsonl")
    (record,) = [json.loads(line) for line in out.splitlines()]
    assert set(record) == {"file", "matched", "result"} and record["result"][0]["keyword"] == "bank"
    _, out, _ = run(capsys, "extract", str(a), "--format", "jsonl")
    assert set(json.loads(out.splitlines()[0])) == {"file", "keywords"}
    assert run(capsys, "match", "-w", "x", "--json", "--format", "jsonl", str(a))[0] == 2


def test_cli_min_score_and_fail_on(capsys, tmp_path, rules_file):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("the bank", encoding="utf-8")
    b.write_text("my password", encoding="utf-8")
    assert run(capsys, "rules", str(rules_file), str(a), "--min-score", "2")[0] == 0
    code, _, err = run(capsys, "rules", str(rules_file), str(a), str(b), "--min-score", "2")
    assert code == 1 and "b.txt: score 1 < --min-score 2" in err
    assert run(capsys, "rules", str(rules_file), str(a), "--fail-on", "secret")[0] == 0
    code, _, err = run(capsys, "rules", str(rules_file), str(a), str(b), "--fail-on", "secret", "--fail-on", "nature")
    assert code == 1 and "'secret' fired" in err
    empty = tmp_path / "e.txt"
    empty.write_text("nothing", encoding="utf-8")
    assert run(capsys, "rules", str(rules_file), str(empty), "--fail-on", "secret")[0] == 0  # nothing fired = ok
    assert run(capsys, "rules", str(rules_file), str(a), "--fail-on", "nope")[0] == 2


def test_cli_negatable_flags_override_rules_file(capsys, tmp_path):
    rules = tmp_path / "r.json"
    rules.write_text(json.dumps({"rules": {"r": "policy"}, "options": {"stemmer": "light"}}), encoding="utf-8")
    text = tmp_path / "t.txt"
    text.write_text("new policies", encoding="utf-8")
    assert run(capsys, "rules", str(rules), str(text))[0] == 0
    assert run(capsys, "rules", str(rules), str(text), "--no-stem")[0] == 1
    rules.write_text(json.dumps({"rules": {"r": "lender"}, "options": {"match_type": "fuzzy"}}), encoding="utf-8")
    text.write_text("a lendr", encoding="utf-8")
    assert run(capsys, "rules", str(rules), str(text))[0] == 0
    assert run(capsys, "rules", str(rules), str(text), "--no-fuzzy")[0] == 1
    assert run(capsys, "match", "-w", "Bank", "--case-sensitive", "--no-case-sensitive", str(text))[0] == 1


def test_cli_duplicate_inputs_are_processed_once(capsys, tmp_path):
    text = tmp_path / "t.txt"
    text.write_text("bank", encoding="utf-8")
    code, out, err = run(capsys, "match", "-w", "bank", "--format", "jsonl", str(text), str(text))
    assert code == 0 and len(out.splitlines()) == 1 and "more than once" in err


def test_cli_check(capsys, tmp_path, rules_file):
    code, out, _ = run(capsys, "rules", "--check", str(rules_file))
    assert code == 0 and "ok (3 rules)" in out
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps({"keywords": [{"keyword": "bank", "category": "finance"}], "rules": {"r": "CATEGORY(sport)"}}),
        encoding="utf-8",
    )
    code, out, _ = run(capsys, "rules", "--check", str(bad))
    assert code == 1 and "category 'sport'" in out
