"""Regression tests for the rules / CLI review fixes (2026-11)."""

from __future__ import annotations

import io
import json
import pickle
import random
import sys
from pathlib import Path

import pytest

from keylex import Keyword, KeywordList
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
    RuleSyntaxError,
    SameParagraph,
    SameSentence,
    TagRule,
    all_of,
    any_of,
    at_least,
    near,
    not_,
    parse_rule,
    same_sentence,
)
from keylex.rules.dsl import MAX_DEPTH, lint_rule

BS = "\\"
EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def run(capsys, *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _engine_doc(rules: dict[str, str], text: str) -> RuleEngine:
    kws = [
        Keyword("inflation", category="monetary"),
        Keyword("interest rate", category="monetary"),
        Keyword("machine learning", acronyms=["ML"]),
        Keyword("bank"),
    ]
    return RuleEngine(kws, rules=rules)


# ---------------------------------------------------------------------- 1. one match = one occurrence
def test_or_counts_each_match_once():
    engine = _engine_doc(
        {
            "a": "ATLEAST(2, inflation OR CATEGORY(monetary))",
            "b": 'ATLEAST(2, ML OR "machine learning")',
            "c": "ATLEAST(2, inflation OR inflation)",
            "d": "ATLEAST(2, inflation OR Inflation)",
        },
        "",
    )
    result = engine.evaluate("Inflation is high. ML is used.")
    assert result.fired == []
    for name in "abcd":
        assert result[name].count == 0
        assert "1 found, need 2" in result[name].explain()  # explain agrees with the count
    result = engine.evaluate("Inflation and the interest rate. ML and machine learning.")
    assert set(result.fired) == {"a", "b"}
    assert result["a"].count == 2 and "2 found, need 2" in result["a"].explain()


def test_or_count_adds_non_match_counting_operands():
    engine = _engine_doc({"r": "NEAR(bank, inflation, 3) OR NEAR(bank, inflation, 3) OR ML"}, "")
    res = engine.evaluate("bank inflation. ML.")["r"]
    assert res.count == 2  # one NEAR pair (duplicates count once) + one ML


def test_any_n_counts_operands_with_distinct_matches():
    engine = _engine_doc(
        {"two": 'ANY(2, ML, "machine learning", bank)', "cat": "ANY(2, inflation, CATEGORY(monetary))"}, ""
    )
    result = engine.evaluate("We use ML daily.")
    assert "two" not in result
    assert "1 with distinct matches" in result["two"].explain()
    assert "two" in engine.evaluate("We use ML daily at the bank.")
    assert "cat" not in engine.evaluate("Inflation.")
    assert "cat" in engine.evaluate("Inflation and the interest rate.")


def test_any_n_identical_operands_count_once():
    text = "a a"
    doc = Document(text, [Match("a", 0, 1, "a", "a"), Match("a", 2, 3, "a", "a")])
    assert not doc.evaluate(parse_rule("ANY(2, a, a)"))
    assert doc.evaluate(parse_rule("ANY(2, NOT b, NOT c)"))
    assert not doc.evaluate(AnyOf(Not("b"), Not("b"), "zzz", min_count=2))


def test_any_n_count_is_nth_largest_distinct_count():
    words = "a a a b b c".split()
    matches, pos = [], 0
    for w in words:
        matches.append(Match(w, pos, pos + 1, w, w))
        pos += 2
    doc = Document(" ".join(words), matches)
    assert doc.evaluate(parse_rule("ANY(2, a, b, c)")).count == 2
    assert doc.evaluate(parse_rule("ANY(3, a, b, c)")).count == 1


# ---------------------------------------------------------------------- 2. single-operand AnyOf / AllOf
@pytest.mark.parametrize(
    "build",
    [
        lambda: near("a", any_of(not_("b"))),
        lambda: same_sentence(any_of(not_("a")), not_("b")),
        lambda: at_least(1, any_of(not_("x"))),
        lambda: at_least(1, all_of(not_("x"))),
        lambda: Near(all_of(any_of(not_("x"))), "b"),
    ],
)
def test_constructors_reject_what_the_parser_rejects(build):
    with pytest.raises(ValueError):
        build()


def test_single_operand_nodes_collapse():
    rule = all_of("a", any_of(all_of("b", "c")))
    again = parse_rule(rule.to_dsl())
    assert again == rule and len(rule.rules) == len(again.rules) == 3  # type: ignore[attr-defined]
    assert isinstance(parse_rule("ANY(1, a)"), KeywordRule)
    assert isinstance(parse_rule("ALL(NOT a)"), Not)
    assert parse_rule("NONE(a)") == Not("a")


def test_engine_save_load_round_trip_single_operand(tmp_path):
    engine = RuleEngine(rules={"r": not_(any_of("x")), "s": any_of(all_of("a"))})
    path = tmp_path / "r.json"
    engine.save(path)
    loaded = RuleEngine.load(path)
    assert {n: r.rule for n, r in loaded.rules.items()} == {n: r.rule for n, r in engine.rules.items()}


_NAMES = ["bank", "a b", "NEAR", "and", "x'y", 'q"z', "don't", "1", "12 monkeys", "C++", "café", "日本", "a" + BS + "b",
          "_u", "x²", "İ", "a,b", "(x)", "!x", "&", "-1", "1.5", chr(0x1F600), "or", "Not", "ATLEAST", "Kw", "Ａ"]  # fmt: skip


def _gen(rng: random.Random, depth: int = 0) -> Rule:
    k = rng.randint(0, 11 if depth < 4 else 2)
    if k <= 2:
        return KeywordRule(rng.choice(_NAMES))
    if k == 3:
        return CategoryRule(rng.choice(_NAMES)) if rng.random() < 0.5 else TagRule(rng.choice(_NAMES))
    if k == 4:
        return Not(_gen(rng, depth + 1))
    if k == 5:
        rules = [_gen(rng, depth + 1) for _ in range(rng.randint(1, 4))]  # incl. single-operand
        return AnyOf(*rules, min_count=rng.randint(1, len(rules)))
    if k == 6:
        return AllOf(*[_gen(rng, depth + 1) for _ in range(rng.randint(1, 4))])
    if k == 7:
        return AtLeast(_gen(rng, depth + 1), rng.randint(1, 4))
    if k == 8:
        return Near(_gen(rng, depth + 1), _gen(rng, depth + 1), rng.randint(0, 9), rng.random() < 0.5)
    if k == 11:
        return AnyOf(_gen(rng, depth + 1))
    rules = [_gen(rng, depth + 1) for _ in range(rng.randint(2, 4))]
    return (SameSentence if k == 9 else SameParagraph)(*rules)


def test_to_dsl_round_trip_fuzz():
    rng = random.Random(20261119)
    built = 0
    for _ in range(4000):
        try:
            rule = _gen(rng)
        except ValueError:
            continue  # NOT in a positive position etc.: rejected by the constructors, like the parser
        built += 1
        dsl = rule.to_dsl()
        again = parse_rule(dsl)
        assert again == rule, dsl
        assert again.to_dsl() == dsl
    assert built > 1000


# ---------------------------------------------------------------------- 3. examples pass --check
@pytest.mark.parametrize("extra", [[], ["-k", str(EXAMPLES / "keywords.json")]])
def test_examples_pass_check(capsys, extra):
    code, out, _ = run(capsys, "rules", "--check", str(EXAMPLES / "rules.json"), *extra)
    assert code == 0 and ": ok (3 rules" in out


def test_check_warnings_and_strict(capsys, tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"keywords": ["bank"], "rules": {"r": "bank AND river"}}), encoding="utf-8")
    code, out, _ = run(capsys, "rules", "--check", str(path))
    assert code == 0 and "warning: rule 'r': keyword 'river' is not in the keyword list" in out
    code, out, _ = run(capsys, "rules", "--check", "--strict", str(path))
    assert code == 1
    path.write_text(json.dumps({"keywords": ["bank"], "rules": {"r": "CATEGORY(nope)"}}), encoding="utf-8")
    code, out, _ = run(capsys, "rules", "--check", str(path))
    assert code == 1 and "error: rule 'r': no keyword has category" in out


# ---------------------------------------------------------------------- 4. strict rules files
@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"rule": {"x": "a"}}, "unknown key 'rule' (did you mean 'rules'?)"),
        ({"rules": {"x": {"rule": "a", "wieght": 3}}}, "rule 'x': unknown key 'wieght' (did you mean 'weight'?)"),
        ({"rules": {"x": "a"}, "options": {"foo": 1}}, "\"options\": unknown key 'foo'"),
        ({"rules": {"x": 5}}, "rule 'x': expected a rule string or an object, got 5"),
        ({"rules": {"x": {"rule": 5}}}, "rule 'x': \"rule\" must be a rule string"),
        ({"rules": {"x": "a"}, "keywords": [5]}, '"keywords": '),
        ({"rules": {"x": "a"}, "keywords": [{"nokey": 1}]}, '"keywords": '),
        ({"rules": {"x": "a"}, "keywords": "abc"}, '"keywords": must be a list or an object'),
    ],
)
def test_rules_file_errors_name_the_file(tmp_path, data, expected):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError) as info:
        RuleEngine.load(path)
    assert str(path) in str(info.value) and expected in str(info.value)


def test_rules_file_allows_version_and_reports_syntax_errors_with_file(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"version": 1, "rules": {"x": "a AND (b"}}), encoding="utf-8")
    with pytest.raises(RuleSyntaxError) as info:
        RuleEngine.load(path)
    assert str(info.value).startswith(f"{path}: rule 'x': ")
    again = pickle.loads(pickle.dumps(info.value))
    assert again.filename == str(path) and str(again) == str(info.value)


def test_unknown_matcher_option_in_python_is_a_value_error():
    with pytest.raises(ValueError, match=r"unknown key 'stemer'.*did you mean 'stemmer'"):
        RuleEngine(rules={"x": "a"}, stemer="light")


def test_check_with_no_rules_fails(capsys, tmp_path):
    path = tmp_path / "r.json"
    path.write_text('{"rules": {}}', encoding="utf-8")
    code, out, _ = run(capsys, "rules", "--check", str(path))
    assert code == 1 and "no rules defined" in out
    text = tmp_path / "t.txt"
    text.write_text("x", encoding="utf-8")
    code, _, err = run(capsys, "rules", str(path), str(text))
    assert code == 2 and "no rules defined" in err


def test_yaml_rule_names(tmp_path):
    pytest.importorskip("yaml")
    path = tmp_path / "r.yaml"
    path.write_text("rules:\n  1: bank\n", encoding="utf-8")
    assert list(RuleEngine.load(path).rules) == ["1"]
    path.write_text("rules:\n  yes: bank\n", encoding="utf-8")
    with pytest.raises(ValueError, match="rule names must be strings, got True"):
        RuleEngine.load(path)


# ---------------------------------------------------------------------- 5. empty names
@pytest.mark.parametrize(
    ("source", "pos", "what"),
    [('""', 0, "keyword"), ('"\t"', 0, "keyword"), ('a OR "  "', 5, "keyword"), ('CATEGORY("")', 9, "category"),
     ('TAG(" ")', 4, "tag"), ('KW("")', 3, "keyword")],
)  # fmt: skip
def test_empty_names_are_syntax_errors(source, pos, what):
    with pytest.raises(RuleSyntaxError) as info:
        parse_rule(source)
    assert info.value.pos == pos and info.value.message == f"Empty {what} name"


# ---------------------------------------------------------------------- 6. surrogates, atomic save
def _u(code: str) -> str:
    return BS + "u" + code


def test_surrogate_pairs_combine_like_json():
    rule = parse_rule('"' + _u("d83d") + _u("de00") + '"')
    assert isinstance(rule, KeywordRule) and rule.name == chr(0x1F600)
    assert parse_rule(rule.to_dsl()) == rule
    for bad in (_u("d83d"), _u("de00"), _u("d83d") + "x", _u("d83d") + _u("0041")):
        with pytest.raises(RuleSyntaxError, match="surrogate"):
            parse_rule('"' + bad + '"')
    with pytest.raises(ValueError, match="surrogate"):
        KeywordRule(chr(0xD83D))


def test_save_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "rules.json"
    path.write_text("previous content", encoding="utf-8")
    engine = RuleEngine(rules={"r": "bank"})
    monkeypatch.setattr(RuleEngine, "to_dict", lambda self: {"bad": object()})
    with pytest.raises(TypeError):
        engine.save(path)
    assert path.read_text(encoding="utf-8") == "previous content"
    assert [p.name for p in tmp_path.iterdir()] == ["rules.json"]


# ---------------------------------------------------------------------- 7. glob expansion
def test_cli_expands_glob_patterns(capsys, tmp_path):
    (tmp_path / "a.txt").write_text("bank", encoding="utf-8")
    (tmp_path / "b.txt").write_text("bank bank", encoding="utf-8")
    code, out, _ = run(capsys, "match", "-w", "bank", "-o", "count", str(tmp_path / "*.txt"))
    assert code == 0 and out.count("== ") == 2
    code, _, err = run(capsys, "match", "-w", "bank", str(tmp_path / "*.md"))
    assert code == 2 and "no files match" in err


# ---------------------------------------------------------------------- 8. BOMs / decode errors
@pytest.mark.parametrize("encoding", ["utf-16", "utf-16-be", "utf-32"])
def test_cli_reads_utf16_and_utf32_with_bom(capsys, tmp_path, encoding):
    text = tmp_path / "t.txt"
    bom_encoding = {"utf-16-be": "utf-16-be"}.get(encoding, encoding)
    data = "The Fed raised rates.".encode(bom_encoding)
    if encoding == "utf-16-be":
        data = b"\xfe\xff" + data
    text.write_bytes(data)
    code, out, _ = run(capsys, "match", "-w", "Fed", str(text))
    assert code == 0 and "'Fed' -> Fed" in out
    rules = tmp_path / "r.json"
    rules.write_bytes(b"\xfe\xff" + json.dumps({"rules": {"fed": "Fed"}}).encode("utf-16-be"))
    code, out, _ = run(capsys, "rules", str(rules), str(text))
    assert code == 0 and "fired: fed" in out


def test_rules_file_decode_error_names_the_file(capsys, tmp_path):
    rules = tmp_path / "r.json"
    rules.write_bytes('{"rules": {"x": "café"}}'.encode("latin-1"))
    code, _, err = run(capsys, "rules", "--check", str(rules))
    assert code == 2 and str(rules) in err


def test_keyword_file_errors_name_the_file(capsys, tmp_path):
    kw = tmp_path / "kw.json"
    kw.write_text('[{"nokey": 1}]', encoding="utf-8")
    code, _, err = run(capsys, "match", "-k", str(kw), "-")
    assert code == 2 and str(kw) in err


# ---------------------------------------------------------------------- 9. deep nesting, catch-all
def test_deep_nesting_is_a_syntax_error():
    for source in ("(" * 3000 + "a" + ")" * 3000, "NOT " * 3000 + "a", "NOT " * (MAX_DEPTH + 1) + "a"):
        with pytest.raises(RuleSyntaxError, match="nested too deeply"):
            parse_rule(source)
    rule = parse_rule("NOT " * MAX_DEPTH + "a")
    assert parse_rule(rule.to_dsl()) == rule


def test_cli_catch_all_returns_2(capsys, monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(RuleEngine, "load", boom)
    code, _, err = run(capsys, "rules", str(tmp_path / "r.json"), "-")
    assert code == 2 and err.startswith("keylex: error: RecursionError")


# ---------------------------------------------------------------------- 10. detailed output is JSON lines
def test_match_detailed_text_output_is_json_lines(capsys, tmp_path):
    text = tmp_path / "t.txt"
    text.write_text("bank and bank", encoding="utf-8")
    code, out, _ = run(capsys, "match", "-w", "bank", "-o", "detailed", str(text))
    lines = out.splitlines()
    assert code == 0 and len(lines) == 2
    assert [json.loads(line)["start"] for line in lines] == [0, 9]


# ---------------------------------------------------------------------- 11. README escape example
def test_readme_escape_example_parses():
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    example = "'it" + BS + "'s'"
    assert f"(`{example}`)" in readme
    assert parse_rule(example) == KeywordRule("it's")


# ---------------------------------------------------------------------- 12. broken pipe
def test_broken_pipe_exits_quietly(capsys, monkeypatch, tmp_path):
    import keylex.cli as cli

    def pipe(args):
        raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr(cli, "cmd_match", pipe)
    code, _, err = run(capsys, "match", "-w", "x", "-")
    assert code == 0 and err == ""


# ---------------------------------------------------------------------- 13. usability
def test_tty_stdin_without_files_is_an_error(capsys, monkeypatch):
    import keylex.cli as cli

    monkeypatch.setattr(cli, "_stdin_is_tty", lambda: True)
    code, _, err = run(capsys, "match", "-w", "bank")
    assert code == 2 and "stdin is a terminal" in err


def test_highlight_keeps_trailing_newline_once(capsys, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("a bank\n"))
    code, out, _ = run(capsys, "match", "-w", "bank", "-o", "highlight")
    assert code == 0 and out == "a [bank]\n"


def test_no_color_env(capsys, monkeypatch):
    import keylex.cli as cli

    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    args = cli.build_parser().parse_args(["match", "-w", "x"])
    assert not cli._use_color(args)


def test_rules_json_output_has_count_weight_description(capsys, tmp_path):
    text = tmp_path / "t.txt"
    text.write_text("The Fed fears inflation. Inflation is sticky.", encoding="utf-8")
    code, out, _ = run(capsys, "rules", str(EXAMPLES / "rules.json"), str(text), "--json")
    rule = json.loads(out)["rules"]["inflation_focus"]
    assert code == 0 and rule["count"] == 2 and rule["weight"] == 1 and rule["description"] == ""


def test_save_to_missing_directory_fails_fast(capsys, tmp_path):
    text = tmp_path / "t.txt"
    text.write_text("some text about keyword extraction", encoding="utf-8")
    code, _, err = run(capsys, "extract", str(text), "--save", str(tmp_path / "nope" / "kw.json"))
    assert code == 2 and "directory does not exist" in err


def test_help_texts(capsys):
    with pytest.raises(SystemExit):
        main(["rules", "--help"])
    out = capsys.readouterr().out
    assert "rule language" in out and "NEAR(a, b, 5)" in out
    with pytest.raises(SystemExit):
        main(["match", "--help"])
    out = capsys.readouterr().out
    assert "one JSON object per match" in out and "NO_COLOR" in out
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "keylex" in capsys.readouterr().out


def test_lint_lowercase_operators():
    assert lint_rule("rock and roll")[0].startswith("lower-case 'and'")
    assert lint_rule('"rock and roll"') == lint_rule("rock AND roll") == []
    assert lint_rule("a and NEAR(b, c)") == []
    kl = KeywordList(["rock", "roll"])
    engine = RuleEngine(kl, rules={"r": "rock and roll"})
    assert engine.diagnostics() == [("warning", engine.validate()[0])]
