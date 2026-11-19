from __future__ import annotations

import io
import json
import sys

import pytest

from keylex.cli import main


@pytest.fixture
def files(tmp_path):
    text = tmp_path / "doc.txt"
    text.write_text("The Federal Reserve met. Machne lerning in the U.S.\nA river bank.\n", encoding="utf-8")
    kw = tmp_path / "kw.json"
    kw.write_text(json.dumps([{"keyword": "United States", "acronyms": ["US"]}, "machine learning"]), encoding="utf-8")
    rules = tmp_path / "rules.json"
    rules.write_text(
        json.dumps({"rules": {"fed": {"rule": '"Federal Reserve"', "weight": 2}, "river": "NEAR(bank, river, 2)"}}),
        encoding="utf-8",
    )
    return text, kw, rules


def run(capsys, *argv: str) -> tuple[int, str]:
    code = main(list(argv))
    return code, capsys.readouterr().out


def test_match_basic_and_exit_codes(capsys, files):
    text, kw, _ = files
    code, out = run(capsys, "match", "-k", str(kw), str(text))
    assert code == 0 and "'U.S.' -> United States  (acronym: US)" in out
    code, out = run(capsys, "match", "-w", "nothing", str(text))
    assert code == 1 and out == ""


def test_match_fuzzy_count_json_highlight(capsys, files):
    text, kw, _ = files
    _, out = run(capsys, "match", "-k", str(kw), "--fuzzy", "-o", "count", str(text))
    assert "machine learning  1" in out
    _, out = run(capsys, "match", "-k", str(kw), "--json", str(text))
    assert json.loads(out)[0]["keyword"] == "United States"
    _, out = run(capsys, "match", "-w", "river", "-o", "highlight", str(text))
    assert "[river]" in out
    other = text.with_name("other.txt")
    other.write_text("A river.", encoding="utf-8")
    _, out = run(capsys, "match", "-w", "river", "-w", "lake", "-o", "count", "--all", str(text), str(other))
    assert out.count("lake   0") == 2 and "== " in out


def test_match_stdin(capsys, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("bank policies"))
    code, out = run(capsys, "match", "-w", "policy", "--stem", "-o", "keywords")
    assert code == 0 and out.strip() == "policy"


def test_extract(capsys, files, tmp_path):
    text, _, _ = files
    save = tmp_path / "out.json"
    code, out = run(capsys, "extract", str(text), "-n", "3", "--save", str(save))
    assert code == 0 and len(out.strip().splitlines()) == 3
    assert save.exists()
    _, out = run(capsys, "extract", str(text), "-m", "rake", "--json")
    assert isinstance(json.loads(out), list)


def test_rules(capsys, files):
    text, _, rules = files
    code, out = run(capsys, "rules", str(rules), str(text), "--explain")
    assert code == 0 and "fired: fed, river" in out and "[+]" in out
    _, out = run(capsys, "rules", str(rules), str(text), "--json")
    assert json.loads(out)["score"] == 3.0


def test_rules_file_options_not_overridden(capsys, tmp_path):
    rules = tmp_path / "r.json"
    rules.write_text(json.dumps({"rules": {"r": "policy"}, "options": {"stemmer": "light"}}), encoding="utf-8")
    text = tmp_path / "t.txt"
    text.write_text("new policies", encoding="utf-8")
    code, out = run(capsys, "rules", str(rules), str(text))
    assert code == 0 and "fired: r" in out


def test_errors(capsys, files):
    text, _, _ = files
    assert main(["match", str(text)]) == 2  # no keywords: usage error
    assert main(["match", "-w", "x", "missing-file.txt"]) == 2
    assert main(["match", "-k", "missing.json", str(text)]) == 2
    with pytest.raises(SystemExit) as exc:
        main(["match", "--output", "nope"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
