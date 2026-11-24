from __future__ import annotations

import pytest

from keylex import Keyword, KeywordList


@pytest.fixture
def kl() -> KeywordList:
    return KeywordList(
        [
            Keyword("artificial intelligence", acronyms="AI", category="tech", tags=["hot"]),
            Keyword("bank", synonyms=["lender"], category="finance"),
            "python",
        ]
    )


def test_basic_access(kl: KeywordList):
    assert len(kl) == 3
    assert kl.names == ["artificial intelligence", "bank", "python"]
    assert kl[0].name == "artificial intelligence"
    assert kl["BANK"].name == "bank"
    assert isinstance(kl[1:], KeywordList) and kl[1:].names == ["bank", "python"]
    assert kl.get("missing") is None
    with pytest.raises(KeyError):
        kl["missing"]
    assert kl["bank"].related_terms == ["lender"]
    assert [k.name for k in kl.find_term("ai")] == ["artificial intelligence"]


def test_on_duplicate_policies():
    kl = KeywordList(["a"], on_duplicate="error")
    with pytest.raises(ValueError):
        kl.add("a")
    kl.add(Keyword("a", synonyms="x"), on_duplicate="skip")
    assert kl["a"].synonyms == []
    kl.add(Keyword("a", synonyms="x"), on_duplicate="replace")
    assert kl["a"].synonyms == ["x"]
    with pytest.raises(ValueError):
        KeywordList(on_duplicate="whatever")  # type: ignore[arg-type]


def test_remove_discard_clear(kl: KeywordList):
    kl.remove("bank")
    with pytest.raises(KeyError):
        kl.remove("bank")
    kl.discard("bank")
    kl.clear()
    assert len(kl) == 0


def test_rename_keeps_position(kl: KeywordList):
    kl.rename("bank", "banking")
    assert kl.names == ["artificial intelligence", "banking", "python"]
    assert kl["banking"].synonyms == ["lender"]
    with pytest.raises(ValueError):
        kl.rename("python", "banking")
    with pytest.raises(KeyError):
        kl.rename("nope", "x")


def test_categories_and_filter(kl: KeywordList):
    cats = kl.categories()
    assert {k: [x.name for x in v] for k, v in cats.items()} == {
        "tech": ["artificial intelligence"],
        "finance": ["bank"],
        None: ["python"],
    }
    assert kl.filter(category="finance").names == ["bank"]
    assert kl.filter(tag="hot").names == ["artificial intelligence"]
    assert kl.filter(lambda k: k.name.startswith("p")).names == ["python"]


def test_add_operator_and_copy(kl: KeywordList):
    other = KeywordList([Keyword("bank", synonyms="creditor"), "rust"])
    combined = kl + other
    assert combined["bank"].synonyms == ["lender", "creditor"]
    assert kl["bank"].synonyms == ["lender"]
    assert kl.copy() == kl


def test_from_dict_formats():
    assert KeywordList.from_dict({"a": None, "b": "bee"})["b"].synonyms == ["bee"]
    shorthand = KeywordList.from_dict({"AI": ["machine intelligence", {"acronyms": ["AI"], "variants": ["A.I."]}]})
    assert shorthand["AI"].variants == ["A.I."]
    with pytest.raises(ValueError):
        KeywordList.from_dict({"AI": [{"nope": ["x"]}]})
    with pytest.raises(TypeError):
        KeywordList.from_dict({"AI": [1]})


@pytest.mark.parametrize("suffix", [".json", ".csv", ".yaml"])
def test_save_and_load(tmp_path, kl: KeywordList, suffix: str):
    path = tmp_path / f"kw{suffix}"
    if suffix == ".yaml":
        yaml = pytest.importorskip("yaml")
        path.write_text(yaml.safe_dump(kl.to_list()), encoding="utf-8")
    else:
        kl.save(path)
    loaded = KeywordList.load(path)
    assert loaded.names == kl.names
    assert loaded["artificial intelligence"].acronyms == ["AI"]
    assert loaded["bank"].category == "finance"


@pytest.mark.parametrize(
    ("suffix", "content", "message"),
    [
        (".json", "{not json", "invalid JSON"),
        (".yaml", "bank: [unclosed\n", "invalid YAML"),
        (".csv", "keyword,metadata\nbank,{bad}\n", "invalid JSON in CSV field"),
        (".csv", "keyword,keyword_term\nbank,{bad}\n", "invalid JSON in CSV field"),
    ],
)
def test_load_reports_malformed_keyword_files(tmp_path, suffix, content, message):
    if suffix == ".yaml":
        pytest.importorskip("yaml")
    path = tmp_path / f"bad{suffix}"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message) as exc:
        KeywordList.load(path)
    assert str(path) in str(exc.value)


def test_load_text_and_csv(tmp_path):
    txt = tmp_path / "kw.txt"
    txt.write_text("# comment\nartificial intelligence | machine intelligence | AI\n\npython\n", encoding="utf-8")
    assert KeywordList.load(txt)["artificial intelligence"].synonyms == ["machine intelligence", "AI"]
    csv_text = "keyword,synonyms,acronyms,category,weight\nbank,lender;creditor,,finance,2\n"
    loaded = KeywordList.from_csv(csv_text)
    assert loaded["bank"].synonyms == ["lender", "creditor"] and loaded["bank"].weight == 2.0
    with pytest.raises(ValueError):
        KeywordList.from_csv("name\nx\n")


def test_repr():
    assert repr(KeywordList(["a", "b"])) == "KeywordList(['a', 'b'])"
    assert "+2" in repr(KeywordList([str(i) for i in range(10)]))
