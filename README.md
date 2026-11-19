# keylex

**Keyword matching where keywords are objects.** A keyword in keylex knows its synonyms, acronyms,
spelling variants, regex patterns, the context it must (or must not) appear in, its category and
its weight. On top of that you get typo-tolerant fuzzy matching, a rule engine with a readable rule
language, unsupervised keyword extraction, a spaCy component and a CLI.

```bash
pip install keylex            # core (only dependency: rapidfuzz)
pip install "keylex[spacy]"   # + spaCy pipeline component
pip install "keylex[yaml]"    # + YAML keyword / rule files
```

Requires Python 3.10+.

---

## 30-second tour

```python
from keylex import Keyword, KeywordMatcher

matcher = KeywordMatcher([
    Keyword("artificial intelligence", synonyms=["machine intelligence"], acronyms=["AI"]),
    Keyword("United States", acronyms=["US", "USA"]),
    "machine learning",
])

text = "AI and machine learning are booming in the U.S. -- let us see."
for m in matcher.find(text):
    print(m.keyword, "|", m.text, "|", m.kind)
# artificial intelligence | AI | acronym
# machine learning | machine learning | keyword
# United States | U.S. | acronym          <- "U.S." matches the acronym US; "us" does not

print(matcher.match(text, "count"))
# {'artificial intelligence': 1, 'United States': 1, 'machine learning': 1}
```

## Contents

1. [Keywords](#keywords)
2. [Matching](#matching)
3. [Fuzzy matching](#fuzzy-matching)
4. [Rules](#rules)
5. [Keyword extraction](#keyword-extraction)
6. [spaCy](#spacy)
7. [Command line](#command-line)
8. [Loading and saving keywords](#loading-and-saving-keywords)

---

## Keywords

A `Keyword` has a canonical name plus any number of **terms**:

| Term kind | Example | Notes |
|-----------|---------|-------|
| `synonyms` | `"machine intelligence"` | case-insensitive by default |
| `acronyms` | `"AI"` | **case-sensitive**; `"US"` also matches `"U.S."` |
| `variants` | `"colour"`, `"A.I."` | spellings, inflections |
| `patterns` | `r"ORD-\d{4}"` | regular expressions |

```python
from keylex import Keyword, Term

apple = Keyword(
    "Apple",
    synonyms=["Apple Inc"],
    acronyms=["AAPL"],
    exclude=["pie", "fruit", "juice"],   # reject a match if one of these is within 10 words
    context_window=10,
    category="company",
    tags=["tech", "nasdaq"],
    weight=2.0,
    metadata={"ticker": "AAPL"},
)

# fine-grained control per term
apple.add_synonyms(Term("iPhone maker", fuzzy=False), Term("apple computer", case_sensitive=False))

apple.synonyms        # ['Apple Inc', 'iPhone maker', 'apple computer']
"aapl" in apple       # True  (membership is case-insensitive)
apple.to_dict()       # JSON-ready; Keyword.from_dict() restores it exactly
```

Strings and lists are both accepted everywhere: `add_synonyms("a")`, `add_synonyms("a", "b")` and
`add_synonyms(["a", "b"])` all do what you'd expect.

**Case-sensitivity rules** (first one that applies wins): the term's own `case_sensitive` → the
keyword's `case_sensitive` → acronyms and all-caps terms like `NASA` / `G7` are case-sensitive →
the matcher's default (`case_sensitive=False`).

### KeywordList

```python
from keylex import Keyword, KeywordList

kl = KeywordList(["python", Keyword("rust", category="language")])
kl.add(Keyword("python", synonyms=["py3"]))   # duplicate names merge by default
kl["python"].synonyms                         # ['py3']
kl.rename("rust", "Rust")
kl.filter(category="language").names          # ['Rust']
kl.categories()                               # {None: [python], 'language': [Rust]}
"PY3" in kl                                   # True: names and terms, case-insensitive
```

`on_duplicate` can be `"merge"` (default), `"skip"`, `"replace"` or `"error"`.

---

## Matching

```python
from keylex import KeywordMatcher

matcher = KeywordMatcher(
    ["New York", "New York City", "City Hall", "e-mail", "C++"],
    overlaps="longest",      # or "leftmost", "all"
    stemmer="light",         # plurals: "policies" matches "policy"; or pass any str -> str stemmer
    ignore_accents=False,    # True: "cafe" matches "café"
)

matcher.match("New York City Hall", "keywords")    # ['New York City']
matcher.match("email, e mail, E-Mail", "count")["e-mail"]   # 3
matcher.match("C++ and C#", "keywords")            # ['C++']
```

What the exact matcher handles for you:

* word boundaries (`cat` does not match `category`), including terms like `C++`, `C#`, `.NET`
* flexible whitespace, line breaks and hyphens (`New York` = `New  York` = `New-York`; `e-mail` = `email`)
* possessives and punctuation (`Apple's`, `(Apple)`)
* dotted acronyms (`US` matches `U.S.`)
* overlapping and nested keywords, resolved by the `overlaps` strategy
* Unicode: text and terms are NFKC-normalised before matching, so decomposed accents (`e` + `◌́`),
  full-width letters (`ＡＢＣ`) and ligatures match their plain forms; offsets still point into the
  original text (`unicode_form="NFC"` or `None` to change this)
* hundreds of thousands of keywords: matching is one pass over the text using a token trie, and the
  index is rebuilt only when the keyword list actually changes

Acronym-like terms (`NASA`, `G7`) are case-sensitive by default so that "us" never matches `US`.
`acronym_case="title"` also accepts the Title-case spelling (`Nato`), `"ignore"` drops the rule:

```python
from keylex import KeywordMatcher

KeywordMatcher(["NATO"]).match("Nato and NATO", "count")                        # {'NATO': 1}
KeywordMatcher(["NATO"], acronym_case="title").match("Nato and NATO", "count")  # {'NATO': 2}
```

### Results: `Match` objects

`matcher.find(text)` (or `matcher(text)`) returns `Match` objects:

| field | meaning |
|-------|---------|
| `keyword` | canonical keyword name |
| `text`, `start`, `end` | the exact surface text and its offsets (`text == doc[start:end]`) |
| `term`, `kind` | which term matched and whether it was a keyword, synonym, acronym, variant or pattern |
| `method`, `score` | `"exact"` / `"fuzzy"` / `"pattern"`, and similarity (100 for exact) |
| `category`, `keyword_obj` | the keyword's category and the `Keyword` itself |

### Output formats

`matcher.match(text, output)` where `output` is one of:

| output | returns |
|--------|---------|
| `"matches"` (default) | `list[Match]` |
| `"detailed"` | `list[dict]`, JSON-ready |
| `"keywords"` | unique keyword names in order of appearance |
| `"boolean"` | `{keyword: bool}` for every keyword |
| `"count"` | `{keyword: int}` for every keyword |
| `"categories"` | `{category: int}` |
| `"score"` | sum of keyword weights (fuzzy matches scaled by score) |

Other helpers: `matcher.match_many(texts, output)`, `matcher.contains(text)`, `matcher.count(text)`,
`matcher.score(text)` and `matcher.highlight(text, style)`. The highlight styles are `"brackets"`,
`"labels"`, `"markdown"`, `"ansi"`, `"html"`, a template like `"<{keyword}>{text}</>"`, or a function.

```python
from keylex import KeywordMatcher

matcher = KeywordMatcher(["bank", "river"])
matcher.highlight("The bank by the river.")       # 'The [bank] by the [river].'
```

A matcher built from a `KeywordList` stays **live**: add a keyword or a synonym later and the next
call sees it. There's no need to rebuild anything.

### Context: `exclude` and `require`

```python
from keylex import Keyword, KeywordMatcher

company = Keyword("Apple", exclude=["pie", "fruit"], context_window=5)
river_bank = Keyword("bank", require=["river", "shore"], context_window=5)
matcher = KeywordMatcher([company, river_bank])

matcher.match("Apple pie recipe", "keywords")                          # []
matcher.match("Apple shares rose", "keywords")                         # ['Apple']
matcher.match("the bank of the river", "keywords")                     # ['bank']
matcher.match("I baked an apple pie. Apple shares rose.", "count")["Apple"]   # 1
```

The window counts words on both sides of the match and stays inside the match's sentence, so the
"pie" above only rules out the first "apple" (`context_scope="text"` lets it cross sentences).

---

## Fuzzy matching

`match_type="fuzzy"` (or `FuzzyMatcher`) adds typo-tolerant matching on top of exact matching. The text
is split into word windows the size of each term (±1 word), each window is scored with RapidFuzz,
and the best non-overlapping windows above `threshold` become matches with exact offsets.

```python
from keylex import FuzzyMatcher

matcher = FuzzyMatcher(["machine learning", "environment", "database"], threshold=85)
for m in matcher.find("Machne lerning for the enviroment, stored in a data base."):
    print(m.text, "->", m.keyword, round(m.score))
# Machne lerning -> machine learning 93
# enviroment -> environment 95
# data base -> database 94
```

* Scorers: `"ratio"` (default), `"partial_ratio"`, `"token_sort_ratio"`, `"token_set_ratio"`, `"WRatio"`,
  `"QRatio"`, or any callable `(a, b, **kw) -> float`.
* Acronyms, regex patterns and terms shorter than `min_fuzzy_length` (default 4) are never
  fuzzy-matched, which avoids "USA" ~ "USB" style false positives. You can override this per keyword
  (`Keyword(..., fuzzy=False)`) or per term (`Term("cat", fuzzy=True)`).
* Exact matches always win over fuzzy ones for the same span.
* Install `keylex[fast]` (numpy) for batched, multi-threaded scoring: about 14x faster with
  thousands of fuzzy keywords.

---

## Rules

Rules combine keyword matches into higher-level conditions: boolean logic, proximity, counts,
sentence/paragraph scope, categories and tags. You can build them in Python or write them as strings.

```python
from keylex.rules import RuleEngine

engine = RuleEngine(
    keywords=[{"keyword": "Federal Reserve", "acronyms": ["Fed"]}],
    rules={
        "monetary_policy": {
            "rule": '("interest rate" OR "rate hike") AND SENTENCE("Federal Reserve", inflation)',
            "weight": 3,
        },
        "money_bank": "bank AND NOT NEAR(bank, river, 5)",
        "inflation_story": "ATLEAST(2, inflation)",
    },
    stemmer="light",
)

result = engine.evaluate("The Fed fears inflation and may raise interest rates. Inflation is sticky.")
result.fired        # ['monetary_policy', 'inflation_story']  (highest weight first)
result.score        # 4.0
print(result.explain())
```

Keywords that a rule mentions but that aren't defined (like `inflation` above) are added automatically.

### Rule language

| syntax | meaning |
|--------|---------|
| `bank`, `"interest rate"`, `interest rate` | the keyword occurs (unquoted words form a phrase) |
| `a AND b`, `a & b` | both |
| `a OR b`, `a \| b` | either |
| `NOT a`, `!a`, `~a` | not |
| `NEAR(a, b, 5)` | within 5 words of each other (default 5) |
| `BEFORE(a, b, 5)` | `a` then `b`, within 5 words |
| `ATLEAST(3, a)` | at least 3 occurrences (`NEAR` counts pairs, `SENTENCE` counts sentences) |
| `ANY(2, a, b, c)` | at least 2 of them |
| `ALL(a, b)`, `NONE(a, b)` | all of them / none of them |
| `SENTENCE(a, b)`, `PARAGRAPH(a, b)` | in the same sentence / paragraph (every operand, even `a AND b`, inside that one unit) |
| `SENTENCE(bank, NOT river)` | `bank` in a sentence that has no `river` |
| `CATEGORY(finance)`, `TAG(hot)` | any keyword of that category / tag |

`AND` binds tighter than `OR`; use parentheses to group. Strings take JSON escapes (`'it\'s'`), and bare
words may contain apostrophes (`don't`); quote phrases that contain `and` / `or` / `not`
(`"rock and roll"`; unquoted it means `rock AND roll`). A rule that names a synonym or acronym (`ML`)
refers to the keyword that owns it. One match is one occurrence: `ATLEAST(2, inflation OR
CATEGORY(monetary))` needs two different matches, and `ANY(n, ...)` only counts operands satisfied by
different matches (`ANY(2, ML, "machine learning")` doesn't fire on one "ML"). `CATEGORY` and `TAG` are
case-insensitive. Syntax errors show the position of the problem and the name of the rule; `NOT`
inside `NEAR`, `BEFORE` or `ATLEAST` is rejected because an absence has no position.
`RuleEngine.diagnostics()` reports errors (impossible `ANY(n, ...)`, categories or tags that no
keyword has) and warnings (keywords that will be added automatically, lower-case `and` / `or` between
bare words); `validate()` returns all of their messages. Rules files may only contain the keys
`version`, `keywords`, `rules` and `options`, and a rule object only `rule`, `weight` and `description`.

### Python API

```python
from keylex import KeywordMatcher
from keylex.rules import Document, kw, near, category, same_sentence, any_of

rule = (kw("bank") & ~near("bank", "river", within=5)) | category("finance").at_least(3)
rule.to_dsl()     # '((bank AND NOT NEAR(bank, river, 5)) OR ATLEAST(3, CATEGORY(finance)))'

matcher = KeywordMatcher(["bank", "river"])
text = "The bank raised fees."
result = rule.evaluate(Document(text, matcher.find(text)))
bool(result), result.keywords     # (True, ['bank'])
```

Rule sets can be saved and loaded with `engine.save("rules.json")` and `RuleEngine.load("rules.json")`
(YAML works too).

---

## Keyword extraction

Find the important keywords of a text **without a keyword list**:

```python
from keylex import extract_keywords

article = """The Federal Reserve held interest rates steady on Wednesday. Fed officials said inflation
remains elevated, and the European Central Bank is expected to follow. Interest rates have been at
a two-decade high since July."""

for k in extract_keywords(article, top_n=5):              # YAKE by default
    print(f"{k.phrase:<25} {k.score:.2f}  x{k.count}")

extract_keywords(article, method="rake")                  # longer key phrases
extract_keywords(article, method="textrank")              # graph-based
```

| method | good for |
|--------|----------|
| `yake` (default) | single documents, short or long; uses casing, position, frequency and context |
| `rake` | longer multi-word key phrases |
| `textrank` | the concepts central to a document |
| `tfidf` | what makes a document special within a corpus: `TfidfExtractor().fit(corpus)` |

All extractors are pure Python with built-in stopword lists: English by default, `language="de"`
(`de`, `fr`, `es`, `it`, `el`), or your own `stopwords=[...]`. Near-duplicates (`neural network` /
`neural networks`, `e-mail` / `email`) are merged (their occurrences are counted together), and
`diversity=` (MMR, default 0.3) keeps
overlapping phrases such as "Apple iPhone" / "Apple iPhone sales" from crowding the list.
Without a tagger, verb heuristics keep verbs out of phrases while words that are also nouns stay
usable ("mean squared error", "free fall"); `pos_filter=spacy.load("en_core_web_sm")` lets real
part-of-speech tags decide instead and keeps only noun phrases. URLs, e-mail addresses, @handles and
#hashtags are never part of a phrase. Every extracted phrase is a
real surface form of the text (`AT&T`, `C++`, `Node.js` and all-caps acronyms such as `WHO` stay
intact), so `get_extractor("yake").to_keyword_list(text)` gives a `KeywordList` that finds them again.

---

## spaCy

```python
import spacy
from keylex import Keyword, KeywordList
from keylex.integrations.spacy import add_to_pipeline, to_entity_ruler_patterns

keywords = KeywordList([Keyword("United States", acronyms=["US"], category="GPE"), "machine learning"])

nlp = spacy.blank("en")
add_to_pipeline(nlp, keywords, set_ents=True)     # or nlp.add_pipe("keylex", config={"keywords": keywords.to_list()})
doc = nlp("Machine learning in the U.S.")
[(s.text, s.label_) for s in doc.spans["keylex"]]  # [('Machine learning', 'KEYWORD'), ('U.S.', 'GPE')]
doc.spans["keylex"][1]._.keylex_match.keyword      # 'United States'

patterns = to_entity_ruler_patterns(keywords)     # if you prefer spaCy's EntityRuler
```

Span labels come from the keyword category (`label="category"`), the keyword name (`"keyword"`)
or a fixed label (`"fixed"`).

* Keywords and options are saved with the pipeline (`nlp.to_disk()` / `spacy.load()`, `to_bytes()`).
* `span._.keylex_matches` lists every keyword behind a span (two keywords can share one surface form);
  `doc._.keylex_matches` is the full matcher output (several keylex components add to it).
* Processed docs stay serialisable (`doc.to_bytes()`, `DocBin(store_user_data=True)`,
  `nlp.pipe(texts, n_process=2)`); the matches are stored as plain data and rebuilt on access.
* `set_ents=True` keeps the entities already on the doc unless `overwrite_ents=True`.
* Matches that don't fall on token boundaries are skipped (`alignment_mode="strict"`).
* `to_entity_ruler_patterns` emits token patterns: `ORTH` for case-sensitive terms such as acronyms,
  `LOWER` otherwise; an acronym such as `US` also gets a pattern for spaCy's `U.S.` token. Regex terms
  can't be expressed and are skipped with a warning.

---

## Command line

```bash
keylex match -k keywords.json article.txt              # list matches (line:col, text, keyword)
keylex match -w python -w rust --fuzzy -o count *.md   # inline keywords, typo-tolerant, counts
cat article.txt | keylex match -k kw.csv -o highlight  # highlight in the terminal
keylex match -k kw.json --json article.txt             # JSON output
keylex extract article.txt -m rake -n 15 --save kw.json
keylex rules rules.json article.txt --explain
keylex rules rules.json docs/*.md --fail-on secret --min-score 2   # CI gate
keylex rules --check rules.json                        # validate a rules file (--strict: warnings fail too)
keylex match -k kw.json --format jsonl *.txt           # one JSON record per file
```

Exit codes: `0` when something matched / a rule fired / the check passed, `1` when nothing matched /
no rule fired / the check found errors (or `--fail-on` / `--min-score` were not met), `2` on usage or
input errors. Files are read as UTF-8 (a UTF-8/16/32 BOM is honoured, so PowerShell's `>` output works;
`--encoding` for anything else, with a warning on undecodable bytes). Glob patterns (`docs/*.md`) are
expanded by keylex when the shell doesn't (Windows); a pattern that matches nothing is an error.
`NO_COLOR` disables colours. Boolean options have `--no-...` forms that override a rules file's
`options`. Run `keylex <command> --help` for all options (`keylex rules --help` includes a rule
language summary).

---

## Loading and saving keywords

```python
from keylex import KeywordList

kl = KeywordList.from_dict({
    "artificial intelligence": {"synonyms": ["machine intelligence"], "acronyms": ["AI"]},
    "bank": ["lender", {"acronyms": ["BK"]}],     # shorthand form
    "python": None,
})
kl.to_json()          # versioned JSON; KeywordList.from_json() restores it exactly
```

`KeywordList.load(path)` and `kl.save(path)` support:

* **JSON** – the output of `to_json()`, a list of keyword dicts, or any `from_dict` form
* **CSV** – a `keyword` column plus optional `synonyms`, `acronyms`, `variants`, `patterns`,
  `exclude`, `require`, `tags` (separated by `|` or `;`, or a JSON array), `category`, `weight`,
  `case_sensitive`, `fuzzy` and `context_window`. `to_csv()` output round-trips without loss.
* **YAML** – the same structures (needs `keylex[yaml]`)
* **TXT** – one keyword per line; `keyword | synonym | synonym`; `#` starts a comment

Files may start with a UTF-8 BOM. `kl.find_conflicts()` lists surface forms shared by several
keywords (for example `Bank` and `bank`, or a synonym used twice), which all match the same text:

```python
from keylex import KeywordList

KeywordList(["Bank", "bank", "loan"]).find_conflicts()   # {'bank': ['Bank', 'bank']}
```

---

## Extending

Subclass `keylex.BaseMatcher` and implement `find(text) -> list[Match]` to get every output format,
`match_many`, `highlight` and rule-engine support for free. Rules can be extended the same way:
subclass `keylex.rules.Rule` and implement `evaluate(doc)` and `to_dsl()`.

## Development

```bash
pip install -e ".[dev,all]"
pytest                 # tests + doctests
ruff check . && ruff format --check .
mypy                   # strict
```

## License

Apache 2.0
