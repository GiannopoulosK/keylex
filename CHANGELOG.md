# Changelog

## Unreleased

Fixes and improvements from a full review of 0.1.0 (adversarial probing of every subsystem).

### Matching
- **Unicode normalisation**: text and terms are NFKC-normalised before matching (`unicode_form=`), so
  decomposed accents, full-width letters and ligatures match; offsets still point into the original
  text. Keyword names are stored in NFC. `istanbul` now matches `İstanbul`.
- **`stemmer="light"`**: `movies` -> `movie` (was `movy`), and `news`, `does`, `series` etc. are left
  alone (the keyword `news` used to match "new"). Irregular plurals such as `crises` and `buses` are handled.
- **Context windows stay inside the match's sentence** (`context_scope="sentence"`, the new default;
  `"text"` restores the old behaviour).
- **`acronym_case=`** (`"strict"` default, `"title"` also accepts `Nato` for `NATO`, `"ignore"`).
- **Sentence splitting** no longer breaks after abbreviations (`Dr.`, `U.S.`, `e.g.`), initials
  (`J. K. Rowling`). This affects rules and extractors too.
- **Performance**: the "has the keyword list changed?" check is O(1) instead of O(keywords) per call
  (2,000 short texts with 100k keywords: 7.6 s -> 0.02 s). Fuzzy scoring is batched and multi-threaded
  when numpy is installed (`keylex[fast]`; 5,000 fuzzy keywords on 20k words: 22.6 s -> 1.6 s).
  Compiling 100k keywords is 35% faster.

### Keyword lists
- **CSV round-trips losslessly**: new `case_sensitive`, `fuzzy`, `context_window`, `keyword_term` and
  `metadata` columns. List cells fall back to JSON arrays when items contain `|` or `;`, which used to
  break regex patterns such as `(a|b)`.
- `KeywordList.load` accepts files with a UTF-8 BOM.
- New `KeywordList.find_conflicts()` lists surface forms shared by several keywords.

### Rules
- `NEAR` / `BEFORE` are near-linear instead of quadratic (5k words with a busy `NEAR`: 55 s -> 0.1 s).
- `ATLEAST(n, X)` counts occurrences of `X` (pairs for `NEAR`, units for `SENTENCE`), not matched
  spans; `RuleResult` gains `count` and `occurrences`.
- `SENTENCE` / `PARAGRAPH` evaluate every operand inside the same unit (`SENTENCE(a AND b, c)`), and
  `NOT` operands there mean "absent from that unit". `NOT` inside `NEAR`/`BEFORE`/`ATLEAST` is an error.
- `BEFORE` requires `a` to end before `b` starts; `CATEGORY`/`TAG` are case-insensitive; rules naming
  a synonym or acronym resolve to the keyword that owns it.
- DSL: every failure is a `RuleSyntaxError` (with the rule name when loading files); `'` escapes,
  apostrophes in bare words; constructors validate their arguments, and `to_dsl()` always round-trips.
- Rule files: clean errors for malformed files, UTF-8 BOM accepted. Shared sub-rules are evaluated
  once per document; `explain()` labels nested nodes concisely.
- New `RuleEngine.validate()`.

### Extraction
- Extracted phrases are always real surface forms: possessives end a phrase instead of being cut out
  ("Apple's iPhone" used to yield the non-existent "Apple iPhone").
- All-caps acronyms (`WHO`, `US`, `IT`) are no longer treated as stopwords; technical tokens (`AT&T`,
  `R&D`, `5G`, `C++`, `C#`, `Node.js`, `.NET`) stay whole.
- RAKE splits long content-word runs instead of dropping them.
- Quality: verbs no longer leak into phrases (`EXTRACTION_VERBS`), inner connectors are limited to
  "of", `&`, name particles and "Cap and Cap", capitalised names may span up to 6 words, MMR
  `diversity=` (default 0.3), `e-mail`/`email` dedup, ties broken by first position.
- TF-IDF: sublinear term frequency and corpus IDF that actually changes the ranking.
- New `language=` (`de`, `fr`, `es`, `it`, `el`), `get_stopwords()`, and an optional spaCy
  `pos_filter=`.
- **Behaviour change**: `top_n` must be a positive integer (`0` used to return 10); a string `corpus`
  raises `TypeError`.

### spaCy
- The component serialises its keywords and options (`to_disk`/`from_disk`/`to_bytes`/`from_bytes`);
  pipelines built with `add_to_pipeline` used to lose their keywords on save.
- `span._.keylex_matches` keeps every keyword sharing a span (they used to collide).
- **Behaviour changes**: `to_entity_ruler_patterns` emits `ORTH`/`LOWER` token patterns that respect case
  sensitivity (and warns about skipped regex terms); `set_ents` keeps existing entities unless
  `overwrite_ents=True`; matches not on token boundaries are skipped (`alignment_mode="strict"`).

### Highlighting
- **Behaviour change**: `highlight(..., escape=None)` HTML-escapes the whole text when the template
  looks like HTML (contains `<`), so custom templates no longer pass `<script>` through. Callables
  are escaped with `escape=True`. `matcher.highlight()` accepts `escape=` too.

### CLI
- `extract`: `--language`, `--diversity`; `-n` / `--max-ngram` must be positive.
- Exit codes: 0 = matched / fired / ok, 1 = nothing matched / check failed, 2 = usage or input error
  (a missing file used to exit 1). `extract` exits 1 when it finds nothing.
- Input is read as UTF-8, including stdin on Windows; BOMs are stripped; `--encoding` option; warning on
  undecodable bytes.
- New: `--format text|json|jsonl`, `rules --check`, `--min-score`, `--fail-on`, `--no-...` boolean flags,
  `--explain` included in JSON. `-o highlight --json` no longer prints a stray `[]`.

### Tooling
- CI runs Python 3.10-3.14, lints `examples/`, and tests the core without numpy/spaCy.

## 0.1.0

A rewrite of the matching core plus four new subsystems. Everything is covered by tests: 150+ tests,
about 96% coverage, `mypy --strict` clean.

### New
- **`Match` objects**: results carry the canonical keyword, the exact surface text and offsets, which
  term matched (`term`, `kind`), `method`, `score`, `category` and the `Keyword` itself.
- **`Term` objects**: per-term `case_sensitive`, `fuzzy` and `whole_word` options. Acronyms and all-caps
  terms are case-sensitive by default, and `US` also matches `U.S.`.
- **Richer `Keyword`**: `variants`, regex `patterns`, context rules (`exclude` / `require` within
  `context_window` words), `category`, `tags`, `weight` and `metadata`.
- **Windowed fuzzy matching**: typo-tolerant and multi-word, with exact offsets, every occurrence, a
  configurable RapidFuzz scorer and a length guard for short terms.
- **Rule engine** (`keylex.rules`): AND/OR/NOT, `NEAR`/`BEFORE`, `ATLEAST`, `ANY(n, ...)`, same
  sentence or paragraph, `CATEGORY`, `TAG`. There is a readable rule language (`parse_rule`), named
  weighted rules (`RuleEngine`), explanations, and JSON/YAML rule files.
- **Keyword extraction** without a keyword list (`extract_keywords`): YAKE, RAKE, TextRank and TF-IDF,
  all in pure Python.
- **spaCy integration**: `nlp.add_pipe("keylex")`, `add_to_pipeline()` and EntityRuler pattern export.
- **CLI**: `keylex match | extract | rules`.
- Overlap strategies (`longest`, `leftmost`, `all`), a light plural stemmer or any custom stemmer,
  accent folding, and highlighting (brackets, ANSI, HTML, Markdown, templates).
- `KeywordList` additions: `rename`, `filter`, `categories`, `on_duplicate` policies, CSV/TXT/YAML
  load and save, and `+`.
- Output formats `matches`, `categories` and `score`. `match_many`, `contains`, `count`, `score`.

### Fixed
- `to_dict()`/`to_json()` output can now be loaded back (it used to corrupt every keyword).
- Fuzzy matching only found exact substrings and only the first occurrence.
- `output="count"` counted substrings of keyword names instead of matches.
- Acronyms matched case-insensitively (`US` matched "us").
- Empty keywords matched everywhere; strings passed to `add_synonyms` were split into characters.
- Terms such as `C++`, `C#` and `.NET` never matched.
- `remove_keyword` crashed; renaming a keyword corrupted the list; duplicate adds silently lost data;
  `merge` shared objects between lists.
- Duplicate and overlapping matches; `OutputFormat` enum values rejected; options not validated;
  non-deterministic output order; `<=`/`>` on keywords; whitespace and hyphen variation.

### Changed
- `KeywordMatcher.match()` now returns `list[Match]` by default. `output="detailed"` returns dicts
  with `start`/`end` keys (previously `start_pos`/`end_pos`).
- `Keyword.keyword` is read-only (use `KeywordList.rename`). `Keyword` objects are unhashable because
  they are mutable; use `keyword.name` as a dict key.
- Exact matching is token-based and about 45× faster on large vocabularies (3,000 keywords over a 20k-word text: 7.5 s → 0.16 s).
- `fuzzy_algo=` is deprecated in favour of `scorer=`. The default fuzzy threshold is 85 with the `ratio` scorer.
- Python 3.10+ (the `<3.13` cap was removed). `rapidfuzz>=3,<4` instead of an exact pin.
- Removed the empty top-level `extractors/` package, `tox.ini` and the empty requirements files.

## 0.0.2
- Initial `Keyword`, `KeywordList` and `KeywordMatcher`.
