# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-11-19

A rewrite of the matching core plus four new subsystems: rules, keyword extraction, spaCy
integration and a CLI. Nearly 700 tests (including every README example), `mypy --strict` clean,
CI on Linux, macOS and Windows with Python 3.10-3.14.

### Added
- **`Match` objects**: every result carries the canonical keyword, the exact surface text and its
  offsets into the original text, which term matched (`term`, `kind`), `method`, `score`,
  `category` and the `Keyword` itself.
- **`Term` objects** with per-term `case_sensitive`, `fuzzy` and `whole_word` options. Acronyms and
  all-caps terms are case-sensitive by default (`acronym_case="strict" | "title" | "ignore"`), and
  `US` also matches `U.S.`.
- **Richer `Keyword`**: `variants`, regex `patterns`, context rules (`exclude` / `require` within
  `context_window` words, scoped to the sentence by default), `category`, `tags`, `weight` and
  `metadata`.
- **Windowed fuzzy matching**: typo-tolerant and multi-word, with exact offsets, every occurrence,
  any RapidFuzz scorer and a length guard for short terms. Windows never cross sentence or clause
  boundaries. `keylex[fast]` (numpy) batches the scoring over all cores.
- **Rule engine** (`keylex.rules`): AND/OR/NOT, `NEAR`/`BEFORE`, `ATLEAST`, `ANY(n, ...)`,
  `SENTENCE`/`PARAGRAPH`, `CATEGORY`, `TAG`; a readable rule language (`parse_rule`, `to_dsl()`
  round-trips), named weighted rules (`RuleEngine`), `explain()`, `validate()`/`diagnostics()` and
  JSON/YAML rule files.
- **Keyword extraction** without a keyword list (`extract_keywords`): YAKE, RAKE, TextRank and
  TF-IDF in pure Python, with MMR diversity, near-duplicate merging, stopword lists for `en`, `de`,
  `fr`, `es`, `it` and `el`, and an optional spaCy POS filter. Extracted phrases are always real
  surface forms of the text.
- **spaCy integration**: `nlp.add_pipe("keylex")`, `add_to_pipeline()` and EntityRuler pattern
  export. Keywords are saved with the pipeline, and processed docs serialise (`DocBin`,
  `nlp.pipe(n_process=...)`).
- **CLI**: `keylex match | extract | rules` with text, JSON and JSONL output, highlighting,
  `rules --check`, `--fail-on` / `--min-score` CI gates and exit codes 0 (found) / 1 (nothing) /
  2 (error).
- Overlap strategies (`longest`, `leftmost`, `all`), a light plural stemmer or any custom stemmer,
  accent folding, NFKC normalisation with offsets mapped back to the original text, and
  highlighting (brackets, ANSI, HTML, Markdown, templates, callables; HTML-escaped).
- `KeywordList`: `rename`, `filter`, `categories`, `find_conflicts`, `on_duplicate` policies and
  lossless JSON / CSV / YAML / TXT load and save (atomic writes, BOM detection).
- `KeywordMatcher.configure()` / `options()`; options can also be changed by assignment and are
  picked up on the next call. Matchers are thread-safe and picklable.
- Output formats `matches`, `categories` and `score`, with precise `@overload` types for every
  format; `match_many`, `contains`, `count`, `score`.
- Benchmarks (`benchmarks/bench_exact.py`), a release workflow with PyPI trusted publishing, and
  pre-commit hooks.

### Changed
- `KeywordMatcher.match()` returns `list[Match]` by default. `output="detailed"` returns dicts
  with `start`/`end` keys (previously `start_pos`/`end_pos`).
- `Keyword.keyword` is read-only (use `KeywordList.rename`). `Keyword` objects are unhashable
  because they are mutable; use `keyword.name` as a dict key.
- Exact matching is a single pass over the text with a token trie: matching time no longer grows
  with the vocabulary (100,000 keywords over a 20,000-word text in about 20 ms).
- `fuzzy_algo=` is deprecated in favour of `scorer=`. The default fuzzy threshold is 85 with the
  `ratio` scorer.
- Python 3.10+ (the `<3.13` cap was removed); `rapidfuzz>=3,<4` instead of an exact pin.

### Fixed
- `to_dict()`/`to_json()` output can now be loaded back (it used to corrupt every keyword).
- Fuzzy matching only found exact substrings, and only the first occurrence.
- `output="count"` counted substrings of keyword names instead of matches.
- Acronyms matched case-insensitively (`US` matched "us").
- Empty keywords matched everywhere; strings passed to `add_synonyms` were split into characters.
- Terms such as `C++`, `C#` and `.NET` never matched.
- `remove_keyword` crashed; renaming a keyword corrupted the list; duplicate adds silently lost
  data; `merge` shared objects between lists.
- Duplicate and overlapping matches, `OutputFormat` enum values rejected, options not validated,
  non-deterministic output order, `<=`/`>` on keywords, whitespace and hyphen variation.

### Removed
- The empty top-level `extractors/` package, `tox.ini` and the empty requirements files.

## [0.0.2]
- Initial `Keyword`, `KeywordList` and `KeywordMatcher`.

[0.1.0]: https://github.com/GiannopoulosK/keylex/releases/tag/v0.1.0
[0.0.2]: https://pypi.org/project/keylex/0.0.2/
