# Contributing to keylex

Thanks for helping! Bug reports, feature ideas and pull requests are all welcome.

## Reporting a bug

Open an issue with a minimal snippet that reproduces the problem, what you expected, what you
got, and your Python and keylex versions (`python -m keylex --version`).

## Development setup

```bash
git clone https://github.com/GiannopoulosK/keylex
cd keylex
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev,all]"
python -m spacy download en_core_web_sm            # only for the spaCy tests
pre-commit install                                 # optional
```

## Before opening a pull request

```bash
pytest                                   # tests, doctests and the README examples
ruff check keylex tests examples benchmarks
ruff format keylex tests examples benchmarks
mypy                                     # strict
```

* Add a test for every bug fix and feature. Tests for a fixed bug should fail without the fix.
* README code blocks are executed by `tests/test_readme.py`: a line such as
  `expr   # ['value']` asserts that `expr == ['value']`, so keep examples true.
* Public functions and classes need type hints and a docstring.
* Add an entry under "Unreleased" in [CHANGELOG.md](CHANGELOG.md) for user-visible changes.
* Keep commits focused, with a short imperative subject ("Fix offsets after NFKC normalisation").

## Project layout

| path | what lives there |
|------|------------------|
| `keylex/keyword.py`, `keyword_list.py` | `Term`, `Keyword`, `KeywordList` and file IO |
| `keylex/matchers/` | exact token-trie engine, windowed fuzzy engine, option resolution, post-processing |
| `keylex/text.py` | tokenisation, Unicode normalisation with offset maps, stemming, sentence splitting |
| `keylex/rules/` | rule classes, the rule language parser and the `RuleEngine` |
| `keylex/extractors/` | YAKE, RAKE, TextRank and TF-IDF keyword extraction |
| `keylex/integrations/spacy.py` | the spaCy pipeline component |
| `keylex/cli.py` | the `keylex` command |
| `benchmarks/` | performance benchmarks |

## Releasing (maintainers)

1. Update `keylex/_version.py` and move the "Unreleased" changelog entries under the new version.
2. Commit, then tag and push: `git tag v0.1.1 && git push origin main v0.1.1`.
3. The `Release` workflow builds, checks and publishes to PyPI (trusted publishing) and creates a
   GitHub release.
