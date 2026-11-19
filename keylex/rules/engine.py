"""Named, weighted rules evaluated together: classification, tagging and scoring."""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..keyword import Keyword
from ..keyword_list import KeywordList
from ..match import Match
from ..matchers.matcher import KeywordMatcher, KeywordsInput, as_keyword_list
from .core import AnyOf, CategoryRule, Document, Rule, RuleResult, TagRule, walk
from .dsl import RuleSyntaxError, parse_rule

__all__ = ["EngineResult", "NamedRule", "RuleEngine"]


def _check_weight(name: str, weight: Any) -> float:
    """Weights are finite numbers (negative weights act as penalties)."""
    if isinstance(weight, bool):
        raise ValueError(f"rule {name!r}: weight must be a number, got {weight!r}")
    try:
        value = float(weight)
    except (TypeError, ValueError):
        raise ValueError(f"rule {name!r}: weight must be a number, got {weight!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"rule {name!r}: weight must be finite, got {weight!r}")
    return value


def _fold(name: str) -> str:
    return " ".join(name.split()).casefold()


@dataclass(frozen=True)
class NamedRule:
    name: str
    rule: Rule
    weight: float = 1.0
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"rule": self.rule.to_dsl()}
        if self.weight != 1.0:
            data["weight"] = self.weight
        if self.description:
            data["description"] = self.description
        return data


@dataclass(frozen=True)
class EngineResult:
    """Result of :meth:`RuleEngine.evaluate`."""

    text: str
    matches: list[Match]
    results: dict[str, RuleResult]
    rules: dict[str, NamedRule] = field(repr=False)

    @property
    def fired(self) -> list[str]:
        """Names of the rules that matched, highest weight first."""
        names = [n for n, r in self.results.items() if r.matched]
        return sorted(names, key=lambda n: -self.rules[n].weight)

    @property
    def score(self) -> float:
        """Sum of the weights of the rules that matched."""
        return sum(self.rules[n].weight for n in self.fired)

    @property
    def top(self) -> str | None:
        """The highest-weight rule that matched (handy for single-label classification)."""
        fired = self.fired
        return fired[0] if fired else None

    def __getitem__(self, name: str) -> RuleResult:
        return self.results[name]

    def __contains__(self, name: object) -> bool:
        return name in self.results and self.results[name].matched

    def explain(self) -> str:
        """ASCII tree per rule (top node: the whole rule; nested nodes: short labels)."""
        return "\n".join(
            f"[{name}] weight={self.rules[name].weight:g}\n{res.explain(1)}" for name, res in self.results.items()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fired": self.fired,
            "score": self.score,
            "rules": {
                n: {"matched": r.matched, "keywords": r.keywords, "matches": [m.to_dict() for m in r.matches]}
                for n, r in self.results.items()
            },
        }


class RuleEngine:
    """Evaluate a set of named, weighted rules against texts.

    Keywords referenced by rules but missing from ``keywords`` are added automatically as plain
    keywords, so the quickest start is just::

        engine = RuleEngine(rules={
            "finance": 'bank AND NOT NEAR(bank, river, 5)',
            "ai": 'ANY("machine learning", "neural network", AI)',
        })
        engine.evaluate(text).fired

    A rule name that is not a keyword name but a synonym / acronym / variant of exactly one
    keyword (e.g. ``ML`` for "machine learning") refers to that keyword. If several keywords
    share the term, a warning is issued and the term is matched as a keyword of its own.

    Args:
        keywords: Keyword definitions (synonyms, acronyms, categories, ...). Optional.
        rules: ``{name: rule}`` where a rule is a :class:`Rule`, a rule string, or a dict
            ``{"rule": ..., "weight": 2.0, "description": "..."}``. Weights must be finite
            numbers; negative weights act as penalties in :attr:`EngineResult.score`.
        **matcher_options: Passed to :class:`KeywordMatcher` (``match_type``, ``threshold``,
            ``stemmer``, ``overlaps``, ...). Overlaps default to ``"all"`` here so that rules can
            see nested keywords (e.g. both "interest rate" and "rate").
    """

    def __init__(
        self,
        keywords: KeywordsInput | None = None,
        rules: Mapping[str, Rule | str | Mapping[str, Any]] | None = None,
        **matcher_options: Any,
    ) -> None:
        self.keyword_list = as_keyword_list(keywords) if keywords is not None else KeywordList()
        self._rules: dict[str, NamedRule] = {}
        matcher_options.setdefault("overlaps", "all")
        self.matcher_options = matcher_options
        self._combined = KeywordList()
        self._combined_sig: object = None
        self._aliases: dict[str, str] = {}
        self._matcher = KeywordMatcher(self._combined, **matcher_options)
        if rules is not None and not isinstance(rules, Mapping):
            raise ValueError(f"rules must be a mapping {{name: rule}}, got a {type(rules).__name__}")
        for name, spec in (rules or {}).items():
            if isinstance(spec, Mapping):
                if "rule" not in spec:
                    keys = ", ".join(map(repr, spec)) or "none"
                    raise ValueError(f"rule {name!r}: missing the 'rule' key (keys given: {keys})")
                self.add_rule(
                    name, spec["rule"], weight=spec.get("weight", 1.0), description=spec.get("description") or ""
                )
            else:
                self.add_rule(name, spec)

    # ------------------------------------------------------------------ rules
    def add_rule(self, name: str, rule: Rule | str, weight: float = 1.0, description: str = "") -> RuleEngine:
        """Add (or replace) a named rule. Returns ``self`` for chaining.

        Raises:
            RuleSyntaxError: the rule string is malformed (the message names the rule).
            ValueError: empty name or non-finite weight.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("rule name must be a non-empty string")
        try:
            parsed = parse_rule(rule) if isinstance(rule, str) else rule
        except RuleSyntaxError as exc:
            raise exc.with_rule_name(name) from None
        if not isinstance(parsed, Rule):
            raise TypeError(f"rule {name!r}: must be a Rule or a rule string, got {type(rule).__name__}")
        self._rules[name] = NamedRule(name, parsed, _check_weight(name, weight), str(description))
        self._combined_sig = None
        return self

    def remove_rule(self, name: str) -> None:
        del self._rules[name]
        self._combined_sig = None

    @property
    def rules(self) -> dict[str, NamedRule]:
        return dict(self._rules)

    def _needed(self) -> set[str]:
        return set().union(*(r.rule.keyword_names() for r in self._rules.values())) if self._rules else set()

    def _resolve_terms(self, names: Iterable[str]) -> tuple[dict[str, str], dict[str, list[str]]]:
        """Map rule keyword names that are a term of exactly one user keyword to that keyword."""
        aliases: dict[str, str] = {}
        ambiguous: dict[str, list[str]] = {}
        for name in names:
            if self.keyword_list.get(name) is not None:
                continue  # an exact (case-insensitive) name match wins
            owners = self.keyword_list.find_term(name)
            if len(owners) == 1:
                aliases[name] = owners[0].name
            elif len(owners) > 1:
                ambiguous[name] = [kw.name for kw in owners]
        return aliases, ambiguous

    @property
    def matcher(self) -> KeywordMatcher:
        """The underlying matcher (user keywords + keywords referenced by rules)."""
        needed = self._needed()
        sig = (self.keyword_list.signature(), frozenset(needed))
        if sig != self._combined_sig:
            combined = KeywordList()
            for kw in self.keyword_list:
                combined._items[kw.name] = kw  # share objects so later edits are seen
            aliases, ambiguous = self._resolve_terms(needed)
            for name in sorted(needed):
                if name in ambiguous:
                    warnings.warn(
                        f"rule keyword {name!r} is a term of several keywords ({', '.join(ambiguous[name])}); "
                        "matching it as a separate keyword",
                        stacklevel=3,
                    )
                if name not in aliases and combined.get(name) is None:
                    combined._items[name] = Keyword(name)
            combined._version = self._combined._version + 1
            self._combined = combined
            self._aliases = aliases
            self._matcher.keyword_list = combined
            self._combined_sig = sig
        return self._matcher

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, text: str) -> EngineResult:
        """Match ``text`` once and evaluate every rule (shared sub-rules are evaluated once)."""
        matches = self.matcher.find(text)
        doc = Document(text, matches, aliases=self._aliases)
        results = {name: doc.evaluate(nr.rule) for name, nr in self._rules.items()}
        return EngineResult(text, matches, results, dict(self._rules))

    def evaluate_many(self, texts: Iterable[str]) -> Iterator[EngineResult]:
        for text in texts:
            yield self.evaluate(text)

    def classify(self, text: str) -> list[str]:
        """Names of the rules that fire, highest weight first."""
        return self.evaluate(text).fired

    def score(self, text: str) -> float:
        return self.evaluate(text).score

    # ------------------------------------------------------------------ checks
    def validate(self) -> list[str]:
        """Static checks of the rule set; returns human-readable problems (empty list = clean).

        Reports rules naming keywords that are not in the keyword list (only when a keyword list
        was given: without one, auto-creating keywords is the intended use), names that are a
        term of several keywords, ``CATEGORY`` / ``TAG`` names that no keyword has, ``ANY(n, ...)``
        with fewer than ``n`` operands and non-finite weights.
        """
        problems: list[str] = []
        kl = self.keyword_list
        categories = {_fold(k.category) for k in kl if k.category}
        tags = {_fold(t) for k in kl for t in k.tags}
        for name, nr in self._rules.items():
            if not math.isfinite(nr.weight):
                problems.append(f"rule {name!r}: weight {nr.weight!r} is not a finite number")
            names = nr.rule.keyword_names()
            aliases, ambiguous = self._resolve_terms(names)
            for kw_name in sorted(names):
                if kw_name in ambiguous:
                    problems.append(
                        f"rule {name!r}: {kw_name!r} is a term of several keywords "
                        f"({', '.join(ambiguous[kw_name])}); it is matched as a separate keyword"
                    )
                elif len(kl) and kw_name not in aliases and kl.get(kw_name) is None:
                    problems.append(
                        f"rule {name!r}: keyword {kw_name!r} is not in the keyword list "
                        "(it will be added as a plain keyword)"
                    )
            seen: set[str] = set()
            for node in walk(nr.rule):
                key = node._key()
                if key in seen:
                    continue
                seen.add(key)
                if isinstance(node, CategoryRule) and _fold(node.name) not in categories:
                    problems.append(f"rule {name!r}: no keyword has category {node.name!r}, so {key} never matches")
                elif isinstance(node, TagRule) and _fold(node.name) not in tags:
                    problems.append(f"rule {name!r}: no keyword has tag {node.name!r}, so {key} never matches")
                elif isinstance(node, AnyOf) and node.min_count > len(node.rules):
                    problems.append(f"rule {name!r}: {node.label()} needs more operands than it has")
        return problems

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"rules": {n: r.to_dict() for n, r in self._rules.items()}}
        if len(self.keyword_list):
            data["keywords"] = self.keyword_list.to_list()
        opts = {
            k: v
            for k, v in self.matcher_options.items()
            if isinstance(v, (str, int, float, bool)) and not (k == "overlaps" and v == "all")
        }
        if opts:
            data["options"] = opts
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None, **matcher_options: Any) -> RuleEngine:
        """Inverse of :meth:`to_dict`: ``{"keywords": [...], "rules": {...}, "options": {...}}``.

        Raises:
            ValueError: a malformed structure (the message says what is wrong).
        """
        if data is None:
            raise ValueError('the rules file is empty; expected {"rules": {name: rule, ...}}')
        if not isinstance(data, Mapping):
            raise ValueError(f'expected an object with a "rules" mapping at the top level, got a {type(data).__name__}')
        file_options = data.get("options") or {}
        if not isinstance(file_options, Mapping):
            raise ValueError(f'"options" must be an object, got a {type(file_options).__name__}')
        rules = data.get("rules") or {}
        if not isinstance(rules, Mapping):
            raise ValueError(f'"rules" must be an object {{name: rule}}, got a {type(rules).__name__}')
        keywords = data.get("keywords")
        kl = None
        if keywords is not None:
            if isinstance(keywords, list):
                kl = KeywordList(keywords)
            elif isinstance(keywords, Mapping):
                kl = KeywordList.from_dict(keywords)
            else:
                raise ValueError(f'"keywords" must be a list or an object, got a {type(keywords).__name__}')
        return cls(kl, rules, **{**file_options, **matcher_options})

    @classmethod
    def load(cls, path: str | Path, **matcher_options: Any) -> RuleEngine:
        """Load a JSON (or YAML, with PyYAML) rules file (UTF-8; a byte-order mark is ignored).

        Raises:
            OSError: the file cannot be read.
            ValueError: malformed content (the message names the file); rule syntax errors are
                :class:`RuleSyntaxError` naming the rule.
            ImportError: a YAML file but PyYAML is not installed.
        """
        path = Path(path)
        content = path.read_text(encoding="utf-8-sig")
        data: Any
        if path.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    f"{path}: reading YAML rules files needs PyYAML (pip install 'keylex[yaml]')"
                ) from None
            try:
                data = yaml.safe_load(content)
            except yaml.YAMLError as exc:
                raise ValueError(f"{path}: invalid YAML: {exc}") from None
        else:
            try:
                data = json.loads(content) if content.strip() else None
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSON: {exc}") from None
        try:
            return cls.from_dict(data, **matcher_options)
        except RuleSyntaxError:
            raise
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from None

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    def __repr__(self) -> str:
        return f"RuleEngine({len(self._rules)} rules, {len(self.keyword_list)} keywords)"
