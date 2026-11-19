r"""A small, readable rule language.

Grammar (keywords are case-insensitive)::

    expr     := or
    or       := and (("OR" | "|") and)*
    and      := not (("AND" | "&") not)*
    not      := ("NOT" | "!" | "~") not | atom
    atom     := "(" expr ")" | phrase | FUNC "(" args ")"
    phrase   := STRING | WORD+            # consecutive bare words form one phrase

    NEAR(a, b[, n])       a and b within n words (default 5)
    BEFORE(a, b[, n])     a before b, within n words
    ATLEAST(n, expr)      expr occurs at least n times (see keylex.rules.core for "occurrences")
    ANY(n, e1, e2, ...)   at least n of the expressions match
    ALL(e1, e2, ...)      all match            NONE(e1, ...)  none match
    SENTENCE(e1, e2, ...) all in one sentence  PARAGRAPH(...) all in one paragraph
                          (a NOT operand means "not in that sentence": SENTENCE(bank, NOT river))
    CATEGORY(name)        any keyword of a category (case-insensitive)
    TAG(name)             any keyword carrying a tag (case-insensitive)
    KW(name)              explicit keyword reference

NOT operands are rejected inside NEAR, BEFORE and ATLEAST. Strings use double or single quotes with
JSON escapes (``\"``, ``\'``, ``\\``, ``\n``, ``\uXXXX``); bare words may contain apostrophes
(``don't``, ``O'Brien``). Every malformed rule raises :class:`RuleSyntaxError`.

Examples::

    "interest rate" AND NOT NEAR(bank, river, 5)
    ATLEAST(2, inflation) OR SENTENCE("central bank", "rate hike")
    ANY(2, python, rust, go) & !CATEGORY(sports)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .core import (
    AllOf,
    AnyOf,
    AtLeast,
    CategoryRule,
    KeywordRule,
    Near,
    Not,
    Rule,
    SameParagraph,
    SameSentence,
    TagRule,
)

__all__ = ["RuleSyntaxError", "parse_rule"]

_WS_RE = re.compile(r"\s+")
# A bare word may contain apostrophes (don't, rock'n'roll, O'Brien) but cannot start with one.
_WORD_RE = re.compile(r"""[^\s(),&|!~"'][^\s(),&|!~"]*""")
_OPS = "(),&|!~"
_ESCAPES = {'"': '"', "'": "'", "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
_INT_RE = re.compile(r"[0-9]+")
_FUNCS = {"NEAR", "BEFORE", "ATLEAST", "ANY", "ALL", "NONE", "SENTENCE", "PARAGRAPH", "CATEGORY", "TAG", "KW"}


class RuleSyntaxError(ValueError):
    """Raised for malformed rule strings; the message points at the offending position.

    Attributes:
        message: The bare error message (without position / pointer).
        source: The rule string.
        pos: Character offset of the problem in ``source``.
        rule_name: Name of the rule being loaded, when known (prefixed to the message).
    """

    def __init__(self, message: str, source: str, pos: int, rule_name: str | None = None) -> None:
        self.message, self.source, self.pos, self.rule_name = message, source, pos, rule_name
        prefix = f"rule {rule_name!r}: " if rule_name else ""
        one_line = re.sub(r"[\r\n\t]", " ", source)
        pointer = f"\n  {one_line}\n  {' ' * pos}^"
        super().__init__(f"{prefix}{message} (at position {pos}){pointer}")

    def with_rule_name(self, name: str) -> RuleSyntaxError:
        """A copy of this error whose message names the rule it came from."""
        return RuleSyntaxError(self.message, self.source, self.pos, name)

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (self.message, self.source, self.pos, self.rule_name))


@dataclass(frozen=True)
class _Tok:
    kind: str  # string | op | word | eof
    value: str
    pos: int


def _read_string(source: str, start: int) -> tuple[str, int]:
    """Decode the quoted string starting at ``source[start]``; returns (value, end position)."""
    quote = source[start]
    out: list[str] = []
    i = start + 1
    while i < len(source):
        ch = source[i]
        if ch == quote:
            return "".join(out), i + 1
        if ch in "\r\n":
            raise RuleSyntaxError("Line break inside a string (write \\n instead)", source, i)
        if ch == "\\":
            if i + 1 >= len(source):
                break
            esc = source[i + 1]
            if esc in _ESCAPES:
                out.append(_ESCAPES[esc])
                i += 2
                continue
            if esc == "u":
                digits = source[i + 2 : i + 6]
                if len(digits) == 4 and all(c in "0123456789abcdefABCDEF" for c in digits):
                    out.append(chr(int(digits, 16)))
                    i += 6
                    continue
                raise RuleSyntaxError("Invalid \\u escape (expected 4 hex digits)", source, i)
            raise RuleSyntaxError(f"Invalid escape sequence \\{esc} in string", source, i)
        out.append(ch)
        i += 1
    raise RuleSyntaxError("Unterminated string", source, start)


def _lex(source: str) -> list[_Tok]:
    toks: list[_Tok] = []
    pos = 0
    n = len(source)
    while pos < n:
        ch = source[pos]
        if ch.isspace():
            m = _WS_RE.match(source, pos)
            pos = m.end() if m else pos + 1
        elif ch in "\"'":
            value, end = _read_string(source, pos)
            toks.append(_Tok("string", value, pos))
            pos = end
        elif ch in _OPS:
            toks.append(_Tok("op", ch, pos))
            pos += 1
        else:
            m = _WORD_RE.match(source, pos)
            if m is None:  # pragma: no cover - every other character starts a word
                raise RuleSyntaxError(f"Unexpected character {ch!r}", source, pos)
            toks.append(_Tok("word", m.group(), pos))
            pos = m.end()
    toks.append(_Tok("eof", "", n))
    return toks


class _Parser:
    def __init__(self, source: str) -> None:
        self.source = source
        self.toks = _lex(source)
        self.i = 0

    @property
    def cur(self) -> _Tok:
        return self.toks[self.i]

    def error(self, msg: str, tok: _Tok | None = None) -> RuleSyntaxError:
        return RuleSyntaxError(msg, self.source, (tok or self.cur).pos)

    def advance(self) -> _Tok:
        tok = self.cur
        self.i += 1
        return tok

    def is_kw(self, *words: str) -> bool:
        return self.cur.kind == "word" and self.cur.value.upper() in words

    def is_op(self, *ops: str) -> bool:
        return self.cur.kind == "op" and self.cur.value in ops

    def expect_op(self, op: str) -> None:
        if not self.is_op(op):
            raise self.error(f"Expected {op!r}, found {self.cur.value or 'end of rule'!r}")
        self.advance()

    def parse(self) -> Rule:
        if self.cur.kind == "eof":
            raise self.error("Empty rule")
        rule = self.parse_or()
        if self.cur.kind != "eof":
            raise self.error(f"Unexpected {self.cur.value!r}")
        return rule

    def parse_or(self) -> Rule:
        rules = [self.parse_and()]
        while self.is_kw("OR") or self.is_op("|"):
            self.advance()
            rules.append(self.parse_and())
        return rules[0] if len(rules) == 1 else AnyOf(*rules)

    def parse_and(self) -> Rule:
        rules = [self.parse_not()]
        while self.is_kw("AND") or self.is_op("&"):
            self.advance()
            rules.append(self.parse_not())
        return rules[0] if len(rules) == 1 else AllOf(*rules)

    def parse_not(self) -> Rule:
        if self.is_kw("NOT") or self.is_op("!", "~"):
            self.advance()
            return Not(self.parse_not())
        return self.parse_atom()

    def parse_int(self) -> int:
        tok = self.cur
        if tok.kind != "word" or not _INT_RE.fullmatch(tok.value):
            raise self.error(f"Expected a number, found {tok.value or 'end of rule'!r}")
        self.advance()
        return int(tok.value)

    def parse_args(self) -> list[Rule | int]:
        self.expect_op("(")
        args: list[Rule | int] = []
        while True:
            if self.cur.kind == "word" and _INT_RE.fullmatch(self.cur.value):
                args.append(self.parse_int())
            else:
                args.append(self.parse_or())
            if self.is_op(","):
                self.advance()
                continue
            self.expect_op(")")
            return args

    def parse_name_arg(self) -> str:
        self.expect_op("(")
        tok = self.cur
        if tok.kind == "string":
            self.advance()
            name = tok.value
        else:
            words = []
            while self.cur.kind == "word":
                words.append(self.advance().value)
            if not words:
                raise self.error("Expected a name")
            name = " ".join(words)
        self.expect_op(")")
        return name

    def parse_atom(self) -> Rule:
        tok = self.cur
        if self.is_op("("):
            self.advance()
            rule = self.parse_or()
            self.expect_op(")")
            return rule
        if tok.kind == "string":
            self.advance()
            return KeywordRule(tok.value)
        if tok.kind == "word":
            upper = tok.value.upper()
            nxt = self.toks[self.i + 1]
            if upper in _FUNCS and nxt.kind == "op" and nxt.value == "(":
                self.advance()
                return self.parse_func(upper, tok)
            if upper in ("AND", "OR", "NOT"):
                raise self.error(f"Expected a keyword, found operator {tok.value!r}")
            words = []
            while self.cur.kind == "word" and self.cur.value.upper() not in ("AND", "OR", "NOT"):
                nxt = self.toks[self.i + 1]
                if self.cur.value.upper() in _FUNCS and nxt.kind == "op" and nxt.value == "(":
                    break
                words.append(self.advance().value)
            return KeywordRule(" ".join(words))
        raise self.error(f"Expected a keyword, found {tok.value or 'end of rule'!r}")

    def parse_func(self, name: str, tok: _Tok) -> Rule:
        if name in ("CATEGORY", "TAG", "KW"):
            arg = self.parse_name_arg()
            if name == "CATEGORY":
                return CategoryRule(arg)
            return TagRule(arg) if name == "TAG" else KeywordRule(arg)
        args = self.parse_args()
        rules = [a for a in args if isinstance(a, Rule)]
        ints = [a for a in args if isinstance(a, int)]
        try:
            if name in ("NEAR", "BEFORE"):
                if len(rules) != 2 or len(ints) > 1 or not isinstance(args[0], Rule) or not isinstance(args[1], Rule):
                    raise ValueError(f"{name}(a, b[, n]) takes two expressions and an optional number")
                return Near(rules[0], rules[1], ints[0] if ints else 5, ordered=name == "BEFORE")
            if name == "ATLEAST":
                if len(args) != 2 or not isinstance(args[0], int) or not isinstance(args[1], Rule):
                    raise ValueError("ATLEAST(n, expr) takes a number and an expression")
                return AtLeast(args[1], args[0])
            if name == "ANY":
                n = 1
                if args and isinstance(args[0], int):
                    n = args[0]
                if len(ints) > (1 if args and isinstance(args[0], int) else 0) or not rules:
                    raise ValueError("ANY([n,] e1, e2, ...) takes an optional leading number and expressions")
                return AnyOf(*rules, min_count=n)
            if ints or not rules:
                raise ValueError(f"{name}(...) takes expressions only")
            if name == "ALL":
                return AllOf(*rules)
            if name == "NONE":
                return Not(AnyOf(*rules))
            if name == "SENTENCE":
                return SameSentence(*rules)
            return SameParagraph(*rules)
        except (ValueError, TypeError) as exc:
            raise RuleSyntaxError(str(exc), self.source, tok.pos) from None


def parse_rule(source: str) -> Rule:
    """Parse a rule string into a :class:`Rule`. ``str(rule.to_dsl())`` round-trips."""
    if not isinstance(source, str):
        raise TypeError("rule source must be a string")
    return _Parser(source).parse()
