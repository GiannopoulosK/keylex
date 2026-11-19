"""Command-line interface: ``keylex match | extract | rules``.

Examples::

    keylex match -k keywords.json article.txt
    keylex match -w python -w rust --fuzzy --output count notes/*.md
    echo "Machine lerning rocks" | keylex match -w "machine learning" --fuzzy
    keylex extract article.txt --method rake --top 15 --save keywords.json
    keylex rules rules.json article.txt --explain
    keylex rules rules.json docs/*.md --format jsonl --fail-on secret_leak
    keylex rules --check rules.json

Exit codes (all subcommands): ``0`` something matched / a rule fired / the check passed,
``1`` nothing matched / no rule fired / a ``--min-score`` / ``--fail-on`` / ``--check`` failure,
``2`` usage, I/O or input errors.

Input is read as UTF-8 by default (a byte-order mark is stripped), from files or stdin; use
``--encoding`` for other encodings. Undecodable bytes are replaced and reported on stderr.
An input file given twice is processed once (a note is printed on stderr).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .extractors import EXTRACTORS, get_extractor
from .highlight import highlight
from .keyword_list import KeywordList
from .matchers.enums import OutputFormat
from .matchers.fuzzy import SCORERS
from .matchers.matcher import KeywordMatcher
from .rules.engine import RuleEngine
from .stopwords import STOPWORDS_BY_LANGUAGE

EXIT_OK, EXIT_NONE, EXIT_ERROR = 0, 1, 2
DEFAULT_ENCODING = "utf-8-sig"


class CLIError(Exception):
    """A usage / input problem: reported on stderr, exit code 2."""


def _warn(message: str) -> None:
    print(f"keylex: {message}", file=sys.stderr)


def _decode(data: bytes, encoding: str, name: str) -> str:
    try:
        return data.decode(encoding)
    except UnicodeDecodeError as exc:
        _warn(
            f"warning: {name}: not valid {encoding} ({exc.reason} at byte {exc.start}); "
            "undecodable bytes were replaced (use --encoding)"
        )
        return data.decode(encoding, errors="replace")


def _read_stdin(encoding: str) -> str:
    buffer = getattr(sys.stdin, "buffer", None)
    if buffer is None:  # already text (e.g. replaced by a StringIO)
        return sys.stdin.read().removeprefix("﻿")
    return _decode(buffer.read(), encoding, "<stdin>")


def _inputs(files: Sequence[str], encoding: str = DEFAULT_ENCODING) -> list[tuple[str, str]]:
    """``(name, text)`` per input; ``-`` or no files = stdin. Duplicates are read once."""
    try:
        "".encode(encoding)
    except LookupError:
        raise CLIError(f"unknown encoding: {encoding}") from None
    if not files:
        return [("<stdin>", _read_stdin(encoding))]
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name in files:
        key = name if name == "-" else os.path.normcase(os.path.abspath(name))
        if key in seen:
            _warn(f"note: {name} given more than once; processing it once")
            continue
        seen.add(key)
        if name == "-":
            out.append(("<stdin>", _read_stdin(encoding)))
            continue
        path = Path(name)
        if not path.exists():
            raise CLIError(f"file not found: {name}")
        if path.is_dir():
            raise CLIError(f"is a directory: {name}")
        out.append((name, _decode(path.read_bytes(), encoding, name)))
    return out


def _input_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("files", nargs="*", help="text files (default: stdin; '-' = stdin)")
    p.add_argument(
        "--encoding",
        default=DEFAULT_ENCODING,
        help=f"encoding of the input files and stdin (default {DEFAULT_ENCODING})",
    )


def _format_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--format",
        choices=["text", "json", "jsonl"],
        default=None,
        help="output format; jsonl = one JSON object per input file with a stable shape",
    )
    p.add_argument("--json", action="store_true", help="shorthand for --format json")


def _output_format(args: argparse.Namespace) -> str:
    if args.json and args.format not in (None, "json"):
        raise CLIError(f"--json conflicts with --format {args.format}")
    return "json" if args.json else (args.format or "text")


def _dump(data: Any, *, lines: bool = False) -> None:
    print(json.dumps(data, ensure_ascii=False) if lines else json.dumps(data, indent=2, ensure_ascii=False))


def _matcher_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("matching options (boolean flags have --no-... forms)")
    bool_flag = argparse.BooleanOptionalAction
    g.add_argument("--fuzzy", action=bool_flag, default=None, help="also match typos / near spellings")
    g.add_argument("--threshold", type=float, default=None, help="fuzzy similarity threshold 0-100 (default 85)")
    g.add_argument("--scorer", default=None, choices=sorted(SCORERS), help="fuzzy scorer (default ratio)")
    g.add_argument("--overlaps", default=None, choices=["longest", "leftmost", "all"])
    g.add_argument(
        "--case-sensitive", action=bool_flag, default=None, help="case-sensitive by default (acronyms always are)"
    )
    g.add_argument("--ignore-accents", action=bool_flag, default=None, help="treat 'cafe' and 'café' as equal")
    g.add_argument("--stem", action=bool_flag, default=None, help="match plurals (light English stemmer)")


def _matcher_options(args: argparse.Namespace) -> dict[str, Any]:
    """Only the options the user actually passed, so rule-file ``options`` aren't overridden."""
    given: dict[str, Any] = {}
    if args.fuzzy is not None:
        given["match_type"] = "fuzzy" if args.fuzzy else "exact"
    if args.threshold is not None:
        given["threshold"] = args.threshold
    if args.scorer is not None:
        given["scorer"] = args.scorer
    if args.overlaps is not None:
        given["overlaps"] = args.overlaps
    if args.case_sensitive is not None:
        given["case_sensitive"] = args.case_sensitive
    if args.ignore_accents is not None:
        given["ignore_accents"] = args.ignore_accents
    if args.stem is not None:
        given["stemmer"] = "light" if args.stem else None
    return given


def _load_keywords(args: argparse.Namespace) -> KeywordList:
    kl = KeywordList()
    for path in args.keywords or []:
        kl.merge(KeywordList.load(path))
    for word in args.word or []:
        kl.add(word)
    return kl


def cmd_match(args: argparse.Namespace) -> int:
    kl = _load_keywords(args)
    if not len(kl):
        raise CLIError("match: give keywords with -k FILE and/or -w WORD")
    matcher = KeywordMatcher(kl, **_matcher_options(args))
    fmt = _output_format(args)
    if args.output == "json" and fmt == "text":
        fmt = "json"
    inputs = _inputs(args.files, args.encoding)
    multi = len(inputs) > 1
    out: dict[str, Any] = {}
    found_any = False
    for name, text in inputs:
        matches = matcher.find(text)
        found_any = found_any or bool(matches)
        result: Any
        if args.output == "highlight":
            ansi = fmt == "text" and sys.stdout.isatty() and not args.no_color
            result = highlight(text, matches, "ansi" if ansi else "brackets")
        elif args.output in ("json", "matches") and fmt != "text":
            result = matcher.format_output(matches, "detailed")
        else:
            result = matcher.format_output(matches, "detailed" if args.output == "json" else args.output)
        if fmt == "jsonl":
            _dump({"file": name, "matched": bool(matches), "result": result}, lines=True)
            continue
        if fmt == "json":
            out[name] = result
            continue
        if multi:
            print(f"== {name}")
        if args.output == "highlight":
            print(result)
        elif args.output == "matches":
            for m in matches:
                line, col = text.count("\n", 0, m.start) + 1, m.start - (text.rfind("\n", 0, m.start) + 1) + 1
                extra = f"  ({m.kind}: {m.term})" if m.term != m.keyword or m.kind != "keyword" else ""
                fuzz = f"  [fuzzy {m.score:.0f}]" if m.method == "fuzzy" else ""
                print(f"{line}:{col}\t{m.text!r} -> {m.keyword}{extra}{fuzz}")
        elif isinstance(result, dict):
            width = max((len(str(k)) for k in result), default=0)
            for k, v in result.items():
                if args.output == "count" and not v and not args.all:
                    continue
                print(f"{k!s:<{width}}  {v}")
        elif isinstance(result, list):
            print("\n".join(map(str, result)))
        else:
            print(result)
    if fmt == "json":
        _dump(out if multi else next(iter(out.values()), []))
    return EXIT_OK if found_any else EXIT_NONE


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return number


def cmd_extract(args: argparse.Namespace) -> int:
    kwargs: dict[str, Any] = {"top_n": args.top, "max_ngram": args.max_ngram}
    if args.language is not None:
        kwargs["language"] = args.language
    if args.diversity is not None:
        kwargs["diversity"] = args.diversity
    extractor = get_extractor(args.method, **kwargs)
    fmt = _output_format(args)
    inputs = _inputs(args.files, args.encoding)
    multi = len(inputs) > 1
    all_results: dict[str, list[dict[str, object]]] = {}
    combined = KeywordList()
    found_any = False
    for name, text in inputs:
        results = extractor.extract(text)
        found_any = found_any or bool(results)
        all_results[name] = [r.to_dict() for r in results]
        for r in results:
            combined.add(r.phrase)
        if fmt == "jsonl":
            _dump({"file": name, "keywords": all_results[name]}, lines=True)
        elif fmt == "text":
            if multi:
                print(f"== {name}")
            width = max((len(r.phrase) for r in results), default=0)
            for r in results:
                print(f"{r.phrase:<{width}}  {r.score:.3f}  x{r.count}")
    if fmt == "json":
        _dump(all_results if multi else next(iter(all_results.values()), []))
    if args.save:
        combined.save(args.save)
        _warn(f"saved {len(combined)} keywords to {args.save}")
    return EXIT_OK if found_any else EXIT_NONE


def cmd_rules(args: argparse.Namespace) -> int:
    engine = RuleEngine.load(args.rules, **_matcher_options(args))
    for path in args.keywords or []:
        engine.keyword_list.merge(KeywordList.load(path))
    if args.check:
        problems = engine.validate()
        for problem in problems:
            print(problem)
        if not problems:
            print(f"{args.rules}: ok ({len(engine.rules)} rules)")
        return EXIT_NONE if problems else EXIT_OK
    unknown = [r for r in args.fail_on or [] if r not in engine.rules]
    if unknown:
        raise CLIError(f"--fail-on: no rule named {', '.join(map(repr, unknown))} in {args.rules}")
    fmt = _output_format(args)
    inputs = _inputs(args.files, args.encoding)
    multi = len(inputs) > 1
    out: dict[str, Any] = {}
    fired_any = False
    failures: list[str] = []
    for name, text in inputs:
        result = engine.evaluate(text)
        fired_any = fired_any or bool(result.fired)
        if args.min_score is not None and result.score < args.min_score:
            failures.append(f"{name}: score {result.score:g} < --min-score {args.min_score:g}")
        failures.extend(f"{name}: --fail-on rule {r!r} fired" for r in args.fail_on or [] if r in result)
        record = result.to_dict()
        if args.explain:
            record["explanation"] = result.explain()
        if fmt == "jsonl":
            _dump({"file": name, **record}, lines=True)
            continue
        if fmt == "json":
            out[name] = record
            continue
        if multi:
            print(f"== {name}")
        print(f"fired: {', '.join(result.fired) or '(none)'}   score: {result.score:g}")
        if args.explain:
            print(result.explain())
    if fmt == "json":
        _dump(out if multi else next(iter(out.values()), {}))
    if args.min_score is None and not args.fail_on:
        return EXIT_OK if fired_any else EXIT_NONE
    for failure in failures:
        _warn(failure)
    return EXIT_NONE if failures else EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="keylex",
        description="Keyword matching, rules and extraction.",
        epilog="Exit codes: 0 = matched / fired / ok, 1 = nothing matched / no rule fired / check failed, "
        "2 = usage, I/O or input error.",
    )
    parser.add_argument("--version", action="version", version=f"keylex {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    pm = sub.add_parser(
        "match",
        help="find keywords in text",
        description="Find keywords in files (or stdin). Exit code 1 if nothing matched, 2 on errors.",
    )
    _input_args(pm)
    pm.add_argument(
        "-k", "--keywords", action="append", metavar="FILE", help="keyword file (.json/.csv/.txt/.yaml); repeatable"
    )
    pm.add_argument("-w", "--word", action="append", metavar="WORD", help="a keyword given inline; repeatable")
    pm.add_argument("-o", "--output", default="matches", choices=[*OutputFormat.list(), "highlight", "json"])
    _format_args(pm)
    pm.add_argument("--all", action="store_true", help="with --output count: also list zero counts")
    pm.add_argument("--no-color", action="store_true")
    _matcher_args(pm)
    pm.set_defaults(func=cmd_match)

    pe = sub.add_parser(
        "extract",
        help="extract keywords without a keyword list",
        description="Extract keywords. Exit code 1 if no keyword was found, 2 on errors.",
    )
    _input_args(pe)
    pe.add_argument("-m", "--method", default="yake", choices=sorted(EXTRACTORS))
    pe.add_argument("-n", "--top", type=_positive_int, default=10, help="number of keywords (default 10)")
    pe.add_argument("--max-ngram", type=_positive_int, default=3, help="longest phrase in words (default 3)")
    pe.add_argument("--language", choices=sorted(STOPWORDS_BY_LANGUAGE), help="stopword language (default en)")
    pe.add_argument("--diversity", type=float, help="0 = no redundancy filter, 1 = maximal diversity")
    _format_args(pe)
    pe.add_argument("--save", metavar="FILE", help="save the keywords as a keyword file (.json/.csv)")
    pe.set_defaults(func=cmd_extract)

    pr = sub.add_parser(
        "rules",
        help="evaluate a rules file",
        description="Evaluate rules. Exit code 1 if no rule fired (or a --min-score / --fail-on / --check "
        "condition failed), 2 on errors.",
    )
    pr.add_argument(
        "rules", help='rules file: {"keywords": [...], "rules": {"name": "RULE" | {"rule": ..., "weight": n}}}'
    )
    _input_args(pr)
    pr.add_argument("-k", "--keywords", action="append", metavar="FILE", help="extra keyword file; repeatable")
    pr.add_argument("--explain", action="store_true", help="show why each rule did or didn't fire (also in JSON)")
    _format_args(pr)
    pr.add_argument("--check", action="store_true", help="only validate the rules file; exit 1 if problems are found")
    pr.add_argument("--min-score", type=float, default=None, metavar="X", help="exit 1 if any input scores below X")
    pr.add_argument("--fail-on", action="append", metavar="RULE", help="exit 1 if RULE fires on any input; repeatable")
    _matcher_args(pr)
    pr.set_defaults(func=cmd_rules)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (CLIError, ValueError, KeyError, TypeError, OSError, ImportError, LookupError) as exc:
        print(f"keylex: error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
