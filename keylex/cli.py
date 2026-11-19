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

Input is read as UTF-8 by default (a UTF-8/16/32 byte-order mark is honoured, so files written by
PowerShell's ``>`` work), from files or stdin; use ``--encoding`` for other encodings. Undecodable
bytes are replaced and reported on stderr. File arguments may be glob patterns (``docs/*.md``),
expanded by keylex itself when the shell doesn't (Windows). An input file given twice is processed
once (a note is printed on stderr). ``NO_COLOR`` disables colours.
"""

from __future__ import annotations

import argparse
import errno
import glob
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .extractors import EXTRACTORS, get_extractor
from .highlight import highlight
from .keyword_list import KeywordList, decode_bytes
from .matchers.enums import OutputFormat
from .matchers.fuzzy import SCORERS
from .matchers.matcher import KeywordMatcher
from .rules.dsl import RuleSyntaxError
from .rules.engine import RuleEngine
from .stopwords import STOPWORDS_BY_LANGUAGE

EXIT_OK, EXIT_NONE, EXIT_ERROR = 0, 1, 2
DEFAULT_ENCODING = "utf-8-sig"
_GLOB_CHARS = "*?["


class CLIError(Exception):
    """A usage / input problem: reported on stderr, exit code 2."""


def _warn(message: str) -> None:
    print(f"keylex: {message}", file=sys.stderr)


def _decode(data: bytes, encoding: str, name: str) -> str:
    """Decode input bytes; with the default encoding, UTF-8/16/32 byte-order marks are honoured."""
    try:
        if encoding == DEFAULT_ENCODING:
            return decode_bytes(data, "utf-8")
        return data.decode(encoding)
    except UnicodeDecodeError as exc:
        _warn(
            f"warning: {name}: not valid {encoding} ({exc.reason} at byte {exc.start}); "
            "undecodable bytes were replaced (use --encoding)"
        )
        if encoding == DEFAULT_ENCODING:
            return decode_bytes(data, "utf-8", errors="replace")
        return data.decode(encoding, errors="replace")


def _stdin_is_tty() -> bool:
    """True when stdin is an interactive terminal (reading it would wait for typing)."""
    try:
        if not sys.stdin.isatty():
            return False
        if sys.platform == "win32":  # NUL is a character device too: ask for a console
            import ctypes
            import msvcrt

            handle = msvcrt.get_osfhandle(sys.stdin.fileno())
            mode = ctypes.c_uint32()
            return bool(ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)))
        return True
    except (AttributeError, ValueError, OSError):
        return False


def _read_stdin(encoding: str) -> str:
    buffer = getattr(sys.stdin, "buffer", None)
    if buffer is None:  # already text (e.g. replaced by a StringIO)
        return sys.stdin.read().removeprefix("\N{ZERO WIDTH NO-BREAK SPACE}")
    return _decode(buffer.read(), encoding, "<stdin>")


def _expand(names: Sequence[str]) -> list[str]:
    """Expand glob patterns the shell left alone (Windows); a pattern matching no file is an error."""
    out: list[str] = []
    for name in names:
        if name != "-" and any(c in name for c in _GLOB_CHARS) and not os.path.exists(name):
            hits = [h for h in sorted(glob.glob(name, recursive=True)) if os.path.isfile(h)]
            if not hits:
                raise CLIError(f"no files match {name}")
            out.extend(hits)
        else:
            out.append(name)
    return out


def _inputs(files: Sequence[str], encoding: str = DEFAULT_ENCODING) -> list[tuple[str, str]]:
    """``(name, text)`` per input; ``-`` or no files = stdin. Duplicates are read once."""
    try:
        "".encode(encoding)
    except LookupError:
        raise CLIError(f"unknown encoding: {encoding}") from None
    if not files:
        if _stdin_is_tty():
            raise CLIError("no input files given and stdin is a terminal; pass files, pipe text in, or use '-'")
        return [("<stdin>", _read_stdin(encoding))]
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name in _expand(files):
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
    p.add_argument("files", nargs="*", help="text files or glob patterns (default: stdin; '-' = stdin)")
    p.add_argument(
        "--encoding",
        default=DEFAULT_ENCODING,
        help=f"encoding of the input files and stdin (default {DEFAULT_ENCODING}; BOMs are honoured)",
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


def _load_keyword_file(path: str) -> KeywordList:
    try:
        return KeywordList.load(path)
    except UnicodeDecodeError as exc:
        raise CLIError(f"{path}: not valid {exc.encoding} ({exc.reason} at byte {exc.start})") from None
    except (ValueError, TypeError, KeyError) as exc:
        message = str(exc)
        raise CLIError(message if path in message else f"{path}: {message}") from None


def _keyword_files(args: argparse.Namespace) -> list[str]:
    return _expand(args.keywords or [])


def _load_keywords(args: argparse.Namespace) -> KeywordList:
    kl = KeywordList()
    for path in _keyword_files(args):
        kl.merge(_load_keyword_file(path))
    for word in args.word or []:
        kl.add(word)
    return kl


def _use_color(args: argparse.Namespace) -> bool:
    return not args.no_color and not os.environ.get("NO_COLOR") and sys.stdout.isatty()


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
            ansi = fmt == "text" and _use_color(args)
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
            print(result, end="" if result.endswith("\n") else "\n")
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
            for item in result:  # detailed: one JSON object per line
                print(json.dumps(item, ensure_ascii=False) if isinstance(item, dict) else item)
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
    if args.save:
        parent = Path(args.save).parent
        if not parent.is_dir():
            raise CLIError(f"--save: directory does not exist: {parent}")
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
    for path in _keyword_files(args):
        engine.keyword_list.merge(_load_keyword_file(path))
    if args.check:
        diagnostics = engine.diagnostics()
        errors = [m for level, m in diagnostics if level == "error"]
        notes = [m for level, m in diagnostics if level != "error"]
        for message in errors:
            print(f"error: {message}")
        for message in notes:
            print(f"warning: {message}")
        failed = bool(errors) or (args.strict and bool(notes))
        if not failed:
            extra = f", {len(notes)} warning(s)" if notes else ""
            print(f"{args.rules}: ok ({len(engine.rules)} rules{extra})")
        return EXIT_NONE if failed else EXIT_OK
    if not engine.rules:
        raise CLIError(f"{args.rules}: no rules defined")
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


_OUTPUT_HELP = (
    "what to print: matches (line:col, text, keyword; default), detailed (one JSON object per match), "
    "keywords, boolean, count, categories, score, highlight (the text with matches marked), "
    "json (= --format json)"
)

_RULES_EPILOG = """\
rule language (keywords case-insensitive):
  bank, "interest rate", interest rate   the keyword occurs (bare words form one phrase)
  a AND b, a & b     a OR b, a | b       NOT a, !a, ~a       ( ... ) to group
  NEAR(a, b, 5)       within 5 words (default 5)     BEFORE(a, b, 5)  a then b
  ATLEAST(3, a)       at least 3 occurrences          ANY(2, a, b, c)  at least 2 of them
  ALL(a, b)  NONE(a, b)  SENTENCE(a, b)  PARAGRAPH(a, b)  SENTENCE(bank, NOT river)
  CATEGORY(finance)  TAG(hot)  KW(name)
  strings: "..." or '...' with JSON escapes ('it\\'s'); quote phrases containing and/or/not

rules file: {"keywords": [...], "rules": {"name": "RULE" | {"rule": ..., "weight": n,
"description": ...}}, "options": {matcher options}}

exit codes: 0 = a rule fired / check passed, 1 = no rule fired / check failed, 2 = error"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="keylex",
        description="Keyword matching, rules and extraction.",
        epilog="Exit codes: 0 = matched / fired / ok, 1 = nothing matched / no rule fired / check failed, "
        "2 = usage, I/O or input error. NO_COLOR disables colours.",
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
    pm.add_argument(
        "-o", "--output", default="matches", choices=[*OutputFormat.list(), "highlight", "json"], help=_OUTPUT_HELP
    )
    _format_args(pm)
    pm.add_argument("--all", action="store_true", help="with --output count: also list zero counts")
    pm.add_argument(
        "--no-color",
        action="store_true",
        help="no ANSI colours with --output highlight (also when NO_COLOR is set or output is not a terminal)",
    )
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
        epilog=_RULES_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    pr.add_argument(
        "rules", help='rules file: {"keywords": [...], "rules": {"name": "RULE" | {"rule": ..., "weight": n}}}'
    )
    _input_args(pr)
    pr.add_argument("-k", "--keywords", action="append", metavar="FILE", help="extra keyword file; repeatable")
    pr.add_argument("--explain", action="store_true", help="show why each rule did or didn't fire (also in JSON)")
    _format_args(pr)
    pr.add_argument(
        "--check",
        action="store_true",
        help="only validate the rules file; exit 1 if errors are found (warnings, e.g. keywords that will be "
        "added automatically, are printed but don't fail)",
    )
    pr.add_argument("--strict", action="store_true", help="with --check: warnings fail the check too")
    pr.add_argument("--min-score", type=float, default=None, metavar="X", help="exit 1 if any input scores below X")
    pr.add_argument("--fail-on", action="append", metavar="RULE", help="exit 1 if RULE fires on any input; repeatable")
    _matcher_args(pr)
    pr.set_defaults(func=cmd_rules)
    return parser


def _broken_stdout(exc: BaseException) -> bool:
    """A write to a closed pipe (``| head``); on Windows that surfaces as EINVAL on flush."""
    if isinstance(exc, BrokenPipeError):
        return True
    return (
        sys.platform == "win32"
        and isinstance(exc, OSError)
        and exc.errno == errno.EINVAL
        and getattr(exc, "filename", None) is None
    )


def _silence_stdout() -> None:
    """Point stdout at devnull so the interpreter's final flush doesn't fail again."""
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        os.close(devnull)
    except (OSError, ValueError, AttributeError):
        pass


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        code = int(args.func(args))
        sys.stdout.flush()
        return code
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        if _broken_stdout(exc):
            _silence_stdout()
            return EXIT_OK
        if isinstance(exc, (CLIError, RuleSyntaxError, ValueError, KeyError, TypeError, OSError, ImportError)):
            message = str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc)
        elif isinstance(exc, LookupError):
            message = str(exc)
        else:  # an unexpected failure: still a clean message and exit code 2, never a traceback
            message = f"{type(exc).__name__}: {exc}"
        print(f"keylex: error: {message}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
