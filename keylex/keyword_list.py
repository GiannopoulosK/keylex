"""The :class:`KeywordList` collection."""

from __future__ import annotations

import codecs
import contextlib
import csv
import io
import json
import os
import re
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, Literal, overload

from .keyword import _MUTATIONS, Keyword, _clean

__all__ = ["KeywordList", "OnDuplicate"]

OnDuplicate = Literal["merge", "skip", "replace", "error"]
_ON_DUPLICATE = ("merge", "skip", "replace", "error")
FORMAT_VERSION = 1

KeywordLike = Keyword | str | Mapping[str, Any]

_FORMATS = (".json", ".csv", ".yaml", ".yml", ".txt")
#: A ``#`` comment starts a line or follows whitespace (so ``C#`` is not a comment).
_TEXT_COMMENT_RE = re.compile(r"(?:^|(?<=\s))#")
_BOMS = (
    (codecs.BOM_UTF32_LE, "utf-32"),  # before UTF-16 LE, whose BOM is a prefix of it
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)


def decode_bytes(data: bytes, encoding: str = "utf-8", errors: str = "strict") -> str:
    """Decode ``data``, honouring a UTF-8/16/32 byte-order mark if there is one."""
    for bom, codec in _BOMS:
        if data.startswith(bom):
            return data.decode(codec, errors)
    return data.decode(encoding, errors)


def read_text_file(path: str | Path, encoding: str = "utf-8") -> str:
    """Read a text file, detecting BOMs; decode errors name the file."""
    path = Path(path)
    try:
        return decode_bytes(path.read_bytes(), encoding)
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: not valid {exc.encoding} text ({exc.reason} at byte {exc.start})") from None


def write_text_atomic(path: str | Path, content: str) -> None:
    """Write UTF-8 text via a temporary file + rename, so a failure never truncates ``path``."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(content)
        # mkstemp creates the file private (0600); keep the existing file's mode, else the usual 0644
        os.chmod(tmp, path.stat().st_mode & 0o777 if path.exists() else 0o644)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _yaml() -> Any:
    try:
        import yaml
    except ImportError:  # pragma: no cover - optional dependency
        raise ImportError("YAML support requires PyYAML: pip install 'keylex[yaml]'") from None
    return yaml


_CSV_LIST_COLS = ("synonyms", "acronyms", "variants", "patterns", "exclude", "require", "tags")
_CSV_COLS = (
    "keyword",
    *_CSV_LIST_COLS[:-1],
    "category",
    "tags",
    "weight",
    "case_sensitive",
    "fuzzy",
    "context_window",
    "keyword_term",
    "metadata",
)


def _split_multi(value: str | None) -> list[Any]:
    """A CSV list cell: ``a|b`` or ``a;b``, or a JSON array (used when items contain separators
    or carry per-term options)."""
    if not value:
        return []
    if value.startswith("["):
        try:
            parsed = json.loads(value)
        except ValueError:
            pass
        else:
            if isinstance(parsed, list):
                return parsed
    sep = "|" if "|" in value else ";"
    return [v.strip() for v in value.split(sep) if v.strip()]


def _join_multi(items: list[Any]) -> str:
    if not items:
        return ""
    if all(
        isinstance(i, str) and "|" not in i and ";" not in i and not i.startswith("[") and i == i.strip() for i in items
    ):
        return "|".join(items)
    return json.dumps(items, ensure_ascii=False)


def _parse_bool(value: str, column: str) -> bool:
    low = value.strip().lower()
    if low in ("true", "1", "yes", "y"):
        return True
    if low in ("false", "0", "no", "n"):
        return False
    raise ValueError(f"CSV column {column!r}: expected true/false, got {value!r}")


class KeywordList:
    """An ordered collection of :class:`Keyword` objects with unique names.

    Example::

        kl = KeywordList(["python", "rust"])
        kl.add(Keyword("artificial intelligence", acronyms="AI"))
        kl.save("keywords.json"); KeywordList.load("keywords.json")

    Args:
        keywords: Keywords, plain strings, or keyword dicts.
        on_duplicate: What to do when a keyword with an existing name is added:
            ``"merge"`` (default, combine terms), ``"skip"``, ``"replace"`` or ``"error"``.
    """

    __hash__ = None  # type: ignore[assignment]

    def __init__(
        self,
        keywords: Iterable[KeywordLike] | None = None,
        *,
        on_duplicate: OnDuplicate = "merge",
    ) -> None:
        if on_duplicate not in _ON_DUPLICATE:
            raise ValueError(f"on_duplicate must be one of {_ON_DUPLICATE}")
        self.on_duplicate: OnDuplicate = on_duplicate
        self._items: dict[str, Keyword] = {}
        self._version = 0
        if keywords is not None:
            if isinstance(keywords, (str, Keyword)):
                keywords = [keywords]
            self.extend(keywords)

    # ------------------------------------------------------------------ construction
    @staticmethod
    def _coerce(item: KeywordLike) -> Keyword:
        if isinstance(item, Keyword):
            return item
        if isinstance(item, str):
            return Keyword(item)
        if isinstance(item, Mapping):
            return Keyword.from_dict(item)
        raise TypeError(f"Expected Keyword, str or dict, got {type(item).__name__}: {item!r}")

    @classmethod
    def from_list(cls, keywords: Iterable[KeywordLike], **kwargs: Any) -> KeywordList:
        """Build from strings, :class:`Keyword` objects and/or keyword dicts."""
        return cls(keywords, **kwargs)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], **kwargs: Any) -> KeywordList:
        """Build from a dict. Accepts every format keylex produces or documents:

        * ``to_dict()`` output: ``{"AI": {"synonyms": [...], "acronyms": [...], ...}}``
        * shorthand: ``{"AI": ["machine intelligence", {"acronyms": ["AI"]}]}``
        * simple: ``{"AI": "machine intelligence"}`` / ``{"AI": None}``
        * versioned JSON document: ``{"version": 1, "keywords": [...]}``
        """
        if "version" in data and isinstance(data.get("keywords"), list):
            if set(data) - {"version", "keywords"}:
                raise ValueError(f"Unknown key(s) in keyword document: {sorted(set(data) - {'version', 'keywords'})}")
            if data["version"] != FORMAT_VERSION:
                raise ValueError(f"Unsupported keyword document version {data['version']!r}")
            return cls(data["keywords"], **kwargs)
        out = cls(**kwargs)
        for name, spec in data.items():
            if not isinstance(name, str):
                raise TypeError(
                    f"Keyword names must be strings, got {name!r} ({type(name).__name__}); "
                    "in YAML, quote names such as 'yes', 'null' or '1984'"
                )
            if spec is None:
                out.add(Keyword(name))
            elif isinstance(spec, str):
                out.add(Keyword(name, synonyms=[spec]))
            elif isinstance(spec, Mapping):
                out.add(Keyword.from_dict({"keyword": name, **spec}))
            elif isinstance(spec, Iterable):
                kw = Keyword(name)
                for term in spec:
                    if isinstance(term, str):
                        kw.add_synonyms(term)
                    elif isinstance(term, Mapping):
                        for type_, terms in term.items():
                            if type_ not in ("synonyms", "acronyms", "variants", "patterns"):
                                raise ValueError(f"Unknown type {type_!r} in related terms of {name!r}")
                            getattr(kw, f"add_{type_}")(terms)
                    else:
                        raise TypeError(f"Invalid term format for {name!r}: {type(term).__name__}")
                out.add(kw)
            else:
                raise TypeError(f"Invalid value for keyword {name!r}: {type(spec).__name__}")
        return out

    @classmethod
    def from_json(cls, data: str, **kwargs: Any) -> KeywordList:
        parsed = json.loads(data)
        if isinstance(parsed, list):
            return cls(parsed, **kwargs)
        if isinstance(parsed, dict):
            return cls.from_dict(parsed, **kwargs)
        raise ValueError("JSON must contain an object or an array")

    @classmethod
    def from_csv(cls, data: str, **kwargs: Any) -> KeywordList:
        """Parse CSV with a ``keyword`` column and optional ``synonyms``, ``acronyms``,
        ``variants``, ``patterns``, ``exclude``, ``require``, ``tags`` (``;`` or ``|`` separated,
        or a JSON array), ``category``, ``weight``, ``case_sensitive``, ``fuzzy`` and
        ``context_window`` columns. Output of :meth:`to_csv` round-trips losslessly."""
        reader = csv.DictReader(io.StringIO(data))
        if not reader.fieldnames or "keyword" not in [f.strip().lower() for f in reader.fieldnames]:
            raise ValueError("CSV needs a 'keyword' column")
        out = cls(**kwargs)
        for raw in reader:
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items()}
            if not row.get("keyword"):
                continue
            spec: dict[str, Any] = {"keyword": row["keyword"]}
            for col in _CSV_LIST_COLS:
                if row.get(col):
                    spec[col] = _split_multi(row[col])
            if row.get("category"):
                spec["category"] = row["category"]
            if row.get("weight"):
                spec["weight"] = float(row["weight"])
            for col in ("case_sensitive", "fuzzy"):
                if row.get(col):
                    spec[col] = _parse_bool(row[col], col)
            if row.get("context_window"):
                spec["context_window"] = int(row["context_window"])
            for col in ("keyword_term", "metadata"):
                if row.get(col):
                    spec[col] = json.loads(row[col])
            out.add(Keyword.from_dict(spec))
        return out

    @classmethod
    def from_text(cls, data: str, **kwargs: Any) -> KeywordList:
        """One keyword per line; optional synonyms after ``|``.

        ``#`` starts a comment at the start of a line or after whitespace, so ``C#`` and ``F#`` are
        keywords, not comments. Example line: ``artificial intelligence | machine intelligence | AI``
        """
        out = cls(**kwargs)
        for line in data.splitlines():
            line = _TEXT_COMMENT_RE.split(line, 1)[0].strip()
            if not line:
                continue
            name, *syns = [p.strip() for p in line.split("|")]
            out.add(Keyword(name, synonyms=[s for s in syns if s]))
        return out

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> KeywordList:
        """Load from ``.json``, ``.csv``, ``.yaml``/``.yml`` (needs PyYAML) or ``.txt``.

        Files are read as UTF-8; UTF-8/16/32 byte-order marks are detected automatically.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix not in _FORMATS:
            raise ValueError(f"Unsupported keyword file type {path.suffix!r}; use one of {', '.join(_FORMATS)}")
        content = read_text_file(path)
        try:
            if suffix == ".json":
                return cls.from_json(content, **kwargs)
            if suffix == ".csv":
                return cls.from_csv(content, **kwargs)
            if suffix in (".yaml", ".yml"):
                yaml = _yaml()
                try:
                    parsed = yaml.safe_load(content)
                except yaml.YAMLError as exc:
                    raise ValueError(f"invalid YAML: {exc}") from exc
                if parsed is None:
                    return cls(**kwargs)
                if isinstance(parsed, list):
                    return cls(parsed, **kwargs)
                if isinstance(parsed, Mapping):
                    return cls.from_dict(parsed, **kwargs)
                raise ValueError("YAML must contain a mapping or a list")
            return cls.from_text(content, **kwargs)
        except json.JSONDecodeError as exc:
            detail = "invalid JSON in CSV field" if suffix == ".csv" else "invalid JSON"
            raise ValueError(f"{path}: {detail}: {exc}") from exc
        except TypeError as exc:
            raise TypeError(f"{path}: {exc}") from exc
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc

    # ------------------------------------------------------------------ serialisation
    def to_dict(self) -> dict[str, dict[str, Any]]:
        """``{name: {"synonyms": [...], "acronyms": [...], ...}}`` -- round-trips via :meth:`from_dict`."""
        out: dict[str, dict[str, Any]] = {}
        for kw in self:
            data = kw.to_dict()
            data.pop("keyword")
            out[kw.name] = data
        return out

    def to_list(self) -> list[dict[str, Any]]:
        """A list of keyword dicts (each includes the ``"keyword"`` key)."""
        return [kw.to_dict() for kw in self]

    def to_json(self, indent: int | None = 2) -> str:
        """Versioned JSON document; read back with :meth:`from_json`."""
        return json.dumps({"version": FORMAT_VERSION, "keywords": self.to_list()}, indent=indent, ensure_ascii=False)

    def to_csv(self) -> str:
        """CSV with one row per keyword; round-trips through :meth:`from_csv` without loss."""
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(_CSV_COLS), lineterminator="\n")
        writer.writeheader()
        for kw in self:
            data = kw.to_dict()
            row: dict[str, Any] = {"keyword": kw.name, "category": kw.category or ""}
            for col in _CSV_LIST_COLS:
                row[col] = _join_multi(list(data.get(col, [])))
            row["weight"] = "" if kw.weight == 1.0 else kw.weight
            for col in ("case_sensitive", "fuzzy"):
                value = data.get(col)
                row[col] = "" if value is None else str(value).lower()
            row["context_window"] = data.get("context_window", "")
            for col in ("keyword_term", "metadata"):
                row[col] = json.dumps(data[col], ensure_ascii=False) if col in data else ""
            writer.writerow(row)
        return buf.getvalue()

    def to_yaml(self) -> str:
        """YAML document (a list of keyword dicts); needs PyYAML. Round-trips via :meth:`load`."""
        return str(_yaml().safe_dump(self.to_list(), allow_unicode=True, sort_keys=False))

    def to_text(self) -> str:
        """The plain-text format: ``keyword | synonym | synonym``, one keyword per line.

        The format only holds names and synonyms. Raises ``ValueError`` if a keyword has anything
        else (acronyms, options, ...) that would be lost; use JSON, CSV or YAML for those.
        """
        lines = []
        for kw in self:
            lossy = kw.to_dict()
            lossy.pop("keyword")
            synonyms = lossy.pop("synonyms", [])
            if lossy.get("acronyms") == []:
                lossy.pop("acronyms")
            if lossy or not all(isinstance(s, str) for s in synonyms):
                raise ValueError(
                    f"Keyword {kw.name!r} has data the .txt format can't hold ({', '.join(lossy) or 'term options'}); "
                    "save as .json, .csv or .yaml instead"
                )
            for item in (kw.name, *synonyms):
                if "|" in item or _TEXT_COMMENT_RE.search(item) or item.startswith("#"):
                    raise ValueError(f"{item!r} can't be written to the .txt format ('|' or ' #'); use .json or .csv")
            lines.append(" | ".join((kw.name, *synonyms)))
        return "\n".join(lines) + ("\n" if lines else "")

    def save(self, path: str | Path) -> None:
        """Save as ``.json``, ``.csv``, ``.yaml``/``.yml`` or ``.txt`` depending on the file extension.

        The file is written atomically: a failure never leaves a truncated file behind.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        writers: dict[str, Callable[[], str]] = {
            ".json": self.to_json,
            ".csv": self.to_csv,
            ".yaml": self.to_yaml,
            ".yml": self.to_yaml,
            ".txt": self.to_text,
        }
        if suffix not in writers:
            raise ValueError(f"Unsupported keyword file type {path.suffix!r}; use one of {', '.join(_FORMATS)}")
        write_text_atomic(path, writers[suffix]())

    # ------------------------------------------------------------------ mutation
    def _touch(self) -> None:
        self._version += 1
        _MUTATIONS[0] += 1

    def add(self, keyword: KeywordLike, *, on_duplicate: OnDuplicate | None = None) -> Keyword:
        """Add a keyword (object, string or dict). Returns the keyword stored in the list."""
        kw = self._coerce(keyword)
        policy = on_duplicate or self.on_duplicate
        if policy not in _ON_DUPLICATE:
            raise ValueError(f"on_duplicate must be one of {_ON_DUPLICATE}")
        existing = self._items.get(kw.name)
        if existing is None:
            self._items[kw.name] = kw
        elif existing is kw or policy == "skip":
            return existing
        elif policy == "error":
            raise ValueError(f"Keyword {kw.name!r} already exists")
        elif policy == "replace":
            self._items[kw.name] = kw
        else:  # merge
            for term in kw.iter_terms("synonym", "acronym", "variant", "pattern"):
                existing.add_terms(term.kind, term)
            existing.exclude = existing.exclude + kw.exclude
            existing.require = existing.require + kw.require
            existing.tags = existing.tags + kw.tags
            if existing.category is None and kw.category is not None:
                existing.category = kw.category
            existing.metadata = {**kw.metadata, **existing.metadata}
            return existing
        self._touch()
        return self._items[kw.name]

    def add_keyword(self, keyword: KeywordLike, **kwargs: Any) -> Keyword:
        """Alias of :meth:`add`."""
        return self.add(keyword, **kwargs)

    def extend(self, keywords: Iterable[KeywordLike]) -> None:
        for kw in keywords:
            self.add(kw)

    def remove(self, keyword: Keyword | str) -> Keyword:
        """Remove by name (or by a keyword with that name). Raises ``KeyError`` if missing."""
        name = keyword.name if isinstance(keyword, Keyword) else keyword
        if name not in self._items:
            raise KeyError(f"Keyword {name!r} not found")
        self._touch()
        return self._items.pop(name)

    def discard(self, keyword: Keyword | str) -> None:
        """Remove if present; never raises."""
        name = keyword.name if isinstance(keyword, Keyword) else keyword
        if self._items.pop(name, None) is not None:
            self._touch()

    def remove_keyword(self, keyword: Keyword | str) -> None:
        """Remove if present (legacy alias of :meth:`discard`)."""
        self.discard(keyword)

    def rename(self, old: str, new: str) -> Keyword:
        """Rename a keyword in place, keeping its position and all of its terms."""
        new = _clean(new, "keyword")
        if old not in self._items:
            raise KeyError(f"Keyword {old!r} not found")
        if new != old and new in self._items:
            raise ValueError(f"Keyword {new!r} already exists")
        renamed = self._items[old].renamed(new)
        self._items = {(new if k == old else k): (renamed if k == old else v) for k, v in self._items.items()}
        self._touch()
        return renamed

    def clear(self) -> None:
        self._items.clear()
        self._touch()

    def merge(self, other: KeywordList | Iterable[KeywordLike]) -> KeywordList:
        """Merge another list into this one (copies, never shares objects). Returns ``self``."""
        for kw in other:
            self.add(self._coerce(kw).copy(), on_duplicate="merge")
        return self

    def copy(self) -> KeywordList:
        """Deep copy."""
        return KeywordList([kw.copy() for kw in self], on_duplicate=self.on_duplicate)

    # ------------------------------------------------------------------ queries
    @property
    def keywords(self) -> list[Keyword]:
        """The keywords as a list (a copy; mutate the list through its methods)."""
        return list(self._items.values())

    @property
    def names(self) -> list[str]:
        return list(self._items)

    def get(self, name: str, default: Keyword | None = None) -> Keyword | None:
        """Get by name; falls back to a case-insensitive lookup."""
        kw = self._items.get(name)
        if kw is not None:
            return kw
        folded = name.casefold()
        for key, value in self._items.items():
            if key.casefold() == folded:
                return value
        return default

    def get_keyword(self, keyword_str: str) -> Keyword | None:
        return self.get(keyword_str)

    def get_index(self, keyword_name: str, raise_error: bool = False) -> int | None:
        for index, name in enumerate(self._items):
            if name == keyword_name:
                return index
        if raise_error:
            raise ValueError(f"Keyword {keyword_name!r} not found.")
        return None

    def get_all_keywords(self) -> list[str]:
        return self.names

    def get_keywords_and_related_terms(self) -> dict[str, list[str]]:
        return {kw.name: kw.related_terms for kw in self}

    def find_term(self, text: str) -> list[Keyword]:
        """Keywords that have ``text`` as their name or one of their terms (case-insensitive)."""
        return [kw for kw in self if text in kw]

    def find_conflicts(self) -> dict[str, list[str]]:
        """Surface forms shared by more than one keyword, ``{form: [keyword names]}``.

        Names that differ only in case (``"Bank"`` / ``"bank"``) and synonyms reused across
        keywords both show up here. Such keywords all match the same text, which is fine when
        they are meant to be told apart by context rules, and usually a mistake otherwise.
        """
        owners: dict[str, list[str]] = {}
        for kw in self:
            seen: set[str] = set()
            for term in kw.iter_terms("keyword", "synonym", "acronym", "variant"):
                key = " ".join(term.text.casefold().split())
                if key not in seen:
                    seen.add(key)
                    owners.setdefault(key, []).append(kw.name)
        return {form: names for form, names in owners.items() if len(names) > 1}

    def categories(self) -> dict[str | None, list[Keyword]]:
        """Group keywords by category."""
        groups: dict[str | None, list[Keyword]] = {}
        for kw in self:
            groups.setdefault(kw.category, []).append(kw)
        return groups

    def filter(
        self,
        predicate: Callable[[Keyword], bool] | None = None,
        *,
        category: str | None = None,
        tag: str | None = None,
    ) -> KeywordList:
        """A new list (sharing the same keyword objects) with the keywords that pass all filters."""
        out = KeywordList(on_duplicate=self.on_duplicate)
        for kw in self:
            if category is not None and kw.category != category:
                continue
            if tag is not None and tag not in kw.tags:
                continue
            if predicate is not None and not predicate(kw):
                continue
            out._items[kw.name] = kw
        return out

    def signature(self) -> tuple[int, int, tuple[int, ...]]:
        """Changes whenever the list or any keyword in it changes (used for cache invalidation)."""
        # list() snapshots the dict in one step, so a concurrent add() can't break the iteration
        return (id(self), self._version, tuple(kw._version for kw in list(self._items.values())))

    # ------------------------------------------------------------------ dunder
    def __iter__(self) -> Iterator[Keyword]:
        return iter(list(self._items.values()))

    def __len__(self) -> int:
        return len(self._items)

    @overload
    def __getitem__(self, index: int | str) -> Keyword: ...
    @overload
    def __getitem__(self, index: slice) -> KeywordList: ...
    def __getitem__(self, index: int | str | slice) -> Keyword | KeywordList:
        if isinstance(index, str):
            kw = self.get(index)
            if kw is None:
                raise KeyError(index)
            return kw
        if isinstance(index, slice):
            out = KeywordList(on_duplicate=self.on_duplicate)
            for kw in list(self._items.values())[index]:
                out._items[kw.name] = kw
            return out
        return list(self._items.values())[index]

    def __contains__(self, item: object) -> bool:
        """``"AI" in kl`` checks names *and* terms, case-insensitively."""
        if isinstance(item, Keyword):
            return item.name in self._items
        if not isinstance(item, str):
            return False
        return item in self._items or any(item in kw for kw in self._items.values())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KeywordList):
            return NotImplemented
        if self._items.keys() != other._items.keys():
            return False
        return all(self._items[k] == other._items[k] for k in self._items)

    def __add__(self, other: KeywordList) -> KeywordList:
        return self.copy().merge(other)

    def __repr__(self) -> str:
        names = self.names
        shown = ", ".join(repr(n) for n in names[:8])
        more = f", ... (+{len(names) - 8})" if len(names) > 8 else ""
        return f"KeywordList([{shown}{more}])"
