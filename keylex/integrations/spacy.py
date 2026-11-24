"""spaCy integration (``pip install keylex[spacy]``).

Add keylex as a pipeline component::

    import spacy
    import keylex.integrations.spacy  # registers the "keylex" factory (also auto-registered via entry point)

    nlp = spacy.blank("en")
    nlp.add_pipe("keylex", config={"keywords": keyword_list.to_list(), "set_ents": True})
    doc = nlp("Machine learning in the U.S.")
    doc.spans["keylex"]          # SpanGroup of matches, labelled by category (or keyword)
    doc._.keylex_matches         # all keylex Match objects (the matcher output)
    span._.keylex_matches        # every Match behind a span (identical spans of several keywords)
    span._.keylex_match          # the first of them

Matches are kept in ``doc.user_data`` as plain data, so processed docs survive ``doc.to_bytes()``,
``DocBin(store_user_data=True)`` and ``nlp.pipe(..., n_process=2)``; several keylex components add to
``doc._.keylex_matches``.

The component's keywords and options are saved with ``nlp.to_disk`` / ``nlp.to_bytes`` and restored
by ``spacy.load`` / ``nlp.from_bytes``, including keywords given via :func:`add_to_pipeline`.

Or convert keywords into EntityRuler token patterns with :func:`to_entity_ruler_patterns`.
"""

from __future__ import annotations

import json
import uuid
import warnings
import weakref
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

try:
    import spacy
    from spacy.language import Language
    from spacy.tokens import Doc, Span
    from spacy.util import filter_spans
except ImportError as exc:  # pragma: no cover - exercised only without spaCy
    raise ImportError("keylex's spaCy integration requires spaCy: pip install 'keylex[spacy]'") from exc

from ..keyword import Keyword
from ..keyword_list import KeywordList
from ..match import Match
from ..matchers.matcher import KeywordMatcher, KeywordsInput, as_keyword_list

__all__ = [
    "KeylexComponent",
    "add_to_pipeline",
    "make_keylex_component",
    "match_spans",
    "to_entity_ruler_patterns",
]

LABEL_MODES = ("category", "keyword", "fixed")
ALIGNMENT_MODES = ("strict", "contract", "expand")

#: ``doc.user_data`` keys. Matches are stored as plain rows (see :func:`_to_row`) so docs stay
#: serialisable (``doc.to_bytes()``, ``DocBin(store_user_data=True)``, ``nlp.pipe(n_process=...)``);
#: the extension getters rebuild :class:`Match` objects from them.
_MATCHES_KEY = ("keylex", "matches")
#: ``{"token_start:token_end": [row, ...]}`` index behind ``span._.keylex_matches`` (span extensions
#: are otherwise stored per character offsets, so two keywords matching the same span would overwrite
#: each other).
_INDEX_KEY = ("keylex", "span_matches")

#: Factory options that configure the matcher (everything else configures the component).
_MATCHER_KEYS = ("match_type", "threshold", "overlaps", "stemmer", "ignore_accents")
_COMPONENT_KEYS = ("spans_key", "set_ents", "overwrite_ents", "label", "default_label", "alignment_mode")

_Row = tuple[Any, ...]
_ROW_FIELDS = 9  # keyword, start, end, text, term, kind, method, score, category (+ component token)

#: Live components by token, so ``Match.keyword_obj`` can be restored for docs that went through
#: serialisation (e.g. ``nlp.pipe(n_process=2)``) while the component is still around.
_COMPONENTS: weakref.WeakValueDictionary[str, KeylexComponent] = weakref.WeakValueDictionary()
#: The original :class:`Match` objects (with ``keyword_obj``) of docs processed in this process.
_LIVE: weakref.WeakKeyDictionary[Doc, dict[_Row, Match]] = weakref.WeakKeyDictionary()


def _to_row(doc: Doc, m: Match, token: str | None = None) -> _Row:
    """A msgpack-friendly row for ``m``; the live object is remembered for :func:`_from_row`."""
    row: _Row = (m.keyword, m.start, m.end, m.text, m.term, m.kind, m.method, float(m.score), m.category, token)
    if m.keyword_obj is not None:
        live = _LIVE.get(doc)
        if live is None:
            live = _LIVE[doc] = {}
        live[tuple(row[:_ROW_FIELDS])] = m
    return row


def _from_row(doc: Doc, row: Sequence[Any]) -> Match:
    key = tuple(row[:_ROW_FIELDS])
    live = _LIVE.get(doc)
    if live is not None and key in live:
        return live[key]
    m = Match(*key)
    token = row[_ROW_FIELDS] if len(row) > _ROW_FIELDS else None
    component = _COMPONENTS.get(token) if isinstance(token, str) else None
    if component is not None:
        kw = component.keyword_list.get(m.keyword)
        if kw is not None:
            m = replace(m, keyword_obj=kw)
    return m


def _user_data(doc: Doc) -> dict[Any, Any]:
    data: dict[Any, Any] = doc.user_data
    return data


def _span_index(doc: Doc) -> dict[str, list[_Row]]:
    user_data = _user_data(doc)
    index = user_data.get(_INDEX_KEY)
    if index is None:
        index = user_data[_INDEX_KEY] = {}
    return index


def _span_key(span: Span) -> str:
    return f"{span.start}:{span.end}"


def _get_span_matches(span: Span) -> list[Match]:
    return [_from_row(span.doc, r) for r in _span_index(span.doc).get(_span_key(span), ())]


def _get_span_match(span: Span) -> Match | None:
    found = _get_span_matches(span)
    if not found:
        return None
    # identical spans with different labels: prefer the match the label came from
    label = span.label_
    return next((m for m in found if label and label in (m.category, m.keyword)), found[0])


def _set_span_match(span: Span, value: Match | None) -> None:
    index = _span_index(span.doc)
    key = _span_key(span)
    if value is None:
        index[key] = []
    else:  # the given match becomes the first; the others stay available via keylex_matches
        row = _to_row(span.doc, value)
        others = [r for r in index.get(key, ()) if tuple(r[:_ROW_FIELDS]) != row[:_ROW_FIELDS]]
        index[key] = [row, *others]


def _get_doc_matches(doc: Doc) -> list[Match] | None:
    rows = _user_data(doc).get(_MATCHES_KEY)
    if rows is None:
        return None
    return [_from_row(doc, r) for r in rows]


def _set_doc_matches(doc: Doc, value: Iterable[Match] | None) -> None:
    if value is None:
        _user_data(doc).pop(_MATCHES_KEY, None)
    else:
        _user_data(doc)[_MATCHES_KEY] = [_to_row(doc, m) for m in value]


def _add_doc_matches(doc: Doc, matches: Iterable[Match], token: str | None) -> None:
    """Append to ``doc._.keylex_matches`` (several keylex components accumulate)."""
    rows = list(_user_data(doc).get(_MATCHES_KEY) or ())
    seen = {tuple(r[:_ROW_FIELDS]) for r in rows}
    for m in matches:
        row = _to_row(doc, m, token)
        if row[:_ROW_FIELDS] not in seen:
            seen.add(row[:_ROW_FIELDS])
            rows.append(row)
    _user_data(doc)[_MATCHES_KEY] = rows


Doc.set_extension("keylex_matches", getter=_get_doc_matches, setter=_set_doc_matches, force=True)
Span.set_extension("keylex_match", getter=_get_span_match, setter=_set_span_match, force=True)
Span.set_extension("keylex_matches", getter=_get_span_matches, force=True)


def _label_for(m: Match, mode: str, fixed: str) -> str:
    if mode == "keyword":
        return m.keyword
    if mode == "fixed":
        return fixed
    return m.category or fixed


def _spans_for(
    doc: Doc,
    matches: Iterable[Match],
    label: str,
    default_label: str,
    alignment_mode: str,
    token: str | None = None,
) -> list[Span]:
    if label not in LABEL_MODES:
        raise ValueError(f"label must be one of {LABEL_MODES}")
    if alignment_mode not in ALIGNMENT_MODES:
        raise ValueError(f"alignment_mode must be one of {ALIGNMENT_MODES}")
    index = _span_index(doc)
    spans: list[Span] = []
    seen: set[tuple[int, int, str]] = set()
    for m in matches:
        lab = _label_for(m, label, default_label)
        span = doc.char_span(m.start, m.end, label=lab, alignment_mode=alignment_mode)
        if span is None or not len(span):
            continue  # not aligned to token boundaries
        key = _span_key(span)
        bucket = list(index.get(key, ()))
        row = _to_row(doc, m, token)
        if all(tuple(r[:_ROW_FIELDS]) != row[:_ROW_FIELDS] for r in bucket):
            index[key] = [*bucket, row]
        if (span.start, span.end, lab) not in seen:
            seen.add((span.start, span.end, lab))
            spans.append(span)
    return spans


def match_spans(
    doc: Doc,
    matcher: KeywordMatcher,
    label: str = "category",
    default_label: str = "KEYWORD",
    alignment_mode: str = "strict",
) -> list[Span]:
    """Run ``matcher`` over ``doc.text`` and return token-aligned :class:`Span` objects.

    ``label``: ``"category"`` (keyword category, falling back to ``default_label``), ``"keyword"``
    (the canonical keyword name) or ``"fixed"`` (always ``default_label``).

    ``alignment_mode``: ``"strict"`` (default) skips matches that don't start and end on token
    boundaries (e.g. a ``whole_word=False`` term inside a word); ``"contract"`` / ``"expand"``
    shrink / grow them to whole tokens (spaCy's ``Doc.char_span`` modes).

    Identical spans produced by several keywords appear once per label; ``span._.keylex_matches``
    lists every :class:`Match` behind a span.
    """
    return _spans_for(doc, matcher.find(doc.text), label, default_label, alignment_mode)


def _overlaps(a: Span, b: Span) -> bool:
    return bool(a.start < b.end and b.start < a.end)


class KeylexComponent:
    """spaCy pipeline component that stores keylex matches on each ``Doc``.

    Usually created with ``nlp.add_pipe("keylex", config={...})`` or :func:`add_to_pipeline`.

    Args:
        nlp: The pipeline the component belongs to.
        name: The component's name in the pipeline.
        matcher: The :class:`KeywordMatcher` run on each ``doc.text``.
        spans_key: ``doc.spans`` key the matches are stored under (``""`` to skip).
        set_ents: Also add the matches to ``doc.ents``.
        overwrite_ents: With ``set_ents``, ``False`` (default) keeps existing entities (e.g. from
            NER) and only adds keylex spans that don't overlap them; ``True`` lets keylex spans
            replace overlapping entities.
        label: Span label: ``"category"`` (falling back to ``default_label``), ``"keyword"`` or
            ``"fixed"`` (always ``default_label``).
        default_label: Label used by ``"fixed"`` and for keywords without a category.
        alignment_mode: See :func:`match_spans` (default ``"strict"``).
        matcher_options: Extra :class:`KeywordMatcher` keyword arguments (``case_sensitive``, ...).
    """

    def __init__(
        self,
        nlp: Language,
        name: str,
        matcher: KeywordMatcher,
        spans_key: str = "keylex",
        set_ents: bool = False,
        label: str = "category",
        default_label: str = "KEYWORD",
        *,
        overwrite_ents: bool = False,
        alignment_mode: str = "strict",
        matcher_options: dict[str, Any] | None = None,
    ) -> None:
        if label not in LABEL_MODES:
            raise ValueError(f"label must be one of {LABEL_MODES}")
        if alignment_mode not in ALIGNMENT_MODES:
            raise ValueError(f"alignment_mode must be one of {ALIGNMENT_MODES}")
        self.name = name
        self.matcher = matcher
        self.spans_key = spans_key
        self.set_ents = set_ents
        self.overwrite_ents = overwrite_ents
        self.label = label
        self.default_label = default_label
        self.alignment_mode = alignment_mode
        self.matcher_options: dict[str, Any] = dict(matcher_options or {})
        self._token = uuid.uuid4().hex
        _COMPONENTS[self._token] = self

    # -- processing ---------------------------------------------------------------------------------
    def __call__(self, doc: Doc) -> Doc:
        matches = self.matcher.find(doc.text)
        _add_doc_matches(doc, matches, self._token)
        spans = _spans_for(doc, matches, self.label, self.default_label, self.alignment_mode, self._token)
        if self.spans_key:
            doc.spans[self.spans_key] = spans
        if self.set_ents:
            new = filter_spans(spans)
            old = list(doc.ents)
            if self.overwrite_ents:
                kept = [e for e in old if not any(_overlaps(e, s) for s in new)]
                doc.ents = sorted([*kept, *new], key=lambda s: s.start)
            else:
                added = [s for s in new if not any(_overlaps(e, s) for e in old)]
                doc.ents = sorted([*old, *added], key=lambda s: s.start)
        return doc

    def pipe(self, docs: Iterable[Doc], batch_size: int = 128) -> Iterable[Doc]:
        """Process a stream of docs (called by ``nlp.pipe``)."""
        for doc in docs:
            yield self(doc)

    # -- keywords and serialisation ------------------------------------------------------------------
    @property
    def keyword_list(self) -> KeywordList:
        """The keywords the component matches."""
        return self.matcher.keyword_list

    def set_keywords(self, keywords: KeywordsInput, stemmer: str | Callable[[str], str] | None = None) -> None:
        """Replace the keywords (a live :class:`KeywordList` stays live)."""
        opts = dict(self.matcher_options)
        if stemmer is not None:
            opts["stemmer"] = stemmer
        self.matcher = KeywordMatcher(as_keyword_list(keywords), **opts)

    def _state(self) -> dict[str, Any]:
        opts = dict(self.matcher_options)
        stemmer = getattr(self.matcher, "stemmer", opts.get("stemmer"))
        if stemmer is not None and not isinstance(stemmer, str):
            warnings.warn(
                "keylex: a callable stemmer cannot be serialised; it is saved as None", UserWarning, stacklevel=3
            )
            stemmer = None
        opts["stemmer"] = stemmer
        return {
            "keywords": self.matcher.keyword_list.to_list(),
            "matcher_options": opts,
            "component": {key: getattr(self, key) for key in _COMPONENT_KEYS},
        }

    def _load_state(self, state: dict[str, Any]) -> None:
        for key, value in state.get("component", {}).items():
            if key in _COMPONENT_KEYS:
                setattr(self, key, value)
        self.matcher_options = dict(state.get("matcher_options", self.matcher_options))
        self.matcher = KeywordMatcher(KeywordList.from_list(state.get("keywords", [])), **self.matcher_options)

    def to_bytes(self, *, exclude: Sequence[str] = ()) -> bytes:
        """Serialise keywords and options (used by ``nlp.to_bytes``)."""
        return json.dumps(self._state(), ensure_ascii=False).encode("utf-8")

    def from_bytes(self, data: bytes, *, exclude: Sequence[str] = ()) -> KeylexComponent:
        """Restore keywords and options from :meth:`to_bytes` output (used by ``nlp.from_bytes``)."""
        self._load_state(json.loads(data.decode("utf-8")))
        return self

    def to_disk(self, path: str | Path, *, exclude: Sequence[str] = ()) -> None:
        """Save keywords and options to ``path/keylex.json`` (used by ``nlp.to_disk``)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "keylex.json").write_text(json.dumps(self._state(), ensure_ascii=False, indent=1), encoding="utf-8")

    def from_disk(self, path: str | Path, *, exclude: Sequence[str] = ()) -> KeylexComponent:
        """Load keywords and options saved by :meth:`to_disk` (used by ``spacy.load``)."""
        file = Path(path) / "keylex.json"
        if file.exists():
            self._load_state(json.loads(file.read_text(encoding="utf-8")))
        return self


@Language.factory(  # type: ignore[untyped-decorator,unused-ignore]  # untyped when spaCy is absent
    "keylex",
    default_config={
        "keywords": [],
        "match_type": "exact",
        "threshold": 85.0,
        "overlaps": "longest",
        "stemmer": None,
        "ignore_accents": False,
        "spans_key": "keylex",
        "set_ents": False,
        "overwrite_ents": False,
        "label": "category",
        "default_label": "KEYWORD",
        "alignment_mode": "strict",
        "matcher_options": {},
    },
)
def make_keylex_component(
    nlp: Language,
    name: str,
    keywords: Any,
    match_type: str,
    threshold: float,
    overlaps: str,
    stemmer: str | None,
    ignore_accents: bool,
    spans_key: str,
    set_ents: bool,
    overwrite_ents: bool,
    label: str,
    default_label: str,
    alignment_mode: str,
    matcher_options: dict[str, Any],
) -> KeylexComponent:
    """Factory: ``keywords`` may be a list of strings/keyword dicts, a dict, or a file path.

    ``matcher_options`` holds any further :class:`KeywordMatcher` arguments (``case_sensitive``,
    ``acronym_case``, ...).
    """
    if isinstance(keywords, (str, Path)) and Path(keywords).suffix.lower() in (
        ".json",
        ".csv",
        ".txt",
        ".yaml",
        ".yml",
    ):
        if Path(keywords).exists():
            kl = KeywordList.load(keywords)
        else:
            # ``spacy.load`` rebuilds the component from the saved config (which still names the file)
            # before ``from_disk`` restores the saved keywords, so a moved/deleted file must not fail here.
            warnings.warn(
                f"keylex: keyword file {str(keywords)!r} not found; the component starts without keywords "
                "(a pipeline loaded with spacy.load / from_disk restores its saved keywords)",
                UserWarning,
                stacklevel=2,
            )
            kl = KeywordList()
    else:
        kl = as_keyword_list(keywords)
    opts: dict[str, Any] = {
        "match_type": match_type,
        "threshold": threshold,
        "overlaps": overlaps,
        "stemmer": stemmer,
        "ignore_accents": ignore_accents,
        **dict(matcher_options or {}),
    }
    matcher = KeywordMatcher(kl, **opts)
    return KeylexComponent(
        nlp,
        name,
        matcher,
        spans_key,
        set_ents,
        label,
        default_label,
        overwrite_ents=overwrite_ents,
        alignment_mode=alignment_mode,
        matcher_options=opts,
    )


def add_to_pipeline(nlp: Language, keywords: KeywordsInput, **config: Any) -> KeylexComponent:
    """Convenience: ``add_to_pipeline(nlp, keyword_list, set_ents=True)``.

    Unlike ``nlp.add_pipe("keylex", config=...)`` this accepts a live :class:`KeywordList` (or any
    keyword input) directly. ``config`` keys are the factory's parameters plus ``name``, ``before``,
    ``after``, ``first``, ``last``; any other key (``case_sensitive=True``, a callable ``stemmer``...)
    is passed to the :class:`KeywordMatcher`. The keywords are saved with ``nlp.to_disk``.
    """
    kl = as_keyword_list(keywords)
    name = config.pop("name", "keylex")
    placement = {k: config.pop(k) for k in ("before", "after", "first", "last") if k in config}
    stemmer = config.get("stemmer")
    if callable(stemmer):
        config.pop("stemmer")
    factory_keys = {*_MATCHER_KEYS, *_COMPONENT_KEYS, "matcher_options"}
    extra = {k: config.pop(k) for k in list(config) if k not in factory_keys}
    if extra:
        config["matcher_options"] = {**config.get("matcher_options", {}), **extra}
    component = nlp.add_pipe("keylex", name=name, config={**config, "keywords": []}, **placement)
    assert isinstance(component, KeylexComponent)
    component.set_keywords(kl, stemmer=stemmer if callable(stemmer) else None)
    return component


def _term_case_sensitive(kw: Keyword, term: Any, default: bool) -> bool:
    """Same precedence as the matcher: Term > Keyword > acronym heuristic > default."""
    if term.case_sensitive is not None:
        return bool(term.case_sensitive)
    if kw.case_sensitive is not None:
        return bool(kw.case_sensitive)
    if term.is_acronym_like:
        return True
    return default


def to_entity_ruler_patterns(
    keywords: KeywordsInput,
    label: str | Callable[[Any], str] = "category",
    default_label: str = "KEYWORD",
    *,
    nlp: Language | None = None,
    case_sensitive: bool = False,
) -> list[dict[str, Any]]:
    """Token patterns for spaCy's ``EntityRuler`` (or ``SpanRuler``).

    Case-sensitive terms (acronyms such as ``NASA``, or ``case_sensitive=True`` on the term/keyword,
    else the ``case_sensitive`` default) use ``ORTH`` attributes, all others ``LOWER``, so the ruler
    agrees with :class:`KeywordMatcher` regardless of ``phrase_matcher_attr``; acronyms also get their
    dotted form ("US" -> spaCy's "U.S." token). ``label`` works as in :func:`match_spans` (any other
    string is used as the label itself). Terms are tokenised
    with ``nlp``'s tokenizer (default: ``spacy.blank("en")``). Pattern ``id`` is the canonical keyword
    name, so ``ent.ent_id_`` tells you which keyword hit. Regex pattern terms cannot be expressed
    and are skipped with a warning.
    """
    kl = as_keyword_list(keywords)
    tokenizer = (nlp if nlp is not None else spacy.blank("en")).tokenizer
    patterns: list[dict[str, Any]] = []
    skipped: list[str] = []
    for kw in kl:
        if callable(label):
            lab = label(kw)
        elif label == "keyword":
            lab = kw.name
        elif label == "category":
            lab = kw.category or default_label
        elif label == "fixed":
            lab = default_label
        else:
            lab = label
        skipped.extend(t.text for t in kw.iter_terms("pattern"))
        seen: set[str] = set()
        for term in kw.iter_terms("keyword", "synonym", "acronym", "variant"):
            cs = _term_case_sensitive(kw, term, case_sensitive)
            forms = [term.text]
            text = term.text
            if term.is_acronym_like and text.isalpha() and 2 <= len(text) <= 6:
                forms.append("".join(f"{ch}." for ch in text))  # "US" also matches "U.S." (like the matcher)
            for form in forms:
                tokens = [t.text for t in tokenizer(form) if not t.is_space]
                if not tokens:
                    continue
                pattern = [{"ORTH": t} if cs else {"LOWER": t.lower()} for t in tokens]
                ident = json.dumps(pattern, sort_keys=True)
                if ident in seen:
                    continue
                seen.add(ident)
                patterns.append({"label": lab, "pattern": pattern, "id": kw.name})
    if skipped:
        warnings.warn(
            f"to_entity_ruler_patterns: skipped {len(skipped)} regex pattern term(s) the EntityRuler cannot "
            f"express: {skipped[:5]}",
            UserWarning,
            stacklevel=2,
        )
    return patterns
