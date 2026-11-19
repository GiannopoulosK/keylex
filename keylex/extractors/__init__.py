"""Unsupervised keyword extraction: find the important keywords of a text without a keyword list.

Quick use::

    from keylex import extract_keywords
    extract_keywords(article, top_n=5)                 # YAKE by default
    extract_keywords(article, method="rake")

Every extractor returns :class:`ExtractedKeyword` objects and can hand its results straight to the
matching side via ``extractor.to_keyword_list(text)``.
"""

from __future__ import annotations

from typing import Any

from .algorithms import RakeExtractor, TextRankExtractor, TfidfExtractor, YakeExtractor
from .base import BaseExtractor, ExtractedKeyword

EXTRACTORS: dict[str, type[BaseExtractor]] = {
    "yake": YakeExtractor,
    "rake": RakeExtractor,
    "textrank": TextRankExtractor,
    "tfidf": TfidfExtractor,
}


def get_extractor(method: str = "yake", **kwargs: Any) -> BaseExtractor:
    """Instantiate an extractor by name (``"yake"``, ``"rake"``, ``"textrank"``, ``"tfidf"``)."""
    try:
        cls = EXTRACTORS[method.lower()]
    except KeyError:
        raise ValueError(f"Unknown extraction method {method!r}; choose from {sorted(EXTRACTORS)}") from None
    return cls(**kwargs)


def extract_keywords(text: str, top_n: int | None = 10, method: str = "yake", **kwargs: Any) -> list[ExtractedKeyword]:
    """Extract the ``top_n`` most important keywords/keyphrases from ``text``."""
    return get_extractor(method, top_n=top_n, **kwargs).extract(text)


__all__ = [
    "EXTRACTORS",
    "BaseExtractor",
    "ExtractedKeyword",
    "RakeExtractor",
    "TextRankExtractor",
    "TfidfExtractor",
    "YakeExtractor",
    "extract_keywords",
    "get_extractor",
]
