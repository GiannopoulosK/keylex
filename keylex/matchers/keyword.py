"""Backwards-compatible import path. Use ``from keylex import Keyword``."""

from ..keyword import Keyword, Term

__all__ = ["Keyword", "Term"]
