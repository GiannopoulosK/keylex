"""keylex -- keyword matching with keywords as objects.

>>> from keylex import Keyword, KeywordMatcher
>>> matcher = KeywordMatcher([Keyword("United States", acronyms="US"), "machine learning"])
>>> [m.keyword for m in matcher.find("Machine learning in the U.S.")]
['machine learning', 'United States']
"""

from ._version import __version__
from .extractors import ExtractedKeyword, extract_keywords, get_extractor
from .highlight import highlight
from .keyword import Keyword, Term
from .keyword_list import KeywordList
from .match import Match
from .matchers import (
    BaseMatcher,
    ExactMatcher,
    FuzzyMatcher,
    KeywordMatcher,
    MatchType,
    OutputFormat,
    Overlaps,
)
from .rules import RuleEngine, parse_rule

__all__ = [
    "BaseMatcher",
    "ExactMatcher",
    "ExtractedKeyword",
    "FuzzyMatcher",
    "Keyword",
    "KeywordList",
    "KeywordMatcher",
    "Match",
    "MatchType",
    "OutputFormat",
    "Overlaps",
    "RuleEngine",
    "Term",
    "__version__",
    "extract_keywords",
    "get_extractor",
    "highlight",
    "parse_rule",
]
