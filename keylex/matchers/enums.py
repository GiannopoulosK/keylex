from enum import Enum


class OutputFormat(Enum):
    DETAILED = "detailed"
    KEYWORDS = "keywords"
    BOOLEAN = "boolean"
    COUNT = "count"

    @classmethod
    def list(cls):
        return [e.value for e in cls]


class MatchType(Enum):
    EXACT = "exact"
    FUZZY = "fuzzy"
