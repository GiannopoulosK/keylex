from .base import Matcher
from typing import List
from rapidfuzz import fuzz, process

class FuzzyMatcher(Matcher):
    def __init__(self, keywords: List[str], threshold: float = 80.0):
        """
        Initialize the FuzzyMatcher with keywords and an optional threshold for fuzzy matching.
        :param keywords: List of keywords to match in the text.
        :param threshold: Minimum similarity threshold (0-100) for a match to be considered valid.
        """
        super().__init__(keywords)
        self.threshold = threshold

    def match(self, text: str) -> List[dict]:
        matches = []
        for keyword in self.keywords:
            similarity_score = fuzz.partial_ratio(text, keyword)
            if similarity_score >= self.threshold:
                start_idx = text.lower().find(keyword.lower())
                matches.append({
                    "keyword": keyword,
                    "start_pos": start_idx,
                    "end_pos": start_idx + len(keyword),
                    "method": "fuzzy_match",
                    "similarity_score": similarity_score
                })
        return matches