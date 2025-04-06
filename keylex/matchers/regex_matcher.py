import re
from typing import List
from .base import Matcher

class RegexMatcher(Matcher):
    def match(self, text: str) -> List[dict]:
        matches = []
        for keyword in self.keywords:
            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                matches.append({
                    "keyword": keyword,
                    "start_pos": match.start(),
                    "end_pos": match.end(),
                    "method": "regex_match"
                })
        return matches