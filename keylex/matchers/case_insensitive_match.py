from .base import Matcher
from typing import List

class CaseInsensitiveMatcher(Matcher):
    def match(self, text: str) -> List[dict]:
        matches = []
        for keyword in self.keywords:
            start_idx = text.lower().find(keyword.lower())
            if start_idx != -1:
                matches.append({
                    "keyword": keyword,
                    "start_pos": start_idx,
                    "end_pos": start_idx + len(keyword),
                    "method": "case_insensitive_match"
                })
        return matches