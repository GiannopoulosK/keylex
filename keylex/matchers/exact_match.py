from .base import Matcher
from typing import List

class ExactMatcher(Matcher):
    def match(self, text: str, output: str) -> List[dict]:
        matches = []
        for keyword in self.keywords:
            start_idx = text.find(keyword)
            if start_idx != -1:
                matches.append({
                    "keyword": keyword,
                    "start_pos": start_idx,
                    "end_pos": start_idx + len(keyword),
                    "method": "exact_match"
                })
        return self.format_output(matches, output)