from .base import Matcher
from typing import List

class ProximityMatcher(Matcher):
    def __init__(self, keywords: List[str], proximity_distance: int = 5):
        super().__init__(keywords)
        self.proximity_distance = proximity_distance

    def match(self, text: str, output: str) -> List[dict]:
        matches = []
        words = text.split()
        keyword_positions = {}

        for i, word in enumerate(words):
            for keyword in self.keywords:
                if keyword.lower() == word.lower():
                    if keyword not in keyword_positions:
                        keyword_positions[keyword] = []
                    keyword_positions[keyword].append(i)

        for keyword, positions in keyword_positions.items():
            for i in range(len(positions) - 1):
                if positions[i + 1] - positions[i] <= self.proximity_distance:
                    start_pos = positions[i]
                    end_pos = positions[i + 1]
                    matches.append({
                        "keyword": keyword,
                        "start_pos": start_pos,
                        "end_pos": end_pos,
                        "method": "proximity_match"
                    })

        return self.format_output(matches, output)