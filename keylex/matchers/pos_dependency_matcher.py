from .base import SpacyMatcher
from typing import List


class POSDependencyMatcher(SpacyMatcher):
    def __init__(self, keywords: List[str], spacy_model: str):
        super().__init__(keywords, spacy_model)

    def match(self, text: str) -> List[dict]:
        matches = []
        doc = self.nlp(text)
        for token in doc:
            for keyword in self.keywords:
                if keyword.lower() == token.text.lower():
                    matches.append({
                        "keyword": keyword,
                        "start_pos": token.idx,
                        "end_pos": token.idx + len(token.text),
                        "method": "pos_dependency_match",
                        "dep": token.dep_,
                        "pos": token.pos_
                    })
        return matches