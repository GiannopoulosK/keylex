from .base import SpacyMatcher
from typing import List


class NERMatcher(SpacyMatcher):
    def __init__(self, keywords: List[str], spacy_model: str):
        super().__init__(keywords, spacy_model)

    def match(self, text: str) -> List[dict]:
        matches = []
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.text.lower() in [keyword.lower() for keyword in self.keywords]:
                matches.append({
                    "keyword": ent.text,
                    "start_pos": ent.start_char,
                    "end_pos": ent.end_char,
                    "method": "ner_match"
                })
        return matches