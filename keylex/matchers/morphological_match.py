from .base import SpacyMatcher
from typing import List

class MorphologicalMatcher(SpacyMatcher):
    def __init__(self, keywords: List[str], spacy_model: str):
        super().__init__(keywords, spacy_model)

    def match(self, text: str) -> List[dict]:
        matches = []
        doc = self.nlp(text)

        lemmatized_keywords = [self.nlp(keyword).lemma_ for keyword in self.keywords]

        for token in doc:
            if token.lemma_.lower() in [keyword.lower() for keyword in lemmatized_keywords]:
                start_pos = token.idx
                end_pos = token.idx + len(token.text)
                matches.append({
                    "keyword": token.text,
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "method": "morphological_match"
                })

        return matches