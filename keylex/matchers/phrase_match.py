from .base import SpacyMatcher
from typing import List


class PhraseMatcher(SpacyMatcher):
    def __init__(self, keywords: List[str], spacy_model: str):
        super().__init__(keywords, spacy_model)

    def match(self, text: str) -> List[dict]:
        matches = []
        doc = self.nlp(text)
        
        for keyword in self.keywords:
            keyword_tokens = keyword.split()
            keyword_len = len(keyword_tokens)

            for i in range(len(doc) - keyword_len + 1):
                phrase_tokens = [token.text.lower() for token in doc[i:i + keyword_len]]
                if phrase_tokens == [token.lower() for token in keyword_tokens]:
                    start_pos = doc[i].idx
                    end_pos = doc[i + keyword_len - 1].idx + len(doc[i + keyword_len - 1].text)
                    matches.append({
                        "keyword": keyword,
                        "start_pos": start_pos,
                        "end_pos": end_pos,
                        "method": "phrase_match"
                    })
        return matches