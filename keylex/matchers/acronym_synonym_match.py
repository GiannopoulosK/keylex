from .base import Matcher
from typing import List

class AcronymSynonymMatcher(Matcher):
    def __init__(self, keywords: List[str], acronym_dict: dict):
        super().__init__(keywords)
        self.acronym_dict = acronym_dict

    def match(self, text: str) -> List[dict]:
        matches = []
        for keyword in self.keywords:
            # Check if keyword matches directly
            if keyword.lower() in text.lower():
                start_idx = text.lower().find(keyword.lower())
                matches.append({
                    "keyword": keyword,
                    "start_pos": start_idx,
                    "end_pos": start_idx + len(keyword),
                    "method": "acronym_synonym_match"
                })

            # Check if acronym or synonym matches
            for acronym, full_form in self.acronym_dict.items():
                if keyword.lower() == full_form.lower() and acronym.lower() in text.lower():
                    start_idx = text.lower().find(acronym.lower())
                    matches.append({
                        "keyword": acronym,
                        "start_pos": start_idx,
                        "end_pos": start_idx + len(acronym),
                        "method": "acronym_synonym_match"
                    })
                elif keyword.lower() == acronym.lower() and full_form.lower() in text.lower():
                    start_idx = text.lower().find(full_form.lower())
                    matches.append({
                        "keyword": full_form,
                        "start_pos": start_idx,
                        "end_pos": start_idx + len(full_form),
                        "method": "acronym_synonym_match"
                    })

        return matches