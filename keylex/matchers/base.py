from abc import ABC, abstractmethod
from typing import List
from enum import Enum
import spacy

class OutputFormat(Enum):
    DETAILED = "detailed"
    KEYWORDS = "keywords"
    BOOLEAN = "boolean"
    COUNT = "count"

class Matcher(ABC):
    def __init__(self, keywords: List[str]):
        self.keywords = keywords

    @abstractmethod
    def match(self, text: str, output: str) -> List[dict]:
        pass
    
    def format_output(self, matches: List[dict], output: str):
        output_enum = OutputFormat(output)
        
        if output_enum == OutputFormat.KEYWORDS:
            return [match['keyword'] for match in matches]
        elif output_enum == OutputFormat.BOOLEAN:
            return [{match['keyword']: True} for match in matches] if matches else \
                [{keyword: False} for keyword in self.keywords]
        elif output_enum == OutputFormat.COUNT:
            count_dict = {keyword: 0 for keyword in self.keywords}
            for match in matches:
                count_dict[match['keyword']] += 1
            return [{keyword: count} for keyword, count in count_dict.items()]
        else:
            return matches

class SpacyMatcher(Matcher):
    def __init__(self, keywords: List[str], spacy_model: str):
        super().__init__(keywords)
        self.spacy_model = spacy_model
        self.nlp = spacy.load(self.spacy_model)
