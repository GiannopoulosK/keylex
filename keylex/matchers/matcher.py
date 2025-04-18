from abc import ABC, abstractmethod
from typing import List, Dict, Union
import logging
from .enums import OutputFormat
from rapidfuzz import fuzz
import re
from .keyword_list import KeywordList

logger = logging.getLogger(__name__)


class BaseMatcher(ABC):
    @abstractmethod
    def match(self, text: str, output: str) -> List[Dict]:
        pass


class Matcher(BaseMatcher):
    """
    Abstract base class for keyword matchers.

    Attributes:
        keyword_list (KeywordList): A KeywordList instance.
    """

    def __init__(self, keyword_list: KeywordList):
        if not isinstance(keyword_list, KeywordList):
            logger.error("`keyword_list` must be an instance of KeywordList.")
            raise ValueError("`keyword_list` must be an instance of KeywordList.")

        self.keyword_list = keyword_list

    @staticmethod
    def _validate_text(text: str) -> None:
        """Validates the input text."""
        if not isinstance(text, str):
            logger.error("`text` must be a string.")
            raise ValueError("`text` must be a string.")

    @staticmethod
    def _validate_output_format(output: str) -> None:
        """Validates the output format argument."""
        try:
            OutputFormat(output)
        except ValueError:
            valid_formats = ", ".join(OutputFormat.list())
            logger.error(
                f"Invalid output format: {output}. Supported formats: {valid_formats}"
            )
            raise ValueError(
                f"Invalid output format: {output}. Supported formats: {valid_formats}"
            )

    def _find_matches(self, text: str) -> List[Dict]:
        """Finds all matches for the keywords in the text."""
        raise NotImplementedError("Subclasses must implement `_find_matches`.")

    def format_output(
        self, matches: List[Dict], output: str
    ) -> Union[List[Dict], List[str], Dict[str, Union[bool, int]]]:
        keywords = [keyword.keyword for keyword in self.keyword_list.keywords]

        if output == OutputFormat.DETAILED.value:
            return matches

        elif output == OutputFormat.KEYWORDS.value:
            return list(set([match["keyword"] for match in matches]))

        elif output == OutputFormat.BOOLEAN.value:
            return {
                keyword: any(match["keyword"] == keyword for match in matches)
                for keyword in keywords
            }

        elif output == OutputFormat.COUNT.value:
            text_content = " ".join([match["keyword"] for match in matches]).lower()
            return {
                keyword: text_content.count(keyword.lower()) for keyword in keywords
            }

        else:
            raise ValueError(
                f"Unsupported output format: {output}. Supported formats: {', '.join([format_.value for format_ in OutputFormat])}"
            )

    def match(
        self, text: str, output: str = "detailed"
    ) -> Union[List[Dict], List[str], Dict[str, Union[bool, int]]]:
        """Matches the keywords in the provided text."""
        self._validate_text(text)
        self._validate_output_format(output)
        matches = self._find_matches(text)
        return self.format_output(matches=matches, output=output)


class KeywordMatcher(Matcher):
    def __init__(
        self,
        keyword_list: KeywordList,
        match_type: str = "exact",
        threshold: int = 80,
        fuzzy_algo: str = "partial_ratio",
    ):
        """
        Initializes the keyword matcher.

        Args:
            keyword_list (List[Keyword]): List of Keyword objects.
            match_type (str): Type of match ('exact' or 'fuzzy').
            threshold (int): Minimum score for fuzzy matches. Default is 80.
            fuzzy_algo (str): Fuzzy matching algorithm ('partial_ratio', 'token_sort_ratio', etc.).
        """
        super().__init__(keyword_list)

        self.match_type = match_type
        self.threshold = threshold
        self.fuzzy_algo = fuzzy_algo

        self.fuzzy_algorithms = {
            "partial_ratio": fuzz.partial_ratio,
            "token_sort_ratio": fuzz.token_sort_ratio,
            "token_set_ratio": fuzz.token_set_ratio,
            "ratio": fuzz.ratio,
        }

        if self.fuzzy_algo not in self.fuzzy_algorithms:
            raise ValueError(
                f"Invalid fuzzy algorithm: {self.fuzzy_algo}. Available algorithms: {', '.join(self.fuzzy_algorithms.keys())}"
            )

    def _find_matches(self, text: str) -> List[Dict]:
        """Overrides the base method to find matches based on match type."""
        if self.match_type == "exact":
            return self._exact_match(text)
        elif self.match_type == "fuzzy":
            return self._fuzzy_match(text)
        else:
            raise ValueError("Invalid match_type. Choose either 'exact' or 'fuzzy'.")

    def _exact_match(self, text: str) -> List[Dict]:
        """Finds exact matches using regular expressions."""
        matches = []
        for keyword in self.keyword_list.keywords:
            for term in [keyword.keyword] + keyword.related_terms:
                pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
                for match in pattern.finditer(text):
                    matches.append(
                        {
                            "keyword": keyword.keyword,  # Canonical keyword
                            "start_pos": match.start(),
                            "end_pos": match.end(),
                            "method": "exact_match",
                        }
                    )
        return matches

    def _fuzzy_match(self, text: str) -> List[Dict]:
        """Finds fuzzy matches based on the selected algorithm."""
        matches = []
        fuzzy_function = self.fuzzy_algorithms[self.fuzzy_algo]
        text_lower = text.lower()

        for keyword in self.keyword_list.keywords:
            for term in [keyword.keyword] + keyword.related_terms:
                term_lower = term.lower()
                score = fuzzy_function(text_lower, term_lower)
                if score >= self.threshold:
                    start_pos = text_lower.find(term_lower)
                    if start_pos >= 0:
                        matches.append(
                            {
                                "keyword": keyword.keyword,  # term
                                "start_pos": start_pos,
                                "end_pos": start_pos + len(term),
                                "score": score,
                                "method": "fuzzy_match",
                            }
                        )
        return matches
