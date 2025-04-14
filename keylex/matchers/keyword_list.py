from .keyword import Keyword
from typing import List, Dict, Optional, Set, Union
import json


class KeywordList:
    def __init__(self, keywords: Optional[List["Keyword"]] = None):
        """
        Initializes a list of Keyword objects.

        Args:
            keywords (Optional[List[Keyword]]): A list of Keyword objects (default empty list).
        """
        self.keywords: List["Keyword"] = []
        self._keyword_set: Set[str] = set()
        if keywords:
            for keyword in keywords:
                self.add_keyword(keyword)

    @classmethod
    def from_list(cls, keywords: List[Union[str, "Keyword"]]) -> "KeywordList":
        """
        Creates a KeywordList from a list of keywords (strings or Keyword objects).

        Args:
            keywords (List[Union[str, Keyword]]): List containing either strings or Keyword instances.

        Returns:
            KeywordList: A populated KeywordList.
        """
        keyword_list = cls()
        for item in keywords:
            if isinstance(item, Keyword):
                keyword_list.add_keyword(item)
            elif isinstance(item, str):
                keyword_list.add_keyword(Keyword(item))
            else:
                raise TypeError(f"Expected str or Keyword, got {type(item)}")
        return keyword_list

    @classmethod
    def from_dict(
        cls, keyword_dict: Dict[str, List[Union[str, Dict[str, List[str]]]]]
    ) -> "KeywordList":
        """
        Creates a KeywordList from a dictionary of keywords with related terms.

        Args:
            keyword_dict (dict): Dictionary with keywords as keys and lists of related terms as values.

        Returns:
            KeywordList: A populated KeywordList.
        """
        keyword_list = cls()
        for keyword, related_terms in keyword_dict.items():
            kw = Keyword(keyword)

            for term in related_terms:
                if isinstance(term, str):
                    kw.add_synonyms([term])
                elif isinstance(term, dict):
                    for type_, terms in term.items():
                        if type_ == "acronyms":
                            kw.add_acronyms(terms)
                        elif type_ == "synonyms":
                            kw.add_synonyms(terms)
                        else:
                            raise ValueError(
                                f"Unknown type '{type_}' in related terms."
                            )
                else:
                    raise TypeError(f"Invalid term format: {type(term)}")

            keyword_list.add_keyword(kw)
        return keyword_list

    def to_dict(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Return the keywords as a dictionary, with the keyword as the key
        and its synonyms and acronyms as values.

        Returns:
            dict: Dictionary representation of the keywords.
        """
        return {
            keyword.keyword: {
                "synonyms": list(keyword._synonyms),
                "acronyms": list(keyword._acronyms),
            }
            for keyword in self.keywords
        }

    def to_list(self) -> List[Dict[str, List[str]]]:
        """
        Return the keywords as a list of dictionaries,
        each including the keyword, synonyms, and acronyms.

        Returns:
            list: List of dictionaries representing keywords.
        """
        return [keyword.to_dict() for keyword in self.keywords]

    def to_json(self) -> str:
        """
        Serialize the KeywordList to a JSON string.

        Returns:
            str: JSON-formatted string of the KeywordList.
        """
        return json.dumps(self.to_dict(), indent=2)

    def merge(self, other: "KeywordList") -> "KeywordList":
        """
        Merge another KeywordList into this one, combining related terms.

        Args:
            other (KeywordList): Another KeywordList instance.

        Returns:
            KeywordList: The merged KeywordList (self).
        """
        for other_keyword in other.keywords:
            existing_keyword = self.get_keyword(other_keyword.keyword)
            if existing_keyword:
                existing_keyword.add_synonyms(other_keyword.synonyms)
                existing_keyword.add_acronyms(other_keyword.acronyms)
            else:
                self.add_keyword(other_keyword)
        return self

    def add_keyword(self, keyword: "Keyword") -> None:
        """
        Add a new Keyword to the list if it's not already present.

        Args:
            keyword (Keyword): The Keyword object to add.
        """
        if keyword.keyword not in self._keyword_set:
            self.keywords.append(keyword)
            self._keyword_set.add(keyword.keyword)

    def remove_keyword(self, keyword: "Keyword") -> None:
        """
        Remove a Keyword object from the list.

        Args:
            keyword (Keyword): The Keyword object to remove.
        """
        if keyword.keyword in self._keyword_set:
            self.keywords.remove(keyword)
            self._keyword_set.remove(keyword.keyword)

    def get_index(self, keyword_name: str, raise_error: bool = False) -> Optional[int]:
        """
        Get the index of a keyword in the list.

        Args:
            keyword_name (str): The keyword to search for.
            raise_error (bool): Whether to raise an error if not found.

        Returns:
            Optional[int]: The index of the keyword, or None if not found.
        """
        for index, keyword in enumerate(self.keywords):
            if keyword.keyword == keyword_name:
                return index
        if raise_error:
            raise ValueError(f"Keyword '{keyword_name}' not found.")
        return None

    def get_keyword(self, keyword_str: str) -> Optional["Keyword"]:
        """
        Retrieve a Keyword object by its keyword string.

        Args:
            keyword_str (str): The keyword to search for.

        Returns:
            Keyword or None: The Keyword object if found, otherwise None.
        """
        for keyword in self.keywords:
            if keyword.keyword == keyword_str:
                return keyword
        return None

    def clear(self) -> None:
        """Clear all keywords from the list."""
        self.keywords.clear()
        self._keyword_set.clear()

    def get_all_keywords(self) -> List[str]:
        """Return a list of all primary keywords."""
        return [keyword.keyword for keyword in self.keywords]

    def get_keywords_and_related_terms(self) -> Dict[str, Set[str]]:
        """Return a mapping of each keyword to its related terms."""
        return {keyword.keyword: keyword.related_terms for keyword in self.keywords}

    def __iter__(self):
        """Allow iteration over the KeywordList."""
        return iter(self.keywords)

    def __getitem__(self, index: Union[int, slice]) -> Union[Keyword, List[Keyword]]:
        """Allow indexed access to the KeywordList."""
        return self.keywords[index]

    def __len__(self) -> int:
        """Return the number of keywords in the list."""
        return len(self.keywords)

    def __eq__(self, other: object) -> bool:
        """Check equality based on the keywords."""
        if not isinstance(other, KeywordList):
            return NotImplemented
        return set(self.get_all_keywords()) == set(other.get_all_keywords())

    def __contains__(self, term: str) -> bool:
        """Check if a term is in the keyword list as a primary keyword or related term."""
        if term in self._keyword_set:
            return True
        return any(
            term in keyword.related_terms or term == keyword.keyword
            for keyword in self.keywords
        )

    def __repr__(self):
        """Return a string representation of the KeywordList."""
        return f"KeywordList(keywords={self.keywords!r})"
