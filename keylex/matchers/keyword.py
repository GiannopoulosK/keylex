from typing import Set, List, Dict


class Keyword:
    def __init__(self, keyword: str):
        self.keyword = keyword
        self._synonyms: Set[str] = set()
        self._acronyms: Set[str] = set()

    # --- Synonyms Management ---

    @property
    def synonyms(self) -> List[str]:
        """Get the list of synonyms."""
        return list(self._synonyms)

    @synonyms.setter
    def synonyms(self, synonyms: List[str]) -> None:
        """Set the list of synonyms."""
        self._synonyms = set(synonyms)

    def add_synonyms(self, synonyms: List[str]) -> None:
        """Add a list of synonyms."""
        self._synonyms.update(synonyms)

    def remove_synonyms(self, synonyms: List[str]) -> None:
        """Remove a list of synonyms if present."""
        self._synonyms.difference_update(synonyms)

    def set_synonyms(self, synonyms: List[str]) -> None:
        """Overwrite existing synonyms with new ones."""
        self._synonyms = set(synonyms)

    def clear_synonyms(self) -> None:
        """Remove all synonyms."""
        self._synonyms.clear()

    @property
    def has_synonyms(self) -> bool:
        """Whether the keyword has any synonyms."""
        return bool(self._synonyms)

    # --- Acronyms Management ---

    @property
    def acronyms(self) -> List[str]:
        """Get the list of acronyms."""
        return list(self._acronyms)

    @acronyms.setter
    def acronyms(self, acronyms: List[str]) -> None:
        """Set the list of acronyms."""
        self._acronyms = set(acronyms)

    def add_acronyms(self, acronyms: List[str]) -> None:
        """Add a list of acronyms."""
        self._acronyms.update(acronyms)

    def remove_acronyms(self, acronyms: List[str]) -> None:
        """Remove a list of acronyms if present."""
        self._acronyms.difference_update(acronyms)

    def set_acronyms(self, acronyms: List[str]) -> None:
        """Overwrite existing acronyms with new ones."""
        self._acronyms = set(acronyms)

    def clear_acronyms(self) -> None:
        """Remove all acronyms."""
        self._acronyms.clear()

    @property
    def has_acronyms(self) -> bool:
        """Whether the keyword has any acronyms."""
        return bool(self._acronyms)

    # --- Related Terms Management ---

    @property
    def related_terms(self) -> List[str]:
        """Get the list of related terms (synonyms and acronyms)."""
        return list(self._synonyms | self._acronyms)

    @related_terms.setter
    def related_terms(self, terms: Dict[str, List[str]]) -> None:
        """Set the list of related terms (synonyms and acronyms)."""
        self._synonyms = set(terms.get("synonyms", []))
        self._acronyms = set(terms.get("acronyms", []))

    def add_related_terms(
        self, synonyms: List[str] = [], acronyms: List[str] = []
    ) -> None:
        """Add synonyms and/or acronyms to related terms."""
        if synonyms:
            self._synonyms.update(synonyms)
        if acronyms:
            self._acronyms.update(acronyms)

    def remove_related_terms(
        self, synonyms: List[str] = [], acronyms: List[str] = []
    ) -> None:
        """Remove synonyms and/or acronyms from related terms."""
        if synonyms:
            self._synonyms.difference_update(synonyms)
        if acronyms:
            self._acronyms.difference_update(acronyms)

    def clear_related_terms(self) -> None:
        """Clear all related terms (synonyms and acronyms)."""
        self._synonyms.clear()
        self._acronyms.clear()

    def has_related_terms(self) -> bool:
        """Check if the keyword has any related terms."""
        return bool(self._synonyms or self._acronyms)

    # --- Utility Methods ---

    def copy(self) -> "Keyword":
        """Return a deep copy of the keyword."""
        new_keyword = Keyword(self.keyword)
        new_keyword._synonyms = set(self._synonyms)
        new_keyword._acronyms = set(self._acronyms)
        return new_keyword

    def to_dict(self) -> Dict[str, List[str]]:
        """Serialize the keyword to a dictionary."""
        return {
            "keyword": self.keyword,
            "synonyms": list(self._synonyms),
            "acronyms": list(self._acronyms),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, List[str]]) -> "Keyword":
        """Deserialize a dictionary into a Keyword object."""
        obj = cls(data["keyword"])
        obj.set_synonyms(data.get("synonyms", []))
        obj.set_acronyms(data.get("acronyms", []))
        return obj

    # --- Magic Methods ---

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Keyword):
            return NotImplemented
        return (
            self.keyword == other.keyword
            and self._synonyms == other._synonyms
            and self._acronyms == other._acronyms
        )

    def __lt__(self, other: "Keyword") -> bool:
        if not isinstance(other, Keyword):
            return NotImplemented
        return (self.keyword.lower(), len(self._synonyms), len(self._acronyms)) < (
            other.keyword.lower(),
            len(other._synonyms),
            len(other._acronyms),
        )

    def __hash__(self) -> int:
        return hash(
            (self.keyword, frozenset(self._synonyms), frozenset(self._acronyms))
        )

    def __repr__(self) -> str:
        return (
            f"Keyword(keyword={self.keyword!r}, "
            f"synonyms={self._synonyms!r}, "
            f"acronyms={self._acronyms!r})"
        )

    def __str__(self) -> str:
        return f"{self.keyword} (Synonyms: {len(self._synonyms)}, Acronyms: {len(self._acronyms)})"
