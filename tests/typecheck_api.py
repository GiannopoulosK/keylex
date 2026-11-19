"""Static checks of the public API's types (run by ``mypy``; never executed by pytest).

Every ``assert_type`` fails the mypy run if the inferred type drifts from the documented one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    from typing_extensions import assert_type

    from keylex import KeywordMatcher, Match, OutputFormat

    def check_output_types(matcher: KeywordMatcher, text: str) -> None:
        assert_type(matcher.find(text), list[Match])
        assert_type(matcher(text), list[Match])
        assert_type(matcher.match(text), list[Match])
        assert_type(matcher.match(text, "matches"), list[Match])
        assert_type(matcher.match(text, "detailed"), list[dict[str, Any]])
        assert_type(matcher.match(text, "keywords"), list[str])
        assert_type(matcher.match(text, OutputFormat.KEYWORDS), list[str])
        assert_type(matcher.match(text, "boolean"), dict[str, bool])
        assert_type(matcher.match(text, "count"), dict[str, int])
        assert_type(matcher.match(text, "categories"), dict[str | None, int])
        assert_type(matcher.match(text, "score"), float)
        assert_type(matcher.count(text), dict[str, int])
        assert_type(matcher.score(text), float)
        assert_type(matcher.contains(text), bool)
        assert_type(matcher.match_many([text]), Iterator[list[Match]])
        assert_type(matcher.match_many([text], "keywords"), Iterator[list[str]])
        assert_type(matcher.highlight(text), str)
