"""Shared test helpers."""

from __future__ import annotations

import os
import time
from collections.abc import Callable

import pytest

#: Wall-clock budgets in performance tests are scaled by this factor. Shared CI runners (and
#: coverage tracing) are several times slower than a developer machine, while the regressions these
#: tests guard against (quadratic algorithms) are 10-100x slower, so a generous factor keeps them
#: meaningful without making them flaky. Override with KEYLEX_TIME_SLACK.
TIME_SLACK = float(os.environ.get("KEYLEX_TIME_SLACK") or (5.0 if os.environ.get("CI") else 1.0))


@pytest.fixture
def time_budget() -> Callable[[float, float], None]:
    """``time_budget(start, seconds)`` asserts that less than ``seconds`` (x slack) passed since
    ``start = time.perf_counter()``."""

    def check(start: float, seconds: float) -> None:
        elapsed = time.perf_counter() - start
        budget = seconds * TIME_SLACK
        assert elapsed < budget, f"took {elapsed:.2f}s, budget {budget:.2f}s ({seconds}s x {TIME_SLACK} slack)"

    return check
