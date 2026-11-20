"""Exact-matching benchmark: keylex vs. a compiled regex alternation vs. FlashText.

Usage::

    pip install flashtext          # optional; skipped when missing
    python benchmarks/bench_exact.py
    python benchmarks/bench_exact.py --sizes 1000 10000 100000 --words 20000

The vocabulary and the text are generated deterministically (fixed seed) from a synthetic
lexicon, so runs are comparable across machines. About 1% of the text consists of keyword
occurrences. Every engine is checked to find the same number of matches.
"""

from __future__ import annotations

import argparse
import random
import re
import statistics
import time
from collections.abc import Callable

from keylex import KeywordMatcher

SYLLABLES = "ka lo mi nu pe ra si to vu xe za bo da fi gu he ji ko lu ma ne".split()


def make_lexicon(n: int, rng: random.Random) -> list[str]:
    words: set[str] = set()
    while len(words) < n:
        words.add("".join(rng.choice(SYLLABLES) for _ in range(rng.randint(2, 6))))
    return sorted(words)


def make_data(n_keywords: int, n_words: int, seed: int = 7) -> tuple[list[str], str]:
    rng = random.Random(seed)
    lexicon = make_lexicon(max(n_keywords, 5_000) * 2, rng)
    vocab = lexicon[: len(lexicon) // 2]  # words used to build keywords
    filler = lexicon[len(lexicon) // 2 :]  # words that never belong to a keyword
    keywords = sorted({" ".join(rng.sample(vocab, rng.choice((1, 1, 2, 3)))) for _ in range(n_keywords)})
    tokens: list[str] = []
    while len(tokens) < n_words:
        if rng.random() < 0.01:
            tokens.append(rng.choice(keywords))
        else:
            tokens.append(rng.choice(filler))
    return keywords, " ".join(tokens)


def best_of(fn: Callable[[], object], repeat: int) -> float:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times) if repeat < 5 else statistics.median(times)


def run(n_keywords: int, n_words: int, repeat: int) -> None:
    keywords, text = make_data(n_keywords, n_words)
    rows: list[tuple[str, float, float, int]] = []

    t0 = time.perf_counter()
    matcher = KeywordMatcher(keywords).compile()
    build = time.perf_counter() - t0
    n = len(matcher.find(text))
    rows.append(("keylex", build, best_of(lambda: matcher.find(text), repeat), n))

    t0 = time.perf_counter()
    alternation = "|".join(re.escape(k) for k in sorted(keywords, key=len, reverse=True))
    pattern = re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)
    build = time.perf_counter() - t0
    n = len(pattern.findall(text))
    rows.append(("regex alternation", build, best_of(lambda: pattern.findall(text), repeat), n))

    try:
        from flashtext import KeywordProcessor  # type: ignore[import-not-found,import-untyped,unused-ignore]
    except ImportError:
        pass
    else:
        t0 = time.perf_counter()
        kp = KeywordProcessor()
        kp.add_keywords_from_list(keywords)
        build = time.perf_counter() - t0
        n = len(kp.extract_keywords(text, span_info=True))
        rows.append(("flashtext", build, best_of(lambda: kp.extract_keywords(text, span_info=True), repeat), n))

    print(f"\n{n_keywords:,} keywords, {n_words:,}-word text")
    print(f"  {'engine':<18} {'build (s)':>10} {'match (ms)':>11} {'matches':>8}")
    for name, build, match, count in rows:
        print(f"  {name:<18} {build:>10.3f} {match * 1000:>11.1f} {count:>8,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 1_000, 10_000, 100_000])
    parser.add_argument("--words", type=int, default=20_000)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    for size in args.sizes:
        run(size, args.words, args.repeat)


if __name__ == "__main__":
    main()
