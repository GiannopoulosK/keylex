"""Keyword extraction algorithms (pure Python, no corpus or keyword list required)."""

from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from statistics import mean, median, pstdev
from typing import Any

from .base import _NP_TAGS, MAX_NAME_LENGTH, BaseExtractor, PreparedText, Word, _Collector, leading_modifier_ok

__all__ = ["RakeExtractor", "TextRankExtractor", "TfidfExtractor", "YakeExtractor"]


class YakeExtractor(BaseExtractor):
    """YAKE! (Campos et al., 2020): statistical features of a single document.

    Considers casing, position, frequency, context diversity and sentence spread. Works well on
    short and long documents without any training data. This is the default extractor.

    Args:
        window: Co-occurrence window used for the context-diversity feature.
    """

    name = "yake"

    def __init__(self, top_n: int | None = 10, *, window: int = 2, **kwargs: Any) -> None:
        super().__init__(top_n, **kwargs)
        self.window = window

    def _score(self, prepared: PreparedText) -> tuple[dict[tuple[str, ...], float], _Collector]:
        words = prepared.words
        tf: Counter[str] = Counter()
        tf_upper: Counter[str] = Counter()
        tf_acronym: Counter[str] = Counter()
        sentences: dict[str, set[int]] = defaultdict(set)
        left: dict[str, list[str]] = defaultdict(list)
        right: dict[str, list[str]] = defaultdict(list)

        for chunk in prepared.chunks():
            for i, w in enumerate(chunk):
                if w.is_number:
                    continue
                tf[w.key] += 1
                sentences[w.key].add(w.sentence)
                if w.acronym:
                    tf_acronym[w.key] += 1
                elif w.surface[0].isupper() and not w.sentence_start:
                    tf_upper[w.key] += 1
                for j in range(max(0, i - self.window), i):
                    other = chunk[j]
                    if not other.is_stop and not other.is_number:
                        left[w.key].append(other.key)
                        right[other.key].append(w.key)

        if not tf:
            return {}, _Collector()
        stop = {w.key for w in words if w.is_stop}
        valid = [c for t, c in tf.items() if t not in stop] or list(tf.values())
        mean_tf, std_tf, max_tf = mean(valid), pstdev(valid), max(valid)
        n_sent = max(1, prepared.n_sentences)

        word_score: dict[str, float] = {}
        for term, freq in tf.items():
            t_case = max(tf_upper[term], tf_acronym[term]) / (1.0 + math.log(freq))
            t_pos = math.log(math.log(3.0 + median(sorted(sentences[term]))))
            t_freq = freq / (mean_tf + std_tf)
            wl = len(set(left[term])) / len(left[term]) if left[term] else 0.0
            wr = len(set(right[term])) / len(right[term]) if right[term] else 0.0
            t_rel = 1.0 + (wl + wr) * (freq / max_tf)
            t_sent = len(sentences[term]) / n_sent
            word_score[term] = (t_rel * t_pos) / (t_case + t_freq / t_rel + t_sent / t_rel)

        collector = _Collector()
        for cand, _ in prepared.candidates(self.max_ngram):
            collector.add(cand)
        scores: dict[tuple[str, ...], float] = {}
        for key, count in collector.count.items():
            content = [word_score[t] for t in key if t not in stop and t in word_score]
            if not content:
                continue
            prod = math.prod(content)
            yake = prod / (count * (1.0 + sum(content)))
            scores[key] = 1.0 / (yake + 1e-9)  # YAKE: lower is better -> invert
        return scores, collector


class RakeExtractor(BaseExtractor):
    """RAKE (Rose et al., 2010): phrases are runs of non-stopwords; words are scored by
    degree / frequency. Favours longer, specific phrases.

    Takes the :class:`BaseExtractor` arguments; ``max_ngram`` defaults to 4 here.
    """

    name = "rake"

    def __init__(self, top_n: int | None = 10, *, max_ngram: int = 4, **kwargs: Any) -> None:
        super().__init__(top_n, max_ngram=max_ngram, **kwargs)

    def _segments(self, chunk: list[Word]) -> Iterator[list[Word]]:
        """Runs of content words, split after possessives, around verb-like "-ed"/"-ly" words and
        (with a POS tagger) at non-nominal words."""
        run: list[Word] = []
        for k, w in enumerate(chunk):
            content = not w.is_stop and not w.is_number and (w.pos is None or w.pos in _NP_TAGS)
            # "-ed" after an ambiguous noun/adjective stays inside ("mean squared error")
            modifier = bool(run) and run[-1].ambiguous
            if content and w.weak_end and run and not modifier:
                yield run
                run = []
            if not content:
                if run:
                    yield run
                run = []
                continue
            if w.weak_end and not modifier and not leading_modifier_ok(chunk, k):
                continue  # a verb ("praised Apple"), not a leading adjective ("a distributed store")
            run.append(w)
            if w.possessive:
                yield run
                run = []
        if run:
            yield run

    def _phrases(self, run: list[Word]) -> Iterator[list[Word]]:
        while run and run[-1].bad_end:
            run = run[:-1]
        if not run:
            return
        size = len(run)
        if size <= self.max_ngram or (size <= MAX_NAME_LENGTH and all(w.capitalised for w in run)):
            yield run
            return
        # too long: split into balanced pieces instead of dropping the run
        pieces = math.ceil(size / self.max_ngram)
        bounds = [round(i * size / pieces) for i in range(pieces + 1)]
        for a, b in itertools.pairwise(bounds):
            yield from self._phrases(run[a:b])

    def _score(self, prepared: PreparedText) -> tuple[dict[tuple[str, ...], float], _Collector]:
        collector = _Collector()
        phrases: list[tuple[str, ...]] = []
        for chunk in prepared.chunks():
            for run in self._segments(chunk):
                for words in self._phrases(run):
                    phrases.append(collector.add(words))
        freq: Counter[str] = Counter()
        degree: Counter[str] = Counter()
        for phrase in phrases:
            for word in phrase:
                freq[word] += 1
                degree[word] += len(phrase)
        word_score = {w: degree[w] / freq[w] for w in freq}
        scores = {key: sum(word_score[w] for w in key) for key in collector.count}
        return scores, collector


class TextRankExtractor(BaseExtractor):
    """TextRank (Mihalcea & Tarau, 2004): PageRank over a word co-occurrence graph.

    Args:
        window: Co-occurrence window (in content words).
        damping: PageRank damping factor.
        iterations: Maximum number of PageRank iterations (stops earlier once converged).
    """

    name = "textrank"

    def __init__(
        self, top_n: int | None = 10, *, window: int = 3, damping: float = 0.85, iterations: int = 50, **kwargs: Any
    ) -> None:
        super().__init__(top_n, **kwargs)
        self.window = window
        self.damping = damping
        self.iterations = iterations

    def _score(self, prepared: PreparedText) -> tuple[dict[tuple[str, ...], float], _Collector]:
        graph: dict[str, Counter[str]] = defaultdict(Counter)
        by_sentence: dict[int, list[str]] = defaultdict(list)
        for w in prepared.words:
            # nodes are nominal words only: no stopwords, numbers, "-ed"/"-ly" words (or non-nouns/adjectives
            # when a POS tagger is available)
            if not w.is_stop and not w.is_number and not w.weak_end and (w.pos is None or w.pos in _NP_TAGS):
                by_sentence[w.sentence].append(w.key)
        for seq in by_sentence.values():
            for i, a in enumerate(seq):
                graph[a]  # ensure node exists
                for b in seq[i + 1 : i + self.window]:
                    if a != b:
                        graph[a][b] += 1
                        graph[b][a] += 1
        if not graph:
            return {}, _Collector()
        nodes = list(graph)
        rank = dict.fromkeys(nodes, 1.0 / len(nodes))
        out_weight = {n: sum(graph[n].values()) for n in nodes}
        for _ in range(self.iterations):
            new = {}
            for n in nodes:
                incoming = sum(rank[m] * w / out_weight[m] for m, w in graph[n].items() if out_weight[m])
                new[n] = (1 - self.damping) / len(nodes) + self.damping * incoming
            delta = sum(abs(new[n] - rank[n]) for n in nodes)
            rank = new
            if delta < 1e-6:
                break
        collector = _Collector()
        for cand, _ in prepared.candidates(self.max_ngram):
            if all(not w.is_stop for w in cand):
                collector.add(cand)
        scores = {key: sum(rank.get(w, 0.0) for w in key) / math.sqrt(len(key)) for key in collector.count}
        return scores, collector


class TfidfExtractor(BaseExtractor):
    """TF-IDF over candidate phrases.

    Call :meth:`fit` with a corpus (an iterable of documents) to learn document frequencies; terms
    that are common across the corpus then drop in the ranking. Without fitting, the sentences of
    the input text serve as pseudo-documents. Term frequency is sublinear (``1 + log(count)``),
    multi-word phrases get a small length bonus, and ties are broken by first occurrence.

    Args:
        corpus: Optional documents to :meth:`fit` on right away.
    """

    name = "tfidf"

    def __init__(self, top_n: int | None = 10, *, corpus: Iterable[str] | None = None, **kwargs: Any) -> None:
        super().__init__(top_n, **kwargs)
        self.df: Counter[tuple[str, ...]] = Counter()
        self.n_docs = 0
        if corpus is not None:
            self.fit(corpus)

    def _keys(self, prepared: PreparedText) -> set[tuple[str, ...]]:
        keys = {tuple(w.key for w in cand) for cand, _ in prepared.candidates(self.max_ngram)}
        # single content words too, so unseen phrases can fall back on their words' frequencies
        keys.update((w.key,) for w in prepared.words if not w.is_stop and not w.is_number)
        return keys

    def fit(self, corpus: Iterable[str]) -> TfidfExtractor:
        """Learn document frequencies from a corpus (a non-empty iterable of strings). Returns ``self``."""
        if isinstance(corpus, (str, bytes)):
            raise TypeError("corpus must be an iterable of documents (e.g. a list of str), not a single string")
        df: Counter[tuple[str, ...]] = Counter()
        n_docs = 0
        for doc in corpus:
            if not isinstance(doc, str):
                raise TypeError(f"corpus documents must be str, got {type(doc).__name__}")
            df.update(self._keys(self.prepare(doc)))
            n_docs += 1
        if not n_docs:
            raise ValueError("corpus is empty: fit() needs at least one document")
        self.df, self.n_docs = df, n_docs
        return self

    @property
    def is_fitted(self) -> bool:
        """``True`` once :meth:`fit` has learned document frequencies from a corpus."""
        return self.n_docs > 0

    def _score(self, prepared: PreparedText) -> tuple[dict[tuple[str, ...], float], _Collector]:
        collector = _Collector()
        per_sentence: dict[int, set[tuple[str, ...]]] = defaultdict(set)
        for cand, _ in prepared.candidates(self.max_ngram):
            key = collector.add(cand)
            per_sentence[cand[0].sentence].add(key)
            per_sentence[cand[0].sentence].update((w.key,) for w in cand)
        if self.is_fitted:
            df, n = self.df, self.n_docs
        else:
            df = Counter(k for keys in per_sentence.values() for k in keys)
            n = max(1, len(per_sentence))

        def idf(key: tuple[str, ...]) -> float:
            return math.log((1 + n) / (1 + df.get(key, 0))) + 1.0

        scores = {}
        for key, count in collector.count.items():
            weight = idf(key)
            if len(key) > 1:
                words = [idf((t,)) for t in key if t not in prepared.stopwords]
                if words:  # an unseen phrase of common words is still fairly common
                    weight = (weight + sum(words) / len(words)) / 2
            tf = 1.0 + math.log(count)
            scores[key] = tf * weight * (1.0 + 0.25 * (len(key) - 1))
        return scores, collector
