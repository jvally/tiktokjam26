"""Person 2: small BM25 baseline, replaceable by a hybrid index."""

import math
from collections import Counter, defaultdict
from typing import Protocol, Sequence

from shared import Candidate, Product, SearchState
from shared.text import tokens


class Retriever(Protocol):
    def retrieve(self, state: SearchState, limit: int = 200) -> list[Candidate]:
        ...


class LexicalRetriever:
    def __init__(self, products: Sequence[Product]) -> None:
        self.products = list(products)
        if len({p.parent_asin for p in products}) != len(products):
            raise ValueError("Duplicate catalog parent_asin")
        self.term_counts = [Counter(tokens(p.text())) for p in self.products]
        self.lengths = [sum(counts.values()) for counts in self.term_counts]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths)) or 1
        self.postings = defaultdict(list)
        for index, counts in enumerate(self.term_counts):
            for term, frequency in counts.items():
                self.postings[term].append((index, frequency))

    def retrieve(self, state: SearchState, limit: int = 200) -> list[Candidate]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        query_parts = [state.query]
        for preferences in (state.hard_constraints, state.soft_preferences):
            for key, values in preferences.items():
                if not key.startswith("budget_"):
                    query_parts.extend(values)
        query_terms = set(tokens(" ".join(query_parts)))
        scores = defaultdict(float)
        count = len(self.products)
        k1, b = 1.5, 0.75
        for term in sorted(query_terms):
            postings = self.postings.get(term, [])
            idf = math.log(1 + (count - len(postings) + 0.5) / (len(postings) + 0.5))
            for index, frequency in postings:
                length_norm = 1 - b + b * self.lengths[index] / self.average_length
                scores[index] += idf * frequency * (k1 + 1) / (frequency + k1 * length_norm)
        # Retain a deterministic zero-score fallback. Never filter unknown metadata.
        indices = sorted(range(count), key=lambda i: (-scores[i], self.products[i].parent_asin))[:limit]
        return [Candidate(self.products[i], bm25_score=scores[i]) for i in indices]
