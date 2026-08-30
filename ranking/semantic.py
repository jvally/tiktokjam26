from __future__ import annotations

from typing import Protocol, Sequence

from retrieval import Candidate


class SemanticReranker(Protocol):
    """Scores a small candidate shortlist against the cumulative query."""

    def score(self, query: str, candidates: Sequence[Candidate]) -> Sequence[float]: ...


class CrossEncoderReranker:
    """Optional local CrossEncoder adapter; never used over the whole catalog."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Semantic reranking requires requirements-semantic.txt and a local model"
            ) from exc
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def score(self, query: str, candidates: Sequence[Candidate]) -> list[float]:
        pairs = [(query, item.searchable_text) for item in candidates]
        values = self.model.predict(pairs, show_progress_bar=False)
        return [float(value) for value in values]
