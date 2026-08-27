"""Offline Top-k metrics. A missing target is a miss, never dropped."""

from typing import Any, Sequence


def ranking_metrics(predictions: Sequence[Sequence[str]], targets: Sequence[str], k: int = 10) -> dict[str, Any]:
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have equal lengths")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    hits, reciprocal_ranks = [], []
    for ids, target in zip(predictions, targets):
        unique = list(dict.fromkeys(ids))[:k]
        rank = unique.index(target) + 1 if target in unique else None
        hits.append(int(rank is not None))
        reciprocal_ranks.append(1 / rank if rank else 0.0)
    count = len(targets)
    return {"cases": count, f"hit@{k}": sum(hits) / count if count else 0.0,
            f"mrr@{k}": sum(reciprocal_ranks) / count if count else 0.0}
