"""Optional shortlist-only reranker boundary; no LLM calls are made by default."""

from typing import Any, Protocol, Sequence

from shared import SearchState


class ShortlistReranker(Protocol):
    def order(self, state: SearchState, shortlist: Sequence[dict[str, Any]]) -> Sequence[str]:
        """Return parent_asin values in desired order (at most 20 inputs)."""
        ...


def rerank_shortlist(state: SearchState, ranked: Sequence[dict[str, Any]],
                    reranker: ShortlistReranker, limit: int = 20) -> list[dict[str, Any]]:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
        raise ValueError("Reranker shortlist must contain at most 20 products")
    shortlist = list(ranked[:limit])
    by_id = {row["parent_asin"]: row for row in shortlist}
    if len(by_id) != len(shortlist):
        raise ValueError("Reranker input must have unique parent_asin values")
    proposed = list(reranker.order(state, shortlist))
    if any(not isinstance(asin, str) for asin in proposed):
        raise ValueError("Reranker must return string parent_asin values")
    if len(set(proposed)) != len(proposed) or any(asin not in by_id for asin in proposed):
        raise ValueError("Reranker returned duplicate or unknown product IDs")
    proposed_set = set(proposed)
    ordered = [by_id[asin] for asin in proposed]
    ordered += [row for row in shortlist if row["parent_asin"] not in proposed_set]
    ordered += list(ranked[limit:])
    return [{**row, "rank": rank, "local_rank": row.get("rank")}
            for rank, row in enumerate(ordered, 1)]
