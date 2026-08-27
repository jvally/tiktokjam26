"""Person 5: replay explicit states; target labels stay outside online components."""

import math
import time
from typing import Any, Sequence

from ranking import rank_candidates
from retrieval import Retriever
from shared import SearchState
from .metrics import ranking_metrics


def evaluate_cases(retriever: Retriever, cases: Sequence[dict[str, Any]],
                   candidate_limit: int = 200) -> dict[str, Any]:
    if not cases:
        raise ValueError("Evaluation requires at least one case")
    if isinstance(candidate_limit, bool) or not isinstance(candidate_limit, int) or candidate_limit < 10:
        raise ValueError("candidate_limit must be an integer >= 10")
    targets, before, after, details, durations = [], [], [], [], []
    retrieved_count = 0
    for case in cases:
        target = case.get("target_parent_asin")
        if not isinstance(target, str) or not target:
            raise ValueError("Each evaluation case needs a target_parent_asin")
        state = SearchState.from_dict(case["state"])
        start = time.perf_counter()
        candidates = retriever.retrieve(state, limit=candidate_limit)
        ranked, _ = rank_candidates(state, candidates, top_n=10)
        durations.append((time.perf_counter() - start) * 1000)
        retrieved_ids = [candidate.product.parent_asin for candidate in candidates]
        ranked_ids = [row["parent_asin"] for row in ranked]
        targets.append(target)
        before.append(retrieved_ids[:10])
        after.append(ranked_ids)
        retrieved_count += int(target in retrieved_ids)
        details.append({"session_id": case.get("session_id"), "turn": state.turn,
                        "target_parent_asin": target,
                        "retrieval_rank": retrieved_ids.index(target) + 1 if target in retrieved_ids else None,
                        "ranked_top10_rank": ranked_ids.index(target) + 1 if target in ranked_ids else None})
    baseline, ranked_metrics = ranking_metrics(before, targets), ranking_metrics(after, targets)
    return {"evaluation_mode": "fixed_state_replay_not_interactive_challenge_score",
            "retrieval_top10": baseline, "ranked_top10": ranked_metrics,
            "delta_mrr@10": ranked_metrics["mrr@10"] - baseline["mrr@10"],
            "retrieval_recall": retrieved_count / len(cases), "candidate_limit": candidate_limit,
            "mean_latency_ms": sum(durations) / len(durations),
            "p95_latency_ms": sorted(durations)[math.ceil(.95 * len(durations)) - 1],
            "latency_scope": "retrieve_plus_rank_excludes_index_load",
            "cases": details}
