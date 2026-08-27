"""Relative confidence signals, not calibrated probabilities of a purchase."""

import math
from typing import Any, Sequence


def ranking_diagnostics(ranked: Sequence[dict[str, Any]], temperature: float = 0.15) -> dict[str, Any]:
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be positive and finite")
    scores = [row["score"] for row in ranked]
    top = scores[0] if scores else None
    if len(scores) > 1:
        exp_scores = [math.exp((s - max(scores)) / temperature) for s in scores]
        total = sum(exp_scores)
        probabilities = [s / total for s in exp_scores]
        entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
        normalized_entropy = entropy / math.log(len(scores))
    else:
        entropy = normalized_entropy = 0.0
    return {
        "candidate_count": len(scores), "top_score": top,
        "score_gap_1_2": scores[0] - scores[1] if len(scores) > 1 else None,
        "score_gap_1_10": scores[0] - scores[9] if len(scores) >= 10 else None,
        "candidate_entropy": entropy, "normalized_entropy": normalized_entropy,
        "entropy_temperature": temperature,
        "top_constraint_coverage": ranked[0]["ranking_features"]["constraint_coverage"] if ranked else None,
        "top_hard_violation_ratio": ranked[0]["ranking_features"]["hard_violation_ratio"] if ranked else None,
        "top_negative_violations": ranked[0]["ranking_features"]["negative_violations"] if ranked else None,
        "calibrated": False,
    }
