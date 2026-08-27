"""Person 3: transparent, intent-dependent hybrid ranking baseline."""

from typing import Any, Mapping, Optional, Sequence, Union

from shared import Candidate, SearchState
from shared.contracts import number
from .diagnostics import ranking_diagnostics
from .features import extract_features

# STARTING weights, not fitted or validated against the public development set.
DEFAULT_WEIGHTS = {
    "buying": {"bm25": .20, "dense": .20, "metadata": .15,
               "constraint": .35, "preference": .05, "profile": .05},
    "browsing": {"bm25": .20, "dense": .35, "metadata": .10,
                 "constraint": .10, "preference": .20, "profile": .05},
}
HARD_VIOLATION_PENALTY = .75
NEGATIVE_VIOLATION_PENALTY = 1.0


def rank_candidates(state: Union[SearchState, Mapping[str, Any]],
                    candidates: Sequence[Union[Candidate, Mapping[str, Any]]],
                    top_n: int = 20, weights: Optional[Mapping[str, float]] = None
                    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    if not isinstance(state, SearchState):
        state = SearchState.from_dict(state)
    inputs = [c if isinstance(c, Candidate) else Candidate.from_dict(c) for c in candidates]
    configured = dict(DEFAULT_WEIGHTS[state.intent] if weights is None else weights)
    if not configured or set(configured) - set(DEFAULT_WEIGHTS["buying"]):
        raise ValueError("Weights must use the documented scoring features")
    configured = {key: number(value, f"weight {key}", minimum=0) for key, value in configured.items()}
    if not sum(configured.values()):
        raise ValueError("At least one weight must be positive")
    rows = extract_features(state, inputs)
    # Batch-wide availability: a missing score on one candidate cannot inflate it.
    active = {key: value for key, value in configured.items()
              if any(row["available"][key] for row in rows)}
    total = sum(active.values())
    effective = {key: value / total for key, value in active.items()} if total else {}
    ranked = []
    for candidate, row in zip(inputs, rows):
        features = row["ranking_features"]
        contributions = {key: weight * features[key] for key, weight in effective.items()}
        penalty = (HARD_VIOLATION_PENALTY * features["hard_violation_ratio"]
                   + NEGATIVE_VIOLATION_PENALTY * features["negative_violations"])
        ranked.append({**candidate.to_dict(), **row,
                       "score": sum(contributions.values()) - penalty,
                       "score_contributions": contributions, "penalty": penalty})
    ranked.sort(key=lambda row: (-row["score"], row["parent_asin"]))
    # Keep the best duplicate only, so a target never consumes multiple output slots.
    unique = {}
    for row in ranked:
        unique.setdefault(row["parent_asin"], row)
    ranked = list(unique.values())
    for index, row in enumerate(ranked, 1):
        row["rank"] = index
    diagnostics = ranking_diagnostics(ranked)
    diagnostics.update({"input_count": len(inputs), "duplicate_count": len(inputs) - len(ranked),
                        "returned_count": min(top_n, len(ranked)), "intent": state.intent,
                        "effective_weights": effective, "weights_status": "untuned_baseline",
                        "diagnostics_scope": "all_unique_candidates_before_top_n"})
    return ranked[:top_n], diagnostics
