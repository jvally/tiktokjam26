"""Person 4: deterministic clarification policy; thresholds are not calibrated."""

from typing import Any, Sequence

from shared import SearchState
from ranking.features import observed_values
from shared import Product


def decide(state: SearchState, ranked: Sequence[dict[str, Any]],
           diagnostics: dict[str, Any]) -> dict[str, Any]:
    if state.turn >= 10:
        return {"action": "recommend", "question": None, "attribute": None,
                "reason": "turn_limit", "terminal": True}
    has_violation = bool(diagnostics.get("top_hard_violation_ratio")
                         or diagnostics.get("top_negative_violations"))
    gap = diagnostics.get("score_gap_1_2")
    coverage = diagnostics.get("top_constraint_coverage") or 0
    constraints_known = not state.hard_constraints or coverage == 1
    if ranked and not has_violation and constraints_known and gap is not None and gap >= .12:
        return {"action": "recommend", "question": None, "attribute": None,
                "reason": "clear_leader_heuristic", "terminal": False}
    constrained = set(state.hard_constraints) | set(state.soft_preferences) | set(state.negative_constraints)
    for attribute in ("category", "color", "brand", "size", "material"):
        if attribute in constrained or attribute in state.asked_attributes:
            continue
        options = sorted({value for row in ranked[:10]
                          for value in observed_values(Product.from_dict(row), attribute)})
        if len(options) >= 2:
            return {"action": "clarify", "attribute": attribute,
                    "question": f"Do you have a preferred {attribute}? Options include {', '.join(options[:4])}.",
                    "reason": "ambiguous_shortlist", "terminal": False}
    return {"action": "recommend", "question": None, "attribute": None,
            "reason": "no_candidates" if not ranked else "no_useful_unasked_attribute", "terminal": False}
