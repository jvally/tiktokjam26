"""Run from the repository root: python3 -m examples.ranking_only."""

import json

from ranking import rank_candidates

state = {"intent": "buying", "hard_constraints": {"color": "black", "budget_max": 100},
         "soft_preferences": {"feature": ["comfortable"], "use_case": ["running"]}}
candidates = [
    {"parent_asin": "EXAMPLE1", "title": "Black cushioned running sneaker", "color": "black",
     "price": 79, "bm25_score": .71, "dense_score": .83, "metadata_score": .90},
    {"parent_asin": "EXAMPLE2", "title": "White racing sneaker", "color": "white",
     "price": 149, "bm25_score": .95, "dense_score": .85, "metadata_score": .40},
]
ranked, diagnostics = rank_candidates(state, candidates, top_n=10)
print(json.dumps({"ranked": ranked, "diagnostics": diagnostics}, indent=2))
