"""Person 5: end-to-end orchestration boundary, independent of transport."""

from typing import Any, Mapping, Union

from policy import decide
from ranking import rank_candidates
from retrieval import Retriever
from shared import SearchState


class ShoppingCopilot:
    def __init__(self, retriever: Retriever, candidate_limit: int = 200) -> None:
        if isinstance(candidate_limit, bool) or not isinstance(candidate_limit, int) or candidate_limit < 10:
            raise ValueError("candidate_limit must be an integer >= 10")
        self.retriever = retriever
        self.candidate_limit = candidate_limit

    def search(self, state: Union[SearchState, Mapping[str, Any]]) -> dict[str, Any]:
        if not isinstance(state, SearchState):
            state = SearchState.from_dict(state)
        candidates = self.retriever.retrieve(state, limit=self.candidate_limit)
        ranked, diagnostics = rank_candidates(state, candidates, top_n=20)
        decision = decide(state, ranked, diagnostics)
        next_state = state.to_dict()
        attribute = decision["attribute"]
        if attribute and attribute not in next_state["asked_attributes"]:
            next_state["asked_attributes"].append(attribute)
        return {"state": next_state, "results": ranked[:10],
                "parent_asins": [row["parent_asin"] for row in ranked[:10]],
                "diagnostics": diagnostics, "decision": decision}
