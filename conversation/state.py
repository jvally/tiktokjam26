"""Person 1: replace structured updates with an NLU/LLM adapter here."""

from typing import Any, Mapping, Optional

from shared import SearchState


def update_state(previous: Optional[SearchState], query: str,
                 updates: Optional[Mapping[str, Any]] = None) -> SearchState:
    """Merge explicit slots; null removes a slot. Never mutate the previous state.

    No natural-language constraints are inferred by this starter implementation.
    The caller carries the returned state between turns.
    """
    data = previous.to_dict() if previous else SearchState().to_dict()
    if previous:
        data["turn"] += 1
    data["query"] = query
    patch = dict(updates or {})
    if "turn" in patch or "query" in patch:
        raise ValueError("Pass query separately; turn is managed by update_state")
    from shared.contracts import ALIASES
    for key, value in patch.items():
        if key in {"hard_constraints", "soft_preferences", "negative_constraints", "profile"}:
            if not isinstance(value, Mapping):
                raise ValueError(f"{key} must be an object")
            for attr, choice in value.items():
                attr = ALIASES.get(attr, attr)
                if choice is None:
                    data[key].pop(attr, None)
                else:
                    data[key][attr] = choice
        else:
            data[key] = value
    return SearchState.from_dict(data)
