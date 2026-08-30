from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from statistics import median
from typing import Sequence

from conversation import ConversationState
from ranking import RankedCandidate


QUESTIONS = {
    "other": "What other requirement matters most to you?",
    "feature": "Which product feature matters most to you?",
    "material": "Do you have a material preference?",
    "style": "Do you have a preferred style or fit?",
    "use_case": "What will you mainly use it for?",
    "color": "Do you have a color preference?",
    "budget": "What budget range should I use?",
    "size": "Do you have a size or width requirement?",
    "brand": "Do you prefer a particular brand?",
    "category": "Which product category is the closest match?",
}
FALLBACK_ORDER = ("feature", "material", "style", "use_case", "color", "budget", "size", "brand", "category")
MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "linen", "canvas", "suede")
COLORS = ("black", "white", "blue", "red", "pink", "green", "brown", "gray", "purple", "yellow", "orange", "beige", "navy")
STYLES = ("casual", "formal", "athletic", "vintage", "classic", "modern", "slim", "relaxed", "loose", "fitted")
USE_CASES = ("hiking", "running", "walking", "gym", "training", "winter", "outdoor", "work", "travel", "wedding")
FEATURE_STOPWORDS = {
    "and", "the", "with", "for", "this", "that", "from", "your", "are", "has",
    "have", "product", "products", "women", "womens", "men", "mens", "size",
    "color", "style", "material",
}
PROFILE_ATTRIBUTE_PRIORS = {
    "material": ("material",),
    "style": ("style", "fit"),
    "size": ("fit",),
    "feature": ("comfort", "durability", "weather", "warmth", "quality"),
}
ANSWER_PRIOR = {
    "feature": 0.95, "material": 0.90, "style": 0.78, "use_case": 0.75,
    "color": 0.72, "budget": 0.68, "size": 0.62, "brand": 0.45, "category": 0.40,
}


@dataclass(frozen=True)
class Clarification:
    attribute: str | None
    message: str
    reason: str
    utilities: dict[str, float] = field(default_factory=dict)


def _first_term(text: str, values: Sequence[str]) -> str | None:
    lowered = text.casefold()
    for value in values:
        if re.search(rf"\b{re.escape(value)}\b", lowered):
            return value
    return None


def _entropy_utility(labels: list[str | None]) -> float:
    known = [label for label in labels if label]
    if len(known) < 2:
        return 0.0
    counts = Counter(known)
    if len(counts) < 2:
        return 0.0
    entropy = -sum((count / len(known)) * math.log(count / len(known)) for count in counts.values())
    normalized = entropy / math.log(min(len(counts), 5))
    coverage = len(known) / len(labels)
    return min(1.0, normalized * coverage)


def _feature_utility(rows: Sequence[RankedCandidate]) -> float:
    documents: list[set[str]] = []
    for row in rows:
        tokens = {
            token for token in re.findall(r"[a-z0-9]+", row.candidate.features.casefold())
            if len(token) >= 4 and token not in FEATURE_STOPWORDS
        }
        documents.append(tokens)
    if len(documents) < 3:
        return 0.0
    frequencies = Counter(token for document in documents for token in document)
    fractions = [count / len(documents) for count in frequencies.values() if 1 < count < len(documents)]
    if not fractions:
        return 0.0
    best_split = min(fractions, key=lambda value: abs(0.5 - value))
    return 4.0 * best_split * (1.0 - best_split)


def _feature_labels(rows: Sequence[RankedCandidate]) -> list[str | None]:
    documents: list[set[str]] = []
    for row in rows:
        tokens = {
            token for token in re.findall(r"[a-z0-9]+", row.candidate.features.casefold())
            if len(token) >= 4 and token not in FEATURE_STOPWORDS
        }
        documents.append(tokens)
    if len(documents) < 3:
        return [None] * len(documents)
    frequencies = Counter(token for document in documents for token in document)
    viable = [(token, count / len(documents)) for token, count in frequencies.items()
              if 1 < count < len(documents)]
    if not viable:
        return [None] * len(documents)
    token, _ = min(viable, key=lambda item: (abs(0.5 - item[1]), item[0]))
    return [f"has:{token}" if token in document else f"not:{token}" for document in documents]


def _facet_or_text_label(row: RankedCandidate, attribute: str, vocabulary: Sequence[str]) -> str | None:
    facet_values = row.candidate.facets.values(attribute)
    if facet_values:
        return facet_values[0]
    return _first_term(row.candidate.attribute_text(attribute), vocabulary)


def _candidate_labels(rows: Sequence[RankedCandidate]) -> dict[str, list[str | None]]:
    shortlist = list(rows[:40])
    prices = [row.candidate.price for row in shortlist if row.candidate.price is not None]
    price_midpoint = median(prices) if prices else None
    return {
        "material": [_facet_or_text_label(row, "material", MATERIALS) for row in shortlist],
        "color": [_facet_or_text_label(row, "color", COLORS) for row in shortlist],
        "style": [_facet_or_text_label(row, "style", STYLES) for row in shortlist],
        "use_case": [_facet_or_text_label(row, "use_case", USE_CASES) for row in shortlist],
        "size": [
            (row.candidate.facets.sizes[0] if row.candidate.facets.sizes else
             match.group(0).casefold() if (match := re.search(
                 r"\b(?:wide|narrow|small|medium|large|xl|xxl)\b",
                 row.candidate.attribute_text("size"), re.I,
             )) else None)
            for row in shortlist
        ],
        "brand": [
            row.candidate.facets.brands[0] if row.candidate.facets.brands
            else row.candidate.store.casefold().strip()[:50] or None
            for row in shortlist
        ],
        "category": [
            row.candidate.facets.categories[-1] if row.candidate.facets.categories
            else row.candidate.categories.casefold().strip()[-60:] or None
            for row in shortlist
        ],
        "budget": [
            None if row.candidate.price is None or price_midpoint is None
            else "lower" if row.candidate.price <= price_midpoint else "higher"
            for row in shortlist
        ],
        "feature": _feature_labels(shortlist),
    }


def _counterfactual_utility(labels: list[str | None], rows: Sequence[RankedCandidate]) -> float:
    """Expected score gain if the hidden target's attribute value were revealed."""

    if len(labels) < 11 or len(set(label for label in labels if label)) < 2:
        return 0.0
    # The question matters only if the simultaneous recommendations miss, so place
    # the target prior on candidates currently outside Top 10. Blend rank evidence
    # with a uniform prior to avoid assuming formula scores are probabilities.
    tail = [(index, label) for index, label in enumerate(labels) if index >= 10 and label]
    if not tail:
        return 0.0
    relevance = [1.0 / math.log2(index + 2) for index, _ in tail]
    relevance_total = sum(relevance)
    gains = 0.0
    for tail_index, ((index, label), rank_weight) in enumerate(zip(tail, relevance)):
        prior = 0.70 / len(tail) + 0.30 * rank_weight / relevance_total
        new_rank = 1 + sum(earlier == label for earlier in labels[:index])
        hit_gain = 1.0 if new_rank <= 10 else 0.0
        reciprocal_gain = max(0.0, 1.0 / new_rank - 1.0 / (index + 1))
        gains += prior * (0.75 * hit_gain + 0.25 * reciprocal_gain)
    coverage = len(tail) / max(1, len(labels) - 10)
    return min(1.0, gains * coverage)


def _candidate_information(rows: Sequence[RankedCandidate]) -> dict[str, float]:
    shortlist = list(rows[:40])
    if not shortlist:
        return {attribute: 0.0 for attribute in FALLBACK_ORDER}
    labels = _candidate_labels(shortlist)
    return {attribute: _entropy_utility(values) for attribute, values in labels.items()}


def question_utilities(
    state: ConversationState,
    ranked: Sequence[RankedCandidate],
    use_counterfactual: bool = True,
) -> dict[str, float]:
    """Estimate question value from candidate separation and likely answerability."""

    shortlist = list(ranked[:40])
    labels = _candidate_labels(shortlist) if shortlist else {}
    information = {
        attribute: _entropy_utility(values) for attribute, values in labels.items()
    }
    counterfactual = {
        attribute: _counterfactual_utility(values, shortlist)
        for attribute, values in labels.items()
    }
    known = state.known_attributes()
    profile_tags = {str(tag).casefold() for tag in state.user_profile.get("preference_tags", [])}
    utilities: dict[str, float] = {}
    for attribute in FALLBACK_ORDER:
        if (attribute in known or attribute in state.exhausted_attributes
                or state.asked_counts.get(attribute, 0) >= 1):
            continue
        profile_boost = 0.0
        if profile_tags.intersection(PROFILE_ATTRIBUTE_PRIORS.get(attribute, ())):
            profile_boost = 0.08
        if use_counterfactual:
            utility = (
                0.46 * counterfactual.get(attribute, 0.0)
                + 0.34 * information.get(attribute, 0.0)
                + 0.20 * ANSWER_PRIOR[attribute]
                + profile_boost
            )
        else:
            utility = 0.72 * information.get(attribute, 0.0) + 0.28 * ANSWER_PRIOR[attribute] + profile_boost
        utilities[attribute] = round(min(1.0, utility), 6)
    return utilities


def decide(
    state: ConversationState,
    ranked: Sequence[RankedCandidate],
    use_counterfactual: bool = True,
) -> Clarification:
    if state.turn >= 10:
        return Clarification(None, "Here are the closest matches based on your requirements.", "turn_limit")

    conflicts = state.conflicting_attributes()
    if conflicts:
        attribute = next((item for item in FALLBACK_ORDER if item in conflicts), sorted(conflicts)[0])
        return Clarification(attribute, QUESTIONS[attribute], "resolve_contradiction")

    # `other` is uniquely valuable in the released protocol: one answer can disclose
    # two still-hidden constraints. Candidate-aware selection takes over afterward.
    other_limit = 3 if state.boundary_seen else 2
    other_count = state.asked_counts.get("other", 0)
    if "other" not in state.exhausted_attributes and other_count < other_limit:
        return Clarification("other", QUESTIONS["other"], "collect_high_value_constraints")

    utilities = question_utilities(state, ranked, use_counterfactual)
    if not ranked and "category" in utilities:
        return Clarification("category", QUESTIONS["category"], "recover_empty_retrieval", utilities)
    if utilities:
        order = {attribute: index for index, attribute in enumerate(FALLBACK_ORDER)}
        attribute = min(utilities, key=lambda item: (-utilities[item], order[item]))
        reason = "counterfactual_question_value" if use_counterfactual else "candidate_information_gain"
        if state.rejected_result_sets:
            reason = "strategy_switch_after_rejection"
        return Clarification(attribute, QUESTIONS[attribute], reason, utilities)
    return Clarification(None, "Here are the closest matches based on your requirements.", "attributes_exhausted")
