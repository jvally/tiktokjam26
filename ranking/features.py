"""Explainable features; missing values are distinct from explicit violations."""

from typing import Any, Sequence

from shared import Candidate, Product, SearchState
from shared.text import phrase_match, tokens

CONSTRAINT_WEIGHTS = {"budget_min": 3.0, "budget_max": 3.0, "category": 3.0,
                      "brand": 2.0, "color": 2.0, "material": 2.0, "size": 2.0}
FIELD_NAMES = {"feature": "features", "use_case": "use_cases"}
# This is a deliberately small lexical fallback, NOT a semantic/embedding model.
LEXICAL_EXPANSIONS = {"comfortable": ["cushioned", "cushioning", "padded"],
                      "travelling": ["travel", "walking"],
                      "traveling": ["travel", "walking"],
                      "breathable": ["mesh", "ventilated"]}


def normalize_bm25(candidates: Sequence[Candidate]) -> list[float]:
    known = [c.bm25_score for c in candidates if c.bm25_score is not None]
    if not known:
        return [0.0] * len(candidates)
    low, high = min(known), max(known)
    if low == high:
        return [float(c.bm25_score is not None and high > 0) for c in candidates]
    return [0.0 if c.bm25_score is None else (c.bm25_score - low) / (high - low)
            for c in candidates]


def observed_values(product: Product, key: str) -> list[str]:
    value = getattr(product, FIELD_NAMES.get(key, key), [])
    return [value] if isinstance(value, str) and value else (value if isinstance(value, list) else [])


def match_value(expected: str, observed: str, key: str) -> bool:
    if key == "size":
        return bool(tokens(expected)) and tokens(expected) == tokens(observed)
    alternatives = [expected]
    if key in {"feature", "use_case"}:
        alternatives += LEXICAL_EXPANSIONS.get(expected.casefold(), [])
    return any(phrase_match(value, observed) for value in alternatives)


def constraint_evidence(product: Product, requested: dict[str, Any],
                        negative: bool = False) -> list[dict[str, Any]]:
    evidence = []
    for key, expected in requested.items():
        weight = CONSTRAINT_WEIGHTS.get(key, 1.0)
        if key.startswith("budget_"):
            observed = product.price
            if observed is None:
                status = "unknown"
            else:
                matched = observed <= expected if key == "budget_max" else observed >= expected
                status = "match" if matched else "violation"
        else:
            observed = observed_values(product, key)
            matched = any(match_value(value, actual, key) for value in expected for actual in observed)
            # Only infer positive evidence from text when structured metadata is missing.
            if not observed:
                matched = any(match_value(value, product.text(), key) for value in expected)
            if negative:
                status = "violation" if matched else ("match" if observed else "unknown")
            else:
                status = "match" if matched else ("violation" if observed else "unknown")
        evidence.append({"attribute": key, "expected": expected, "observed": observed,
                         "status": status, "weight": weight, "negative": negative})
    return evidence


def summarize(evidence: list[dict[str, Any]]) -> tuple[float, float, float]:
    total = sum(item["weight"] for item in evidence)
    if not total:
        return 0.0, 0.0, 0.0
    matched = sum(item["weight"] for item in evidence if item["status"] == "match")
    violated = sum(item["weight"] for item in evidence if item["status"] == "violation")
    return matched / total, violated / total, (matched + violated) / total


def preference_score(product: Product, preferences: dict[str, Any]) -> float:
    return summarize(constraint_evidence(product, preferences))[0]


def extract_features(state: SearchState, candidates: Sequence[Candidate]) -> list[dict[str, Any]]:
    normalized = normalize_bm25(candidates)
    rows = []
    for candidate, bm25 in zip(candidates, normalized):
        product = candidate.product
        hard = constraint_evidence(product, state.hard_constraints)
        negative = constraint_evidence(product, state.negative_constraints, negative=True)
        constraint_score, violation_ratio, coverage = summarize(hard)
        preferences = {**state.soft_preferences, **state.hard_constraints}
        fields = {f"{key}_match": preference_score(product, {key: preferences[key]})
                  if key in preferences else 0.0
                  for key in ("category", "brand", "color", "material", "size", "style", "feature", "use_case")}
        semantic_values = [getattr(candidate, key) for key in
                           ("query_title_similarity", "query_feature_similarity", "use_case_similarity")
                           if getattr(candidate, key) is not None]
        dense = candidate.dense_score
        if dense is None and semantic_values:
            dense = sum(semantic_values) / len(semantic_values)
        metadata = candidate.metadata_score
        if metadata is None and preferences:
            metadata = preference_score(product, preferences)
        profile = candidate.profile_score
        if profile is None and state.profile:
            profile = preference_score(product, state.profile)
        soft = preference_score(product, state.soft_preferences)
        budget_evidence = [e for e in hard if e["attribute"].startswith("budget_")]
        features = {
            "bm25": bm25, "dense": dense or 0.0, "metadata": metadata or 0.0,
            "constraint": constraint_score, "preference": soft, "profile": profile or 0.0,
            "hybrid_retrieval": (bm25 + (dense or 0.0)) / (1 + int(dense is not None)),
            "budget_compatibility": summarize(budget_evidence)[0],
            "hard_violation_ratio": violation_ratio, "constraint_coverage": coverage,
            "negative_violations": float(sum(e["status"] == "violation" for e in negative)),
            "use_case_similarity": candidate.use_case_similarity or 0.0,
            "query_title_similarity": candidate.query_title_similarity or 0.0,
            "query_feature_similarity": candidate.query_feature_similarity or 0.0,
            **fields,
        }
        available = {
            "bm25": candidate.bm25_score is not None, "dense": dense is not None,
            "metadata": metadata is not None, "constraint": bool(state.hard_constraints),
            "preference": bool(state.soft_preferences), "profile": profile is not None,
        }
        rows.append({"ranking_features": features, "available": available,
                     "constraint_evidence": hard + negative})
    return rows
