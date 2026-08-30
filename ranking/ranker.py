from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Sequence

from conversation import BudgetRange, Constraint, ConversationState
from retrieval import Candidate
from .ltr import RankingModel
from .semantic import SemanticReranker


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokens(value: str) -> list[str]:
    return TOKEN_RE.findall(value.casefold())


def _phrase_score(needle: str, haystack: str) -> float:
    wanted, actual = _tokens(needle), _tokens(haystack)
    if not wanted:
        return 0.0
    if any(actual[index:index + len(wanted)] == wanted for index in range(len(actual) - len(wanted) + 1)):
        return 1.0
    actual_set = set(actual)
    return 0.65 if all(token in actual_set for token in wanted) else 0.0


def _constraint_score(
    constraint: Constraint,
    candidate: Candidate,
    use_facet_evidence: bool = False,
) -> tuple[float, bool]:
    if constraint.attribute == "budget" and isinstance(constraint.value, BudgetRange):
        if candidate.price is None:
            return 0.0, False
        below = constraint.value.minimum is not None and candidate.price < constraint.value.minimum
        above = constraint.value.maximum is not None and candidate.price > constraint.value.maximum
        if below or above:
            return 0.0, True
        if constraint.value.target is None or constraint.value.target <= 0:
            return 1.0, False
        distance = abs(candidate.price - constraint.value.target) / constraint.value.target
        return max(0.6, 1.0 - distance), False
    if constraint.attribute == "budget" and isinstance(constraint.value, (int, float)):
        if candidate.price is None:
            return 0.0, False
        matched = candidate.price <= float(constraint.value) * 1.10
        return float(matched), not matched
    if use_facet_evidence and constraint.attribute in {
        "color", "material", "size", "style", "brand", "use_case",
    }:
        values = candidate.facets.values(constraint.attribute)
        if values:
            matched = max((_phrase_score(str(constraint.value), value) for value in values), default=0.0)
            return matched, matched == 0.0
    attribute_text = candidate.attribute_text(constraint.attribute) if use_facet_evidence else candidate.searchable_text
    phrase = _phrase_score(constraint.phrase, attribute_text)
    value = _phrase_score(str(constraint.value), attribute_text)
    score = max(phrase, value * 0.8)
    return score, False


@dataclass(frozen=True)
class RankedCandidate:
    candidate: Candidate
    score: float
    diagnostics: dict[str, Any]


def _normalized(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high == low:
        return [1.0 if value > 0 else 0.0 for value in values]
    return [(value - low) / (high - low) for value in values]


def rank_candidates(
    state: ConversationState,
    candidates: Sequence[Candidate],
    semantic_reranker: SemanticReranker | None = None,
    rerank_limit: int = 30,
    use_facet_evidence: bool = False,
    ranking_model: RankingModel | None = None,
    ltr_blend: float = 0.55,
) -> list[RankedCandidate]:
    if not candidates:
        return []
    raw = [math.log1p(item.bm25_score) for item in candidates]
    lexical = _normalized(raw)
    fusion = _normalized([item.fusion_score for item in candidates])
    fusion_enabled = any(item.semantic_rank is not None or item.route_ranks for item in candidates)
    positive = [item for item in state.constraints if item.kind != "negative"]
    negative = [item for item in state.constraints if item.kind == "negative"]
    profile_tags = [str(item) for item in state.user_profile.get("preference_tags", [])]
    category_tokens = set(_tokens(state.category))
    shown_ids = state.shown_ids()
    max_rating_count = max((item.rating_number or 0.0 for item in candidates), default=0.0)
    results: list[RankedCandidate] = []
    for item, lexical_score, fusion_score in zip(candidates, lexical, fusion):
        evidence = [_constraint_score(constraint, item, use_facet_evidence) for constraint in positive]
        hard_scores = [score for constraint, (score, _) in zip(positive, evidence) if constraint.kind == "hard"]
        soft_scores = [score for constraint, (score, _) in zip(positive, evidence) if constraint.kind == "soft"]
        hard_score = sum(hard_scores) / len(hard_scores) if hard_scores else 0.0
        soft_score = sum(soft_scores) / len(soft_scores) if soft_scores else 0.0
        constraint_floor = min(hard_scores) if hard_scores else 0.0
        exact_constraint_ratio = (
            sum(score >= 0.8 for score in hard_scores) / len(hard_scores) if hard_scores else 0.0
        )
        violations = sum(violation for _, violation in evidence)
        negative_hits = sum(
            _constraint_score(constraint, item, use_facet_evidence)[0] > 0 for constraint in negative
        )
        title_tokens = set(_tokens(item.title))
        category_score = len(category_tokens & title_tokens) / len(category_tokens) if category_tokens else 0.0
        profile_score = (sum(_phrase_score(tag, item.searchable_text) > 0 for tag in profile_tags)
                         / len(profile_tags) if profile_tags else 0.0)
        rating = max(0.0, min(1.0, (item.average_rating or 0.0) / 5.0))
        popularity = (
            math.log1p(item.rating_number or 0.0) / math.log1p(max_rating_count)
            if max_rating_count > 0 else 0.0
        )
        rating_quality = rating * (0.75 + 0.25 * popularity)
        facets = item.facets
        facet_coverage = sum(bool(value) for value in (
            facets.colors, facets.materials, facets.sizes, facets.styles,
            facets.use_cases, facets.features, facets.categories, facets.brands,
        )) / 8.0
        retrieval_score = 0.65 * lexical_score + 0.35 * fusion_score if fusion_enabled else lexical_score
        already_shown = item.parent_asin in shown_ids
        if positive:
            formula_score = (0.44 * retrieval_score + 0.50 * hard_score + 0.03 * soft_score
                             + 0.02 * category_score + 0.01 * profile_score
                             - 0.45 * violations - 0.55 * negative_hits)
        else:
            formula_score = retrieval_score
        score = formula_score
        if already_shown:
            # Under the official protocol, another turn means the target was not in
            # the previous scored Top 10. An override starts a fresh intent version.
            score -= 2.0
        results.append(RankedCandidate(item, score, {
            "lexical": lexical_score,
            "hybrid_fusion": fusion_score if fusion_enabled else None,
            "semantic_retrieval": item.semantic_score,
            "hard_constraint": hard_score,
            "soft_constraint": soft_score,
            "constraint_floor": constraint_floor,
            "exact_constraint_ratio": exact_constraint_ratio,
            "category_title": category_score,
            "profile": profile_score,
            "rating_quality": rating_quality,
            "facet_coverage": facet_coverage,
            "price_known": item.price is not None,
            "hard_violations": violations,
            "negative_hits": negative_hits,
            "already_shown": already_shown,
            "retrieval_rank": item.retrieval_rank,
            "retrieval_reciprocal": 1.0 / item.retrieval_rank,
            "base_formula": formula_score,
            "ltr_score": None,
            "semantic_rerank": None,
        }))
    results.sort(key=lambda row: (-row.score, row.candidate.retrieval_rank, row.candidate.parent_asin))

    if ranking_model is not None:
        if not 0.0 <= ltr_blend <= 1.0:
            raise ValueError("ltr_blend must be between zero and one")
        model_values = [float(value) for value in ranking_model.score(
            [row.diagnostics for row in results]
        )]
        if len(model_values) != len(results):
            raise ValueError("Ranking model returned the wrong number of scores")
        normalized_model = _normalized(model_values)
        normalized_formula = _normalized([row.score for row in results])
        learned: list[RankedCandidate] = []
        for row, model_score, formula_score in zip(results, normalized_model, normalized_formula):
            diagnostics = {**row.diagnostics, "ltr_score": model_score}
            safe_score = ltr_blend * model_score + (1.0 - ltr_blend) * formula_score
            safe_score -= 0.45 * float(row.diagnostics["hard_violations"])
            safe_score -= 0.55 * float(row.diagnostics["negative_hits"])
            safe_score -= 2.0 * float(bool(row.diagnostics["already_shown"]))
            learned.append(RankedCandidate(row.candidate, safe_score, diagnostics))
        learned.sort(key=lambda row: (-row.score, row.candidate.retrieval_rank,
                                      row.candidate.parent_asin))
        results = learned

    if semantic_reranker is not None and state.query and results:
        shortlist_size = min(max(1, rerank_limit), len(results))
        shortlist = results[:shortlist_size]
        semantic_values = list(semantic_reranker.score(
            state.query, [row.candidate for row in shortlist]
        ))
        if len(semantic_values) != shortlist_size:
            raise ValueError("Semantic reranker returned the wrong number of scores")
        normalized_semantic = _normalized([float(value) for value in semantic_values])
        reranked: list[RankedCandidate] = []
        for row, semantic_score in zip(shortlist, normalized_semantic):
            diagnostics = {**row.diagnostics, "semantic_rerank": semantic_score}
            # Semantic relevance may refine a shortlist, but it cannot erase an
            # explicit contradiction or budget violation.
            safe_bonus = 0.18 * semantic_score
            reranked.append(RankedCandidate(row.candidate, row.score + safe_bonus, diagnostics))
        reranked.sort(key=lambda row: (-row.score, row.candidate.retrieval_rank,
                                       row.candidate.parent_asin))
        results = reranked + results[shortlist_size:]
    return results
