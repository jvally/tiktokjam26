from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

from conversation import ConversationState, update_from_message
from policy import decide
from ranking import (
    CrossEncoderReranker,
    LinearLTRModel,
    RankingModel,
    SemanticReranker,
    rank_candidates,
)
from retrieval import CatalogIndex
from retrieval.semantic import SemanticRetriever, SentenceTransformerIndex


class Agent:
    """Offline five-stage shopping agent implementing the organizer contract."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        semantic_retriever: SemanticRetriever | None = None,
        semantic_reranker: SemanticReranker | None = None,
        semantic_lexical_weight: float = 0.75,
        semantic_pool_multiplier: int = 2,
        semantic_minimum_pool: int = 300,
        candidate_limit: int = 300,
        initial_candidate_limit: int = 160,
        enable_query_expansion: bool = False,
        enable_multi_route: bool = False,
        use_facet_evidence: bool = True,
        use_persistent_cache: bool = True,
        use_counterfactual_questions: bool = False,
        ranking_model: RankingModel | None = None,
        ltr_blend: float = 0.90,
        use_default_ltr: bool = True,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        if semantic_retriever is None and os.environ.get("TECHJAM_SEMANTIC_INDEX"):
            try:
                semantic_retriever = SentenceTransformerIndex(
                    index_directory=os.environ["TECHJAM_SEMANTIC_INDEX"],
                    model_name=os.environ.get("TECHJAM_EMBEDDING_MODEL"),
                    model_revision=os.environ.get("TECHJAM_EMBEDDING_REVISION"),
                    catalog_path=self.catalog_path,
                )
            except Exception as exc:
                warnings.warn(f"Could not load semantic index; using FTS5 only: {exc}", RuntimeWarning)
        if semantic_reranker is None and os.environ.get("TECHJAM_RERANKER_MODEL"):
            try:
                semantic_reranker = CrossEncoderReranker(os.environ["TECHJAM_RERANKER_MODEL"])
            except Exception as exc:
                warnings.warn(f"Could not load semantic reranker; using formula ranking: {exc}", RuntimeWarning)
        if ranking_model is None and os.environ.get("TECHJAM_LTR_WEIGHTS"):
            try:
                ranking_model = LinearLTRModel.load(os.environ["TECHJAM_LTR_WEIGHTS"])
            except Exception as exc:
                warnings.warn(f"Could not load LTR model; using formula ranking: {exc}", RuntimeWarning)
        if ranking_model is None and use_default_ltr:
            packaged_model = Path(__file__).resolve().parent.parent / "ranking" / "models" / "ltr_v1.json"
            if packaged_model.exists():
                try:
                    ranking_model = LinearLTRModel.load(packaged_model)
                except Exception as exc:
                    warnings.warn(f"Could not load packaged LTR model; using formula ranking: {exc}", RuntimeWarning)
        self.index = CatalogIndex(
            self.catalog_path,
            semantic_retriever=semantic_retriever,
            semantic_lexical_weight=semantic_lexical_weight,
            semantic_pool_multiplier=semantic_pool_multiplier,
            semantic_minimum_pool=semantic_minimum_pool,
            expand_queries=enable_query_expansion,
            multi_route=enable_multi_route,
            use_persistent_cache=use_persistent_cache,
        )
        self.semantic_reranker = semantic_reranker
        self.ranking_model = ranking_model
        self.ltr_blend = ltr_blend
        self.use_facet_evidence = use_facet_evidence
        self.use_counterfactual_questions = use_counterfactual_questions
        self.candidate_limit = candidate_limit
        self.initial_candidate_limit = min(initial_candidate_limit, candidate_limit)
        self._sessions: dict[str, ConversationState] = {}
        self._last_diagnostics: dict[str, dict[str, Any]] = {}
        self._last_rank_features: tuple[str, list[dict[str, Any]]] | None = None

    def reset(self, session_id: str, user_profile: dict) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a nonempty string")
        if not isinstance(user_profile, dict):
            raise ValueError("user_profile must be an object")
        self._sessions[session_id] = ConversationState(dict(user_profile))
        self._last_diagnostics.pop(session_id, None)

    def _retrieval_limit(self, state: ConversationState) -> int:
        has_hard_constraint = any(item.kind == "hard" for item in state.constraints)
        if state.turn == 1 and not has_hard_constraint and not state.rejected_result_sets:
            return self.initial_candidate_limit
        return self.candidate_limit

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict[str, Any]:
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise ValueError("top_k must be an integer from 1 to 100")
        update_from_message(state, user_message, turn)
        retrieval_limit = self._retrieval_limit(state)
        candidates = self.index.retrieve(state, limit=retrieval_limit)
        try:
            ranked = rank_candidates(
                state,
                candidates,
                self.semantic_reranker,
                rerank_limit=30,
                use_facet_evidence=self.use_facet_evidence,
                ranking_model=self.ranking_model,
                ltr_blend=self.ltr_blend,
            )
        except Exception as exc:
            if self.semantic_reranker is None and self.ranking_model is None:
                raise
            warnings.warn(f"Optional ranking failed; using formula ranking: {exc}", RuntimeWarning)
            self.semantic_reranker = None
            self.ranking_model = None
            ranked = rank_candidates(state, candidates, use_facet_evidence=self.use_facet_evidence)
        clarification = decide(state, ranked, self.use_counterfactual_questions)
        state.note_question(clarification.attribute)
        recommendations = [
            {"parent_asin": row.candidate.parent_asin}
            for row in ranked[:top_k]
        ]
        self._last_rank_features = (session_id, [
            {"parent_asin": row.candidate.parent_asin, **row.diagnostics}
            for row in ranked
        ])
        state.note_recommendations([item["parent_asin"] for item in recommendations])
        if any(item.semantic_rank is not None for item in candidates):
            retrieval_mode = "hybrid"
        elif any(item.route_ranks for item in candidates):
            retrieval_mode = "multi_route"
        else:
            retrieval_mode = "lexical"
        self._last_diagnostics[session_id] = {
            "turn": state.turn,
            "intent": state.intent,
            "intent_version": state.intent_version,
            "query": state.query,
            "known_attributes": sorted(state.known_attributes()),
            "candidate_limit": retrieval_limit,
            "candidate_count": len(candidates),
            "candidate_ids": [item.parent_asin for item in candidates],
            "ranked_ids": [row.candidate.parent_asin for row in ranked],
            "retrieval_mode": retrieval_mode,
            "ranking_mode": "ltr_formula_blend" if self.ranking_model is not None else "formula",
            "facet_evidence": self.use_facet_evidence,
            "counterfactual_questions": self.use_counterfactual_questions,
            "index_cache_hit": self.index.cache_hit,
            "query_cache_hit": self.index.last_query_cache_hit,
            "top_score_gap": (
                ranked[0].score - ranked[1].score if len(ranked) > 1 else None
            ),
            "clarification_reason": clarification.reason,
            "question_utilities": clarification.utilities,
        }
        return {
            "message": clarification.message,
            "ask_attribute": clarification.attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    def get_diagnostics(self, session_id: str) -> dict[str, Any]:
        """Return development-only trace data kept out of official responses."""

        trace = self._last_diagnostics.get(session_id)
        if trace is None:
            raise KeyError(f"No completed turn for session {session_id!r}")
        return {
            **trace,
            "candidate_ids": list(trace["candidate_ids"]),
            "ranked_ids": list(trace["ranked_ids"]),
            "known_attributes": list(trace["known_attributes"]),
            "question_utilities": dict(trace["question_utilities"]),
        }

    def get_rank_features(self, session_id: str) -> list[dict[str, Any]]:
        """Return the most recent target-free feature matrix for offline LTR training."""

        if self._last_rank_features is None or self._last_rank_features[0] != session_id:
            raise KeyError(f"No rank features for session {session_id!r}")
        return [dict(row) for row in self._last_rank_features[1]]
