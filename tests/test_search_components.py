from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from conversation import BudgetRange, Constraint, ConversationState
from policy import decide, question_utilities
from ranking import rank_candidates
from retrieval import Candidate, CatalogIndex, ProductFacets
from retrieval.semantic import SemanticHit


def candidate(asin: str, title: str, *, price: float | None = None, rank: int = 1) -> Candidate:
    return Candidate(
        parent_asin=asin, title=title, categories="Shoes", features=title,
        details="", store="", description="", price=price,
        average_rating=4.0, rating_number=10, bm25_score=1 / rank, retrieval_rank=rank,
    )


class RetrievalTests(unittest.TestCase):
    def test_weighted_fts_returns_target_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            rows = [
                {"parent_asin": "BLUE", "title": "Blue running shoe", "features": ["waterproof"], "price": 50},
                {"parent_asin": "RED", "title": "Red dress shoe", "features": ["formal"], "price": 80},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            index = CatalogIndex(path)
            try:
                state = ConversationState({}, category="running shoe")
                results = index.retrieve(state, 2)
                self.assertEqual(results[0].parent_asin, "BLUE")
                self.assertEqual(results[0].price, 50)
            finally:
                index.close()

    def test_multi_route_records_field_specific_ranks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            rows = [
                {"parent_asin": "CORE", "title": "Running shoe", "features": ["basic"]},
                {"parent_asin": "FEATURE", "title": "Shoe", "features": ["waterproof running"]},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            index = CatalogIndex(path, multi_route=True)
            try:
                state = ConversationState({}, category="running shoe")
                state.constraints.append(Constraint("feature", "waterproof", "waterproof", "hard", 1))
                results = index.retrieve(state, 2)
                self.assertTrue(any(item.route_ranks.get("core") for item in results))
                self.assertTrue(any(item.route_ranks.get("constraints") for item in results))
            finally:
                index.close()

    def test_second_index_uses_catalog_fingerprint_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            path.write_text(json.dumps({"parent_asin": "A", "title": "Shoe"}) + "\n", encoding="utf-8")
            first = CatalogIndex(path)
            self.assertFalse(first.cache_hit)
            first.close()
            second = CatalogIndex(path)
            try:
                self.assertTrue(second.cache_hit)
            finally:
                second.close()

    def test_repeated_query_uses_bounded_lexical_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            path.write_text(json.dumps({"parent_asin": "A", "title": "Running shoe"}) + "\n", encoding="utf-8")
            index = CatalogIndex(path)
            try:
                state = ConversationState({}, category="running shoe")
                index.retrieve(state, 10)
                self.assertFalse(index.last_query_cache_hit)
                index.retrieve(state, 10)
                self.assertTrue(index.last_query_cache_hit)
            finally:
                index.close()

    def test_multi_route_cache_includes_normalized_constraint_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            rows = [
                {"parent_asin": "WATER", "title": "Water proof shoe"},
                {"parent_asin": "MESH", "title": "Breathable mesh shoe"},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            index = CatalogIndex(path, multi_route=True)
            try:
                state = ConversationState({}, category="shoe")
                state.constraints = [Constraint("feature", "water proof", "feature", "hard", 1)]
                self.assertEqual(index.retrieve(state, 2)[0].parent_asin, "WATER")
                state.constraints = [Constraint("feature", "breathable mesh", "feature", "hard", 2)]
                self.assertEqual(index.retrieve(state, 2)[0].parent_asin, "MESH")
                self.assertFalse(index.last_query_cache_hit)
            finally:
                index.close()

    def test_hybrid_retrieval_includes_semantic_only_candidate(self) -> None:
        class FakeSemanticRetriever:
            def retrieve(self, query: str, limit: int) -> list[SemanticHit]:
                return [SemanticHit("SEMANTIC", 0.9)]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            rows = [
                {"parent_asin": "LEXICAL", "title": "Running shoe"},
                {"parent_asin": "SEMANTIC", "title": "Jogging footwear"},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            index = CatalogIndex(path, semantic_retriever=FakeSemanticRetriever())
            try:
                results = index.retrieve(ConversationState({}, category="running"), 2)
                semantic = next(item for item in results if item.parent_asin == "SEMANTIC")
                self.assertEqual(semantic.semantic_rank, 1)
                self.assertIsNone(semantic.lexical_rank)
            finally:
                index.close()


class RankingTests(unittest.TestCase):
    def test_exact_constraint_match_beats_lexical_leader(self) -> None:
        state = ConversationState({}, category="shoe")
        state.constraints.append(Constraint("color", "blue", "color: blue", "hard", 1))
        ranked = rank_candidates(state, [
            candidate("RED", "Red shoe", rank=1),
            candidate("BLUE", "Blue shoe color blue", rank=2),
        ])
        self.assertEqual(ranked[0].candidate.parent_asin, "BLUE")
        self.assertGreater(ranked[0].diagnostics["hard_constraint"], 0)

    def test_negative_and_budget_constraints_penalize(self) -> None:
        state = ConversationState({}, category="shoe")
        state.constraints.extend([
            Constraint("budget", 60.0, "budget under $60", "hard", 1),
            Constraint("material", "leather", "leather", "negative", 1),
        ])
        ranked = rank_candidates(state, [
            candidate("BAD", "Leather shoe", price=100, rank=1),
            candidate("GOOD", "Canvas shoe", price=50, rank=2),
        ])
        self.assertEqual(ranked[0].candidate.parent_asin, "GOOD")

    def test_budget_range_respects_both_bounds(self) -> None:
        state = ConversationState({}, category="shoe")
        state.constraints.append(Constraint(
            "budget", BudgetRange(minimum=50, maximum=80), "between $50 and $80", "hard", 1
        ))
        ranked = rank_candidates(state, [
            candidate("CHEAP", "Shoe", price=30, rank=1),
            candidate("IN_RANGE", "Shoe", price=65, rank=2),
        ])
        self.assertEqual(ranked[0].candidate.parent_asin, "IN_RANGE")

    def test_semantic_reranker_only_refines_shortlist(self) -> None:
        class FakeReranker:
            def score(self, query: str, candidates: list[Candidate]) -> list[float]:
                return [0.0, 10.0]

        state = ConversationState({}, category="shoe")
        ranked = rank_candidates(state, [
            candidate("A", "First shoe", rank=1),
            candidate("B", "Second shoe", rank=1),
        ], FakeReranker(), rerank_limit=2)
        self.assertEqual(ranked[0].candidate.parent_asin, "B")
        self.assertEqual(ranked[0].diagnostics["semantic_rerank"], 1.0)

    def test_previously_shown_products_rotate_out(self) -> None:
        state = ConversationState({}, category="shoe")
        state.note_recommendations(["A"])
        ranked = rank_candidates(state, [
            candidate("A", "First shoe", rank=1),
            candidate("B", "Second shoe", rank=2),
        ])
        self.assertEqual(ranked[0].candidate.parent_asin, "B")
        self.assertTrue(ranked[1].diagnostics["already_shown"])

    def test_structured_facets_override_misleading_free_text(self) -> None:
        state = ConversationState({}, category="shoe")
        state.constraints.append(Constraint("color", "blue", "blue", "hard", 1))
        misleading = candidate("TEXT", "Blue-looking shoe", rank=1)
        misleading = Candidate(**{
            **misleading.__dict__, "facets": ProductFacets(colors=("red",)),
        })
        structured = candidate("FACET", "Plain shoe", rank=2)
        structured = Candidate(**{
            **structured.__dict__, "facets": ProductFacets(colors=("blue",)),
        })
        ranked = rank_candidates(state, [misleading, structured], use_facet_evidence=True)
        self.assertEqual(ranked[0].candidate.parent_asin, "FACET")
        self.assertGreater(ranked[1].diagnostics["hard_violations"], 0)


class PolicyTests(unittest.TestCase):
    def test_other_is_repeated_to_collect_all_constraints(self) -> None:
        state = ConversationState({}, turn=1)
        first = decide(state, [])
        self.assertEqual(first.attribute, "other")
        state.note_question("other")
        state.turn = 2
        second = decide(state, [])
        self.assertEqual(second.attribute, "other")
        state.note_question("other")
        state.turn = 3
        self.assertNotEqual(decide(state, []).attribute, "other")

    def test_turn_ten_never_asks(self) -> None:
        state = ConversationState({}, turn=10)
        self.assertIsNone(decide(state, []).attribute)

    def test_candidate_information_selects_discriminating_attribute(self) -> None:
        state = ConversationState({}, turn=3)
        state.constraints.append(Constraint("feature", "waterproof", "waterproof", "hard", 2))
        state.asked_counts["other"] = 2
        ranked = rank_candidates(state, [
            candidate("C1", "Cotton shoe", rank=1),
            candidate("C2", "Cotton shoe", rank=2),
            candidate("L1", "Leather shoe", rank=3),
            candidate("L2", "Leather shoe", rank=4),
        ])
        utilities = question_utilities(state, ranked)
        self.assertGreater(utilities["material"], utilities["color"])
        decision = decide(state, ranked, use_counterfactual=True)
        self.assertEqual(decision.attribute, "material")
        self.assertEqual(decision.reason, "counterfactual_question_value")

    def test_contradiction_is_resolved_before_more_collection(self) -> None:
        state = ConversationState({}, turn=2)
        state.constraints.extend([
            Constraint("color", "blue", "blue", "hard", 1),
            Constraint("color", "blue", "blue", "negative", 2),
        ])
        decision = decide(state, [])
        self.assertEqual(decision.attribute, "color")
        self.assertEqual(decision.reason, "resolve_contradiction")


if __name__ == "__main__":
    unittest.main()
