import json
import math
import unittest

from ranking import rank_candidates
from ranking.features import normalize_bm25
from ranking.learning_to_rank import FEATURE_NAMES, feature_vector, split_by_session
from ranking.reranker import rerank_shortlist
from shared import Candidate, Product, SearchState


def candidate(asin, **kwargs):
    return {"parent_asin": asin, "title": "Generic product", **kwargs}


class RankingTests(unittest.TestCase):
    def test_constraint_match_beats_high_retrieval_violation(self):
        rows, _ = rank_candidates({"hard_constraints": {"color": "black", "budget_max": 100}}, [
            candidate("good", color="black", price=70, bm25_score=.4),
            candidate("bad", color="white", price=150, bm25_score=.9)])
        self.assertEqual(rows[0]["parent_asin"], "good")
        self.assertGreater(rows[1]["penalty"], 0)

    def test_unknown_is_not_violation(self):
        rows, _ = rank_candidates({"hard_constraints": {"color": "black", "budget_max": 100}}, [
            candidate("match", color="black", price=70), candidate("unknown"),
            candidate("violation", color="white", price=150)])
        self.assertEqual([r["parent_asin"] for r in rows], ["match", "unknown", "violation"])
        unknown = rows[1]
        self.assertEqual(unknown["penalty"], 0)
        self.assertTrue(all(e["status"] == "unknown" for e in unknown["constraint_evidence"]))

    def test_unknown_title_can_supply_positive_evidence(self):
        rows, _ = rank_candidates({"hard_constraints": {"color": "black"}}, [
            candidate("A", title="Black sneaker")])
        self.assertEqual(rows[0]["ranking_features"]["color_match"], 1)

    def test_negative_constraints_are_penalized(self):
        rows, _ = rank_candidates({"negative_constraints": {"material": "leather"}}, [
            candidate("leather", material="leather", bm25_score=10),
            candidate("mesh", material="mesh", bm25_score=10)])
        self.assertEqual(rows[0]["parent_asin"], "mesh")
        self.assertEqual(rows[1]["ranking_features"]["negative_violations"], 1)

    def test_word_boundaries_and_exact_size(self):
        rows, _ = rank_candidates({"hard_constraints": {"color": "red", "size": "8"}}, [
            candidate("A", title="Featured sneaker", size="18")])
        self.assertEqual(rows[0]["ranking_features"]["color_match"], 0)
        self.assertEqual(rows[0]["ranking_features"]["size_match"], 0)
        rows, _ = rank_candidates({"hard_constraints": {"size": "8"}}, [candidate("B", size="8.5")])
        self.assertEqual(rows[0]["ranking_features"]["size_match"], 0)

    def test_positive_and_negative_alternative_values(self):
        rows, _ = rank_candidates({"hard_constraints": {"color": ["blue", "black"]},
                                   "negative_constraints": {"material": ["leather", "wool"]}}, [
            candidate("A", color="black", material="cotton")])
        self.assertEqual(rows[0]["ranking_features"]["constraint"], 1)
        self.assertEqual(rows[0]["ranking_features"]["negative_violations"], 0)

    def test_no_constraints_no_bonus(self):
        rows, diagnostics = rank_candidates({}, [candidate("A", bm25_score=2)])
        self.assertEqual(rows[0]["ranking_features"]["constraint"], 0)
        self.assertEqual(diagnostics["effective_weights"], {"bm25": 1})

    def test_partial_missing_scores_use_batch_weights(self):
        rows, diagnostics = rank_candidates({}, [candidate("known", bm25_score=1, dense_score=1),
                                                   candidate("missing", bm25_score=1)])
        self.assertEqual(rows[0]["parent_asin"], "known")
        self.assertEqual(diagnostics["effective_weights"], {"bm25": .5, "dense": .5})

    def test_flat_and_missing_bm25(self):
        product = Product("A", "Shoe")
        self.assertEqual(normalize_bm25([Candidate(product, bm25_score=2), Candidate(product)]), [1, 0])
        self.assertEqual(normalize_bm25([Candidate(product, bm25_score=0)]), [0])

    def test_lexical_preference_and_provided_semantic_features(self):
        rows, _ = rank_candidates({"soft_preferences": {"feature": "comfortable", "use_case": "travelling"}}, [
            candidate("A", title="Cushioned walking sneaker", query_title_similarity=.8, query_feature_similarity=.6)])
        self.assertEqual(rows[0]["ranking_features"]["preference"], 1)
        self.assertAlmostEqual(rows[0]["ranking_features"]["dense"], .7)

    def test_product_features_survive_ranking(self):
        rows, _ = rank_candidates({}, [candidate("A", features=["waterproof"])])
        self.assertEqual(rows[0]["features"], ["waterproof"])
        self.assertEqual(Product.from_dict(rows[0]).features, ["waterproof"])

    def test_deduplicated_ties_stable_and_no_input_mutation(self):
        candidates = [candidate("B", bm25_score=1), candidate("A", bm25_score=1), candidate("A", bm25_score=0)]
        original = json.dumps(candidates)
        rows, diag = rank_candidates({}, candidates)
        self.assertEqual([r["parent_asin"] for r in rows], ["A", "B"])
        self.assertEqual(diag["duplicate_count"], 1)
        self.assertEqual(json.dumps(candidates), original)

    def test_empty_singleton_and_full_diagnostics(self):
        rows, diag = rank_candidates({}, [])
        self.assertEqual(rows, [])
        self.assertIsNone(diag["top_score"])
        rows, diag = rank_candidates({}, [candidate("A")])
        self.assertIsNone(diag["score_gap_1_2"])
        self.assertIsNone(diag["score_gap_1_10"])
        self.assertEqual(diag["candidate_entropy"], 0)
        rows, diag = rank_candidates({}, [candidate(str(i), bm25_score=1) for i in range(12)], top_n=2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(diag["candidate_count"], 12)
        self.assertEqual(diag["score_gap_1_10"], 0)
        self.assertAlmostEqual(diag["normalized_entropy"], 1)
        json.dumps(diag, allow_nan=False)

    def test_intent_weights_and_custom_weights(self):
        candidates = [candidate("A", bm25_score=1, dense_score=.4)]
        _, buying = rank_candidates({"intent": "buying"}, candidates)
        _, browsing = rank_candidates({"intent": "browsing"}, candidates)
        self.assertGreater(browsing["effective_weights"]["dense"], buying["effective_weights"]["dense"])
        for weights in ({}, {"dense": 0}, {"bad": 1}, {"dense": -1}, {"dense": math.nan}):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                rank_candidates({}, candidates, weights=weights)
        for top_n in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                rank_candidates({}, candidates, top_n=top_n)

    def test_feature_export_and_grouped_split(self):
        rows, _ = rank_candidates({}, [candidate("A")])
        self.assertEqual(len(feature_vector(rows[0])), len(FEATURE_NAMES))
        examples = [{"session_id": f"session-{i // 3}", "row": i} for i in range(300)]
        train, validation = split_by_session(examples)
        self.assertTrue(train and validation)
        self.assertFalse({r["session_id"] for r in train} & {r["session_id"] for r in validation})
        self.assertEqual((train, validation), split_by_session(examples))

    def test_reranker_validates_ids_and_preserves_omissions(self):
        rows, _ = rank_candidates({}, [candidate("A"), candidate("B"), candidate("C")])

        class Reverse:
            def order(self, state, shortlist):
                return ["B"]

        reranked = rerank_shortlist(SearchState(), rows, Reverse(), limit=2)
        self.assertEqual([r["parent_asin"] for r in reranked], ["B", "A", "C"])
        self.assertEqual([r["rank"] for r in reranked], [1, 2, 3])

        class Invalid:
            def order(self, state, shortlist):
                return ["invented"]

        with self.assertRaises(ValueError):
            rerank_shortlist(SearchState(), rows, Invalid())
        with self.assertRaises(ValueError):
            rerank_shortlist(SearchState(), rows, Reverse(), limit=21)
