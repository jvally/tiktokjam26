import json
import unittest

from evaluation import evaluate_cases, ranking_metrics
from policy import decide
from retrieval import LexicalRetriever, load_catalog
from shared import Product, SearchState
from shopping_copilot import ShoppingCopilot
from shopping_copilot.__main__ import DEMO_CATALOG, DEMO_CASES, read_json


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.products = load_catalog(str(DEMO_CATALOG))
        cls.retriever = LexicalRetriever(cls.products)
        cls.copilot = ShoppingCopilot(cls.retriever)

    def test_end_to_end_demo(self):
        response = self.copilot.search(read_json(DEMO_CASES)[0]["state"])
        self.assertEqual(len(response["parent_asins"]), 10)
        self.assertEqual(len(set(response["parent_asins"])), 10)
        self.assertEqual(response["parent_asins"][0], "DEMO001")
        self.assertEqual(response["diagnostics"]["candidate_count"], len(self.products))
        json.dumps(response, allow_nan=False)

    def test_empty_catalog(self):
        response = ShoppingCopilot(LexicalRetriever([])).search({"query": "shoes"})
        self.assertEqual(response["parent_asins"], [])
        self.assertEqual(response["decision"]["reason"], "no_candidates")

    def test_turn_10_never_asks(self):
        response = self.copilot.search({"query": "shoes", "turn": 10})
        self.assertTrue(response["decision"]["terminal"])
        self.assertIsNone(response["decision"]["question"])

    def test_policy_remembers_question(self):
        response = self.copilot.search({})
        self.assertEqual(response["decision"]["action"], "clarify")
        attribute = response["decision"]["attribute"]
        self.assertIn(attribute, response["state"]["asked_attributes"])
        next_response = self.copilot.search(response["state"])
        self.assertNotEqual(next_response["decision"]["attribute"], attribute)

    def test_singleton_not_claimed_confident(self):
        response = ShoppingCopilot(LexicalRetriever([Product("A", "Shoe")])).search({})
        self.assertNotEqual(response["decision"]["reason"], "clear_leader_heuristic")

    def test_negative_violation_prevents_clear_leader(self):
        ranked = [Product("A", "Black shoe", color=["black"]).to_dict(),
                  Product("B", "White shoe", color=["white"]).to_dict()]
        decision = decide(SearchState(), ranked, {"top_negative_violations": 1, "score_gap_1_2": .5})
        self.assertNotEqual(decision["reason"], "clear_leader_heuristic")
        self.assertEqual(decision["action"], "clarify")

    def test_200_candidate_pool_returns_ten_unique_results(self):
        products = [Product(f"P{i:03}", "Running shoe", price=50) for i in range(300)]
        response = ShoppingCopilot(LexicalRetriever(products)).search({"query": "running"})
        self.assertEqual(response["diagnostics"]["candidate_count"], 200)
        self.assertEqual(response["diagnostics"]["returned_count"], 20)
        self.assertEqual(len(response["parent_asins"]), 10)
        self.assertEqual(len(set(response["parent_asins"])), 10)

    def test_retrieval_is_bounded_deterministic_and_preserves_unknown(self):
        first = self.retriever.retrieve(SearchState(query="running"), limit=200)
        self.assertEqual([c.product.parent_asin for c in first],
                         [c.product.parent_asin for c in self.retriever.retrieve(SearchState(query="running"))])
        self.assertIn("DEMO017", [c.product.parent_asin for c in first])
        self.assertEqual(len(self.retriever.retrieve(SearchState(), limit=3)), 3)
        for bad_limit in (0, True, 1.5):
            with self.assertRaises(ValueError):
                self.retriever.retrieve(SearchState(), limit=bad_limit)

    def test_duplicate_catalog_ids_rejected(self):
        with self.assertRaises(ValueError):
            LexicalRetriever([Product("A", "Shoe"), Product("A", "Shirt")])

    def test_metrics_count_misses_and_top10_cutoff(self):
        metrics = ranking_metrics([["A", "B"], ["A"], [str(i) for i in range(11)]], ["B", "missing", "10"])
        self.assertAlmostEqual(metrics["hit@10"], 1 / 3)
        self.assertAlmostEqual(metrics["mrr@10"], 1 / 6)
        self.assertEqual(ranking_metrics([], [])["cases"], 0)
        with self.assertRaises(ValueError):
            ranking_metrics([[]], [])

    def test_replay_reports_retrieval_misses_without_dropping(self):
        report = evaluate_cases(self.retriever, [{"session_id": "missing", "state": {},
                                                 "target_parent_asin": "not-in-catalog"}])
        self.assertEqual(report["retrieval_recall"], 0)
        self.assertEqual(report["ranked_top10"]["mrr@10"], 0)
        self.assertEqual(report["ranked_top10"]["cases"], 1)
        with self.assertRaises(ValueError):
            evaluate_cases(self.retriever, [])

    def test_synthetic_smoke_evaluation(self):
        report = evaluate_cases(self.retriever, read_json(DEMO_CASES))
        self.assertEqual(report["ranked_top10"]["cases"], 5)
        self.assertEqual(report["ranked_top10"]["hit@10"], 1)
        self.assertGreaterEqual(report["mean_latency_ms"], 0)
