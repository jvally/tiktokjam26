from __future__ import annotations

import unittest
from types import SimpleNamespace

from evaluation import compare_results, summarize_turns
from evaluation.ltr_experiment import target_disjoint_split
from evaluation.retrieval_benchmark import evaluate_index
from ranking import train_pairwise


class EvaluationToolTests(unittest.TestCase):
    def test_compare_results_reports_directional_deltas(self) -> None:
        baseline = {
            "hit_rate_at_10": .1, "mrr": .05, "mttc": 10,
            "efficiency": .1, "recommended_technical_score": .085,
        }
        enhanced = {
            "hit_rate_at_10": .8, "mrr": .4, "mttc": 4,
            "efficiency": .7, "recommended_technical_score": .66,
        }
        result = compare_results(baseline, enhanced)
        self.assertAlmostEqual(result["delta_hit_rate_at_10"], .7)
        self.assertEqual(result["delta_mttc"], -6)

    def test_pipeline_summary_separates_retrieval_and_ranking(self) -> None:
        rows = [
            {
                "sample_id": "A", "target_retrieval_rank": 20, "target_ranked_rank": 2,
                "candidate_count": 100, "latency_ms": 5, "retrieval_mode": "lexical",
                "clarification_reason": "x",
            },
            {
                "sample_id": "B", "target_retrieval_rank": None, "target_ranked_rank": None,
                "candidate_count": 100, "latency_ms": 7, "retrieval_mode": "lexical",
                "clarification_reason": "y",
            },
        ]
        result = summarize_turns(rows)
        self.assertEqual(result["candidate_recall"]["50"], 0.5)
        self.assertEqual(result["ranker_hit_at_10_given_retrieved"], 1.0)

    def test_retrieval_benchmark_keeps_labels_outside_the_index_call(self) -> None:
        class FakeIndex:
            received_states: list[object] = []

            def retrieve(self, state: object, limit: int):
                self.received_states.append(state)
                return [SimpleNamespace(parent_asin="TARGET")]

        state = object()
        probes = [{
            "sample_id": "sample",
            "scenario_type": "buying",
            "probe_stage": "initial",
            "target_parent_asin": "TARGET",
            "state": state,
        }]
        index = FakeIndex()
        result = evaluate_index(index, probes, candidate_limit=200)
        self.assertEqual(index.received_states, [state])
        self.assertEqual(result["candidate_recall_at_200"], 1.0)
        self.assertEqual(result["by_scenario_type"]["buying"]["retrieval_hit_at_10"], 1.0)

    def test_target_disjoint_split_is_deterministic(self) -> None:
        samples = [
            {
                "sample_id": f"{scenario}_{index}",
                "scenario_type": scenario,
                "ground_truth": {"parent_asin": f"{scenario}_{index}"},
            }
            for scenario in ("buying", "browsing") for index in range(4)
        ]
        first = target_disjoint_split(samples, validation_fraction=0.25, seed=7)
        second = target_disjoint_split(samples, validation_fraction=0.25, seed=7)
        self.assertEqual(first, second)
        train_targets = {row["ground_truth"]["parent_asin"] for row in first[0]}
        validation_targets = {row["ground_truth"]["parent_asin"] for row in first[1]}
        self.assertTrue(train_targets.isdisjoint(validation_targets))

    def test_pairwise_ranker_learns_positive_feature_direction(self) -> None:
        positive = {"lexical": 1.0, "base_formula": 1.0}
        negative = {"lexical": 0.0, "base_formula": 0.0}
        model = train_pairwise([(positive, negative)] * 20, epochs=5, seed=1)
        scores = model.score([positive, negative])
        self.assertGreater(scores[0], scores[1])


if __name__ == "__main__":
    unittest.main()
