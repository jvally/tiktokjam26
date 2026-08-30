from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent


class AgentContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        catalog = Path(self.temporary.name) / "catalog.jsonl"
        rows = [
            {"parent_asin": "BLUE", "title": "Blue running shoe", "features": ["waterproof"]},
            {"parent_asin": "RED", "title": "Red formal shoe", "features": ["leather"]},
        ]
        catalog.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        self.agent = Agent(catalog)

    def tearDown(self) -> None:
        self.agent.index.close()
        self.temporary.cleanup()

    def test_required_response_and_multi_turn_state(self) -> None:
        self.agent.reset("session", {
            "purchase_frequency": "1-2 prior purchases", "average_prior_rating": 4.0,
            "rating_style": "usually positive", "preference_tags": ["comfort"], "summary": "comfort",
        })
        first = self.agent.respond("session", "I'm looking for running shoes, but I'm still exploring.", 1, 10)
        self.assertEqual(first["ask_attribute"], "other")
        self.assertEqual(first["recommendations"][0]["parent_asin"], "BLUE")
        second = self.agent.respond("session", "For that, what matters is: color: blue.", 2, 10)
        self.assertEqual(second["recommendations"][0]["parent_asin"], "BLUE")
        self.assertEqual(second["usage"], {"prompt_tokens": 0, "completion_tokens": 0})
        self.assertEqual(set(second), {"message", "ask_attribute", "recommendations", "usage"})
        diagnostics = self.agent.get_diagnostics("session")
        self.assertEqual(diagnostics["retrieval_mode"], "lexical")
        self.assertEqual(diagnostics["ranking_mode"], "ltr_formula_blend")
        self.assertTrue(diagnostics["facet_evidence"])
        self.assertFalse(diagnostics["counterfactual_questions"])
        self.assertNotIn("candidate_ids", second)
        features = self.agent.get_rank_features("session")
        self.assertIn("base_formula", features[0])
        self.assertIn("parent_asin", features[0])

    def test_reset_required(self) -> None:
        with self.assertRaises(RuntimeError):
            self.agent.respond("missing", "shoe", 1, 10)


if __name__ == "__main__":
    unittest.main()
