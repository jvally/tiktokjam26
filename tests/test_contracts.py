import unittest

from conversation import update_state
from shared import Candidate, Product, SearchState


class ContractTests(unittest.TestCase):
    def test_state_round_trip(self):
        state = SearchState(hard_constraints={"color": "black", "budget_max": 100})
        self.assertEqual(state.hard_constraints["color"], ["black"])
        self.assertEqual(SearchState.from_dict(state.to_dict()), state)

    def test_invalid_state_inputs(self):
        for kwargs in ({"turn": 0}, {"turn": 11}, {"turn": True}, {"intent": "unknown"},
                       {"hard_constraints": {"budget_max": float("nan")}},
                       {"hard_constraints": {"budget_max": "100"}},
                       {"hard_constraints": {"budget_min": 200, "budget_max": 100}},
                       {"hard_constraints": {"unknown": "x"}},
                       {"negative_constraints": {"budget_max": 10}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SearchState(**kwargs)

    def test_unknown_state_fields_rejected(self):
        with self.assertRaises(ValueError):
            SearchState.from_dict({"target_parent_asin": "target"})

    def test_product_missing_metadata_and_scores(self):
        candidate = Candidate.from_dict({"parent_asin": "A", "title": "Shoe"})
        self.assertIsNone(candidate.product.price)
        self.assertIsNone(candidate.dense_score)
        self.assertEqual(candidate.product.color, [])

    def test_nested_candidate_round_trip(self):
        candidate = Candidate.from_dict({"product": {"parent_asin": "A", "title": "Shoe"}, "dense_score": .8})
        self.assertEqual(Candidate.from_dict(candidate.to_dict()), candidate)

    def test_invalid_products_and_scores(self):
        for kwargs in ({"dense_score": 1.1}, {"bm25_score": -1}, {"profile_score": float("inf")},
                       {"metadata_score": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Candidate(Product("A", "Shoe"), **kwargs)
        with self.assertRaises(ValueError):
            Product("", "Shoe")
        with self.assertRaises(ValueError):
            Product("A", "Shoe", price=-1)

    def test_updates_merge_remove_and_do_not_mutate(self):
        old = SearchState(query="shoes", hard_constraints={"color": "black", "budget_max": 100})
        new = update_state(old, "actually blue", {"hard_constraints": {"colors": "blue", "budget_max": None}})
        self.assertEqual(new.turn, 2)
        self.assertEqual(new.hard_constraints, {"color": ["blue"]})
        self.assertEqual(old.hard_constraints, {"color": ["black"], "budget_max": 100})

    def test_turns_and_initial_state(self):
        self.assertEqual(update_state(None, "shoes").turn, 1)
        with self.assertRaises(ValueError):
            update_state(SearchState(turn=10), "one more")
        with self.assertRaises(ValueError):
            update_state(None, "shoes", {"turn": 1})


if __name__ == "__main__":
    unittest.main()
