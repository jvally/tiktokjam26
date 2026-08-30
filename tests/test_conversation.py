from __future__ import annotations

import unittest

from conversation import BudgetRange, ConversationState, update_from_message


class ConversationTests(unittest.TestCase):
    def test_cumulative_query_and_constraint_types(self) -> None:
        state = ConversationState({"preference_tags": ["comfort"]})
        update_from_message(state, "I'm looking for running shoes, but I'm still exploring.", 1)
        state.note_question("other")
        update_from_message(state, "For that, what matters is: waterproof; color: blue.", 2)
        self.assertIn("running shoes", state.query)
        self.assertIn("waterproof", state.query)
        self.assertEqual(state.known_attributes(), {"feature", "color"})
        self.assertEqual(state.intent, "buying")

    def test_intent_override_removes_initial_soft_preference(self) -> None:
        state = ConversationState({})
        update_from_message(state, "I'm looking for shirts. I prefer wool.", 1)
        self.assertIn("wool", state.query)
        update_from_message(state, "Actually, ignore my earlier preference. What I need is: cotton.", 3)
        self.assertNotIn("wool", state.query)
        self.assertIn("cotton", state.query)

    def test_boundary_and_exhausted_answers_are_distinguished(self) -> None:
        state = ConversationState({})
        state.note_question("other")
        update_from_message(state, "I don't have a preference for other; please use your judgment.", 2)
        self.assertTrue(state.boundary_seen)
        self.assertNotIn("other", state.exhausted_attributes)
        state.note_question("other")
        update_from_message(state, "I don't have an additional preference for other.", 3)
        self.assertIn("other", state.exhausted_attributes)

    def test_correction_and_negation(self) -> None:
        state = ConversationState({})
        update_from_message(state, "I'm looking for shoes. A key requirement is: color: black.", 1)
        update_from_message(state, "Actually, I need blue but not red.", 2)
        values = {(item.attribute, item.value, item.kind) for item in state.constraints}
        self.assertIn(("color", "blue", "hard"), values)
        self.assertIn(("color", "red", "negative"), values)
        self.assertNotIn("black", state.query)

    def test_compound_override_replaces_multiple_slots_and_category(self) -> None:
        state = ConversationState({})
        update_from_message(
            state,
            "I'm looking for running shoes. A key requirement is: black leather.",
            1,
        )
        update_from_message(
            state,
            "Actually, ignore my earlier preference. What I need is: casual white sneakers under $80.",
            3,
        )
        values = {(item.attribute, str(item.value), item.kind) for item in state.constraints}
        self.assertEqual(state.category, "sneakers")
        self.assertIn(("color", "white", "hard"), values)
        self.assertIn(("style", "casual", "hard"), values)
        self.assertIn("budget", state.known_attributes())
        self.assertNotIn("black", state.query)
        self.assertNotIn("leather", state.query)
        self.assertEqual(state.override_count, 1)
        self.assertGreaterEqual(len(state.retracted_constraints), 2)

    def test_budget_range_is_normalized(self) -> None:
        state = ConversationState({})
        update_from_message(state, "My budget is between $50 and $100.", 1)
        budget = next(item.value for item in state.constraints if item.attribute == "budget")
        self.assertIsInstance(budget, BudgetRange)
        self.assertEqual((budget.minimum, budget.maximum), (50.0, 100.0))

    def test_material_percentage_is_not_mistaken_for_budget(self) -> None:
        state = ConversationState({})
        update_from_message(state, "I want a shirt made from 100% cotton.", 1)
        self.assertIn("material", state.known_attributes())
        self.assertNotIn("budget", state.known_attributes())

    def test_natural_exclusion_is_negative(self) -> None:
        state = ConversationState({})
        update_from_message(state, "I want shoes but don't want red.", 1)
        self.assertIn(
            ("color", "red", "negative"),
            {(item.attribute, item.value, item.kind) for item in state.constraints},
        )

    def test_rejected_results_are_recorded(self) -> None:
        state = ConversationState({})
        update_from_message(state, "Those options are not quite right yet.", 2)
        self.assertEqual(state.rejected_result_sets, 1)

    def test_recommendation_memory_is_scoped_to_intent(self) -> None:
        state = ConversationState({})
        state.note_recommendations(["A", "B", "A"])
        self.assertEqual(state.shown_ids(), {"A", "B"})
        state.begin_override()
        self.assertEqual(state.shown_ids(), set())


if __name__ == "__main__":
    unittest.main()
