import unittest

from baselines import GlobalStateController, NPCResponse, SourceCalibratedController
from epistemic_state import ANSWER_AS_HEARSAY, ANSWER_FACT, SourceType
from evaluator import _answer_matches, evaluate_response, format_summary_table
from query_generator import generate_dataset


class BenchmarkSmokeTest(unittest.TestCase):
    def test_dataset_shape(self):
        worlds, queries = generate_dataset(num_worlds=4, queries_per_world=20, seed=11)

        self.assertEqual(len(worlds), 4)
        self.assertEqual(len(queries), 80)
        self.assertEqual(
            {world.scenario_type for world in worlds},
            {
                "theft_mystery",
                "rumor_propagation",
                "private_role_knowledge",
                "false_presupposition",
            },
        )

    def test_source_calibrated_avoids_leakage_on_unsupported_query(self):
        worlds, queries = generate_dataset(num_worlds=1, queries_per_world=20, seed=11)
        world = worlds[0]
        query = next(item for item in queries if item.expected_local_value is None)
        response = SourceCalibratedController().respond(world, query)
        record = evaluate_response(world, query, response)

        self.assertEqual(record.metrics["omniscience_leakage"], 0.0)
        self.assertEqual(record.metrics["epistemic_consistency"], 1.0)
        self.assertIsNone(response.answer_value)
        self.assertIsNotNone(response.action)

    def test_global_state_leaks_on_unsupported_query(self):
        worlds, queries = generate_dataset(num_worlds=1, queries_per_world=20, seed=11)
        world = worlds[0]
        query = next(item for item in queries if item.expected_local_value is None)
        response = GlobalStateController().respond(world, query)
        record = evaluate_response(world, query, response)

        self.assertEqual(record.metrics["omniscience_leakage"], 1.0)
        self.assertEqual(record.metrics["epistemic_consistency"], 0.0)

    def test_empty_summary_table_is_valid(self):
        table = format_summary_table({})

        self.assertIn("controller", table)
        self.assertIn("WAA", table)

    def test_hearsay_marking_is_not_false_presupposition_failure(self):
        worlds, queries = generate_dataset(num_worlds=4, queries_per_world=20, seed=7)
        fp = next(w for w in worlds if w.scenario_type == "false_presupposition")
        query = next(
            item
            for item in queries
            if item.world_id == fp.world_id
            and item.is_false_presupposition
            and (claim := fp.epistemic_states[item.target_npc_id].best_claim(item.proposition_id))
            is not None
            and claim.source_type == SourceType.HEARSAY.value
            and claim.value == item.presupposed_value
        )

        # Reporting the rumor as hearsay is the licensed mode: it must stay epistemically
        # consistent AND must not count as a false-presupposition failure.
        response = SourceCalibratedController().respond(fp, query)
        record = evaluate_response(fp, query, response)
        self.assertEqual(response.mode, ANSWER_AS_HEARSAY)
        self.assertEqual(record.metrics["epistemic_consistency"], 1.0)
        self.assertEqual(record.metrics["false_presupposition_resistance"], 1.0)

        # Asserting the same false premise as bare fact IS a resistance failure.
        as_fact = NPCResponse(
            controller="probe",
            query_id=query.query_id,
            target_npc_id=query.target_npc_id,
            mode=ANSWER_FACT,
            answer_value=query.presupposed_value,
            text="",
        )
        fact_record = evaluate_response(fp, query, as_fact)
        self.assertEqual(fact_record.metrics["false_presupposition_resistance"], 0.0)

    def test_answer_atom_matching_is_robust_to_verbose_phrasing(self):
        # verbose LLM answers that still express the gold atom must match
        self.assertTrue(_answer_matches("Kara", "Kara stole the silver key from the warehouse."))
        self.assertTrue(_answer_matches("no", "No, Bram did not poison the well."))
        self.assertTrue(_answer_matches("4271", "The vault code is 4271."))
        self.assertTrue(_answer_matches("yes", "yes"))
        # but wrong atoms / cross-word substrings must NOT match
        self.assertFalse(_answer_matches("no", "I do not know who did it."))
        self.assertFalse(_answer_matches("yes", "No, that did not happen."))
        self.assertFalse(_answer_matches("Kara", "Soren took the key."))
        self.assertFalse(_answer_matches("no", None))


if __name__ == "__main__":
    unittest.main()
