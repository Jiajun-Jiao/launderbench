import json
import unittest

from evaluator import evaluate_response
from llm_controller import LLMNPCController, build_messages
from query_generator import generate_dataset


class FakeChatClient:
    def __init__(self, response):
        self.response = response
        self.messages = None
        self.response_schema = None

    def complete_json(self, messages, response_schema=None):
        self.messages = messages
        self.response_schema = response_schema
        return self.response


class LLMControllerTest(unittest.TestCase):
    def test_local_policy_prompt_does_not_include_global_truth(self):
        worlds, queries = generate_dataset(num_worlds=1, queries_per_world=1, seed=13)
        world = worlds[0]
        query = queries[0]

        local_messages = build_messages(world, query, "local_state")
        source_messages = build_messages(world, query, "source_calibrated")
        global_messages = build_messages(world, query, "global_state")

        local_payload = json.loads(local_messages[1]["content"])
        source_payload = json.loads(source_messages[1]["content"])
        global_payload = json.loads(global_messages[1]["content"])

        self.assertNotIn("global_truth", local_payload)
        self.assertNotIn("global_truth", source_payload)
        self.assertIn("global_truth", global_payload)

    def test_llm_json_response_is_evaluable(self):
        worlds, queries = generate_dataset(num_worlds=1, queries_per_world=20, seed=13)
        world = worlds[0]
        query = next(item for item in queries if item.expected_local_value is None)
        fake = FakeChatClient(
            {
                "mode": "ask_npc",
                "answer_value": None,
                "text": "I do not know. Ask the guard.",
                "claimed_source_type": None,
                "source_agent": None,
                "action": query.valid_actions[0],
            }
        )
        controller = LLMNPCController(policy="source_calibrated", client=fake)
        response = controller.respond(world, query)
        record = evaluate_response(world, query, response)

        self.assertEqual(response.controller, "llm_source_calibrated")
        self.assertEqual(record.metrics["omniscience_leakage"], 0.0)
        self.assertEqual(record.metrics["information_seeking_utility"], 1.0)
        self.assertIsNotNone(fake.response_schema)

    def test_ablation_policies_control_what_is_revealed(self):
        worlds, queries = generate_dataset(num_worlds=1, queries_per_world=1, seed=13)
        world, query = worlds[0], queries[0]

        sc = json.loads(build_messages(world, query, "source_calibrated")[1]["content"])
        nohint = json.loads(build_messages(world, query, "source_calibrated_nohint")[1]["content"])
        careful = json.loads(build_messages(world, query, "careful")[1]["content"])

        # source_calibrated gets the gold-derived mode hint; the clean ablation must not
        self.assertIn("acceptable_modes", sc)
        self.assertNotIn("acceptable_modes", nohint)
        # both source-calibrated variants carry the provenance rule table
        self.assertIn("source_rules", sc)
        self.assertIn("source_rules", nohint)
        # the strong generic baseline has the same local evidence but no taxonomy/hint
        self.assertNotIn("source_rules", careful)
        self.assertNotIn("acceptable_modes", careful)
        self.assertIn("local_claims", careful)
        # none of the fair baselines may see global truth
        for payload in (nohint, careful):
            self.assertNotIn("global_truth", payload)

    def test_structured_controller_enforces_provenance(self):
        from epistemic_state import SourceType, allowed_modes_for_claim
        from llm_controller import StructuredSourceCalibratedController

        worlds, queries = generate_dataset(num_worlds=4, queries_per_world=20, seed=11)
        by_id = {w.world_id: w for w in worlds}
        chosen = None
        for q in queries:
            claim = by_id[q.world_id].epistemic_states[q.target_npc_id].best_claim(q.proposition_id)
            if claim and claim.source_type == SourceType.DIRECT_OBSERVATION.value:
                chosen = (by_id[q.world_id], q, claim)
                break
        self.assertIsNotNone(chosen)
        world, query, claim = chosen
        licensed = allowed_modes_for_claim(
            claim, valid_actions=query.valid_actions, is_false_presupposition=query.is_false_presupposition
        )

        fake = FakeChatClient(
            {"mode": "answer_fact", "answer_value": claim.value, "text": "ok", "action": None}
        )
        response = StructuredSourceCalibratedController(client=fake).respond(world, query)

        # the schema enum is restricted to the provenance-licensed modes (hard enforcement)
        self.assertEqual(fake.response_schema["properties"]["mode"]["enum"], licensed)
        # provenance attribution comes from the real claim, not the model's guess
        self.assertEqual(response.claimed_source_type, claim.source_type)
        record = evaluate_response(world, query, response)
        self.assertEqual(record.metrics["epistemic_consistency"], 1.0)
        self.assertEqual(record.metrics["omniscience_leakage"], 0.0)


if __name__ == "__main__":
    unittest.main()
