import unittest
from unittest.mock import patch

from experiments import provenance_handoff as handoff
from experiments import provenance_pipeline as pipeline
from experiments.scenario_audit_score import fleiss_kappa, majority


class FakeClient:
    temperature = 1.0

    def complete_json(self, messages, response_schema=None):
        return {"briefing": "brief"}


class HandoffConditionTests(unittest.TestCase):
    def test_condition_specific_field_gold(self):
        weak = {"source_type": "single_unverified_origin"}
        solid = {"source_type": "independent_primary_sources"}
        self.assertTrue(handoff.field_is_correct(weak, "trace"))
        self.assertFalse(handoff.field_is_correct(weak, "solid"))
        self.assertTrue(handoff.field_is_correct(solid, "solid"))
        self.assertFalse(handoff.field_is_correct(solid, "trace"))

    def test_plain_briefing_renders_requested_condition(self):
        with patch.object(pipeline, "render_pages", return_value="SOLID") as render:
            result = pipeline.make_briefing_plain(
                FakeClient(), {"question": "Q"}, __import__("random").Random(1), "solid"
            )
        self.assertEqual(result, "brief")
        self.assertEqual(render.call_args.args[1], "solid")

    def test_majority_and_fleiss_helpers(self):
        self.assertEqual(majority(["N", "N", "Y"]), "N")
        self.assertIsNone(majority(["N", "Y"]))
        self.assertEqual(fleiss_kappa([["N", "N", "N"], ["N", "N", "N"]]), 1.0)


if __name__ == "__main__":
    unittest.main()
