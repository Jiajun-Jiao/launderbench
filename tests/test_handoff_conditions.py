import hashlib
import json
import random
import unittest
from unittest.mock import patch

from experiments import provenance_handoff as handoff
from experiments import provenance_handoff_prose as prose
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


class RecordingClient:
    temperature = 1.0

    def __init__(self, record):
        self.calls = []
        self.record = record

    def complete_json(self, messages, response_schema=None):
        self.calls.append((messages, response_schema))
        return {"briefing": "brief", "provenance": dict(self.record), "answer": "a"}


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


class ProvenanceRecordTests(unittest.TestCase):
    """The optional independent-source count must leave the default record and prompts untouched."""

    # SHA-256 of the typer messages, the typer schema, and the carrier messages produced by the code
    # behind every reported number (question "Q", bundle "BUNDLE", default briefing budget).
    DEFAULT_TYPER_MESSAGES = "3b92fd18af1134ec2a93bd2c30aa95d7cac22b283feda11f2ff40643d1d9910e"
    DEFAULT_TYPER_SCHEMA = "e12861fe8753be6670061dd9d0f711d94a4d13f7d812037d80877e0a8974a3b5"
    DEFAULT_CARRIER_MESSAGES = "453aec3506fc0009b7d83b182fd2844632be2b5d8e93f2750c865a67c817d332"
    RECORD = {"value": "V", "source_type": "unknown", "verified": False, "basis": "b"}

    def _typer_call(self, record, **kwargs):
        client = RecordingClient(record)
        with patch.object(pipeline, "render_pages", return_value="BUNDLE"), \
                patch.object(pipeline, "BUDGET", "one or two sentences"):
            _, prov = pipeline.make_briefing_carry(
                client, {"question": "Q"}, random.Random(1), "trace", **kwargs
            )
        messages, schema = client.calls[0]
        return messages, schema, prov

    def test_default_record_and_prompts_are_unchanged(self):
        with patch.object(pipeline, "SOURCE_COUNT", False):
            messages, schema, prov = self._typer_call(self.RECORD)
            self.assertIs(pipeline.carry_schema(), pipeline.SCH_BRIEF_CARRY)
        self.assertEqual(_digest(messages), self.DEFAULT_TYPER_MESSAGES)
        self.assertEqual(_digest(schema), self.DEFAULT_TYPER_SCHEMA)
        self.assertEqual(set(prov), {"value", "source_type", "verified", "basis"})
        carrier = RecordingClient(self.RECORD)
        pipeline.answer_from_briefing(carrier, {"question": "Q"}, "brief", dict(self.RECORD))
        self.assertEqual(_digest(carrier.calls[0][0]), self.DEFAULT_CARRIER_MESSAGES)
        self.assertEqual(
            prose.render_field_template(dict(self.RECORD)),
            "The key value is V. Provenance: its grounding could not be determined. "
            "Independently verified: no. Basis: b",
        )

    def test_source_count_extends_the_record_when_enabled(self):
        record = dict(self.RECORD, independent_source_count="1")
        messages, schema, prov = self._typer_call(record, with_count=True)
        record_schema = schema["properties"]["provenance"]
        self.assertEqual(record_schema["properties"]["independent_source_count"], {"type": "integer"})
        self.assertIn("independent_source_count", record_schema["required"])
        self.assertNotIn(
            "independent_source_count",
            pipeline.SCH_BRIEF_CARRY["properties"]["provenance"]["properties"],
        )
        self.assertIn("independent_source_count", messages[0]["content"])
        self.assertEqual(prov["independent_source_count"], 1)
        carrier = RecordingClient(self.RECORD)
        pipeline.answer_from_briefing(carrier, {"question": "Q"}, "brief", prov)
        self.assertIn('"independent_source_count": 1', carrier.calls[0][0][1]["content"])
        self.assertIn("Independent sources: 1. Basis: b", prose.render_field_template(prov))

    def test_module_flag_enables_the_count(self):
        record = dict(self.RECORD, independent_source_count=2)
        with patch.object(pipeline, "SOURCE_COUNT", True):
            _, schema, prov = self._typer_call(record)
        self.assertIn("independent_source_count", schema["properties"]["provenance"]["required"])
        self.assertEqual(prov["independent_source_count"], 2)

    def test_source_count_normalization(self):
        self.assertEqual(pipeline.as_source_count(2), 2)
        self.assertEqual(pipeline.as_source_count("3"), 3)
        self.assertEqual(pipeline.as_source_count(1.0), 1)
        self.assertEqual(pipeline.as_source_count(0), 0)
        for bad in (-1, True, None, "two", 1.5, []):
            self.assertIsNone(pipeline.as_source_count(bad), bad)


if __name__ == "__main__":
    unittest.main()
