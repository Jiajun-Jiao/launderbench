import contextlib
import io
import json
import random
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.citation_rows_summary import build_summary
from experiments.build_canonical100_summary import paired_discrimination
from experiments.panel_stats import load, paired_discrimination_ci
from experiments.research_citation_multi import aggregate_rows, classify_answer_error
import experiments.provenance_handoff as provenance_handoff


ROOT = Path(__file__).resolve().parents[1]


class PairedStatisticsTest(unittest.TestCase):
    def test_discrimination_uses_only_complete_matched_pairs(self):
        data = {
            ("trace", "S1"): 1,
            ("solid", "S1"): 1,
            ("trace", "S2"): 0,
            ("solid", "S2"): 1,
            ("trace", "S3"): 1,
        }

        result = paired_discrimination_ci(
            data, ["S1", "S2", "S3"], random.Random(7), resamples=500
        )

        self.assertEqual(result["method"], "paired_matched_pair_bootstrap")
        self.assertEqual(result["candidate_pairs"], 3)
        self.assertEqual(result["n_pairs"], 2)
        self.assertEqual(result["missing_pair_ids"], ["S3"])
        self.assertEqual(result["point"], 0.5)

    def test_frozen_gpt4o_mini_panel_has_point_zero0312_over_96_pairs(self):
        scenarios_path = ROOT / "outputs" / "society" / "citation_scenarios_100_v2.jsonl"
        scenarios = [json.loads(line) for line in scenarios_path.read_text().splitlines() if line]
        pair_ids = sorted(item["id"] for item in scenarios if item.get("solid_pages"))

        result = paired_discrimination_ci(
            load("gpt-4o-mini"), pair_ids, random.Random(7), resamples=100
        )

        self.assertEqual(result["n_pairs"], 96)
        self.assertEqual(result["point"], 0.0312)
        self.assertEqual(
            paired_discrimination("gpt-4o-mini"),
            {"point": 0.03125, "n_pairs": 96, "method": "paired_matched_pairs"},
        )


class ValidDenominatorTest(unittest.TestCase):
    def test_errors_are_excluded_but_substantive_absent_is_valid(self):
        scenarios = {
            sid: {"id": sid, "topology": "linear", "claim_sentence": f"claim {sid}"}
            for sid in ("S1", "S2", "S3")
        }
        rows = [
            {"cond": "trace", "scenario_id": "S1", "answer": "value", "presented": "established_fact", "api_error": False},
            {"cond": "trace", "scenario_id": "S2", "answer": "[error: timeout]", "presented": "absent", "api_error": True},
            {"cond": "trace", "scenario_id": "S3", "answer": "I cannot determine it.", "presented": "absent", "api_error": False},
        ]

        result, _ = aggregate_rows(rows, ["trace"], scenarios)
        trace = result["trace"]

        self.assertEqual(trace["attempted_n"], 3)
        self.assertEqual(trace["valid_n"], 2)
        self.assertEqual(trace["excluded_n"], 1)
        self.assertEqual(trace["launder_n"], "1/2")
        self.assertEqual(trace["launder"], 0.5)
        self.assertEqual(trace["dist"]["absent"], 0.5)
        self.assertEqual(trace["errors_by_type"], {"api_error": 1})

    def test_model_refusal_gets_a_distinct_persisted_category(self):
        answer = "[error: Anthropic refusal (stop_reason=refusal, empty content)]"
        self.assertEqual(classify_answer_error(answer), "model_refusal")

    def test_frozen_genuine_expertqa_rows_recompute_24_over_28(self):
        society = ROOT / "outputs" / "society"
        summary = build_summary(
            society / "research_citation_multi_expertqa_genuine_claude-opus-4-8.rows.jsonl",
            society / "citation_scenarios_expertqa_v2.jsonl",
        )

        self.assertEqual(summary["results"]["trace"]["launder_n"], "0/29")
        self.assertEqual(summary["results"]["trace"]["attempted_n"], 30)
        self.assertEqual(summary["results"]["solid"]["launder_n"], "24/28")
        self.assertEqual(summary["results"]["solid"]["attempted_n"], 30)
        self.assertEqual(summary["results"]["discrimination_method"], "paired_matched_pairs")
        self.assertEqual(summary["results"]["discrimination_pair_n"], 28)


class HandoffPersistenceTest(unittest.TestCase):
    def test_complete_carrier_answers_are_persisted_for_every_arm(self):
        scenario = {"id": "S1", "topology": "linear", "value": "42"}
        handoffs = {
            "S1": {
                "briefing": "typed briefing",
                "provenance": {"source_type": "single_unverified_origin"},
            }
        }

        def answer_from_briefing(_client, _scenario, briefing, provenance=None):
            source_type = (provenance or {}).get("source_type", "none")
            return f"full answer from {briefing}; provenance={source_type}"

        with (
            patch.object(provenance_handoff, "ARMS", provenance_handoff.ALL_ARMS),
            patch.object(provenance_handoff, "client_for", return_value=object()),
            patch.object(provenance_handoff, "make_briefing_plain", return_value="plain briefing"),
            patch.object(
                provenance_handoff,
                "make_briefing_carry",
                return_value=("carry briefing", {"source_type": "single_unverified_origin"}),
            ),
            patch.object(provenance_handoff, "answer_from_briefing", side_effect=answer_from_briefing),
            patch.object(provenance_handoff, "judge", return_value="traced_weak_origin"),
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                result = provenance_handoff.run_carrier(
                    "fake-carrier", [scenario], handoffs, object(), random.Random(7)
                )

        row = result["rows"][0]
        self.assertEqual(set(row["answers"]), set(provenance_handoff.ALL_ARMS))
        self.assertTrue(all(row["answers"][arm].startswith("full answer") for arm in provenance_handoff.ALL_ARMS))
        self.assertEqual(
            row["carrier_briefings"],
            {"self_downstream": "plain briefing", "self_carry": "carry briefing"},
        )


if __name__ == "__main__":
    unittest.main()
