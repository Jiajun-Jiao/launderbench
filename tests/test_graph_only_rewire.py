import itertools
import json
import unittest
from pathlib import Path

from experiments.graph_only_rewire import (
    ARMS,
    DEFAULT_INSTRUMENT,
    DEFAULT_SOURCE,
    EDGE_TEMPLATES,
    JUDGE_SCHEMA,
    ROOT_SCHEMA,
    build_instrument,
    claim_root_ids,
    dry_run_plan,
    follow_terminal,
    indegrees,
    render_scenario,
    score_root_response,
    strip_reference_lines,
    summarize_rows,
    terminal_ids,
    validate_instrument,
)
from experiments.graph_only_rejudge import summarize_rejudgments


ROOT = Path(__file__).resolve().parents[1]


class GraphOnlyRewireDesignTest(unittest.TestCase):
    def test_fixed_terminal_count_cannot_change_roots_when_every_page_bears_claim(self):
        # Exhaust all acyclic, at-most-one-outgoing-edge graphs on three claim-bearing pages.
        # For every graph, terminal claim roots equal terminals; therefore fixing |T| fixes root count.
        ids = ("A", "B", "C")
        pages = [{"id": page_id, "claim_bearing": True} for page_id in ids]
        checked = 0
        for targets in itertools.product(*[(None, *[other for other in ids if other != page_id]) for page_id in ids]):
            edges = dict(zip(ids, targets))
            try:
                for page_id in ids:
                    follow_terminal(page_id, edges)
            except ValueError:
                continue
            checked += 1
            self.assertEqual(len(claim_root_ids(pages, edges)), len(terminal_ids(edges)))
        self.assertGreater(checked, 0)

    def test_five_page_placeholder_design_holds_all_structural_invariants(self):
        instrument = build_instrument(DEFAULT_SOURCE, limit=3)
        report = validate_instrument(instrument)
        self.assertTrue(report["all_invariants_pass"])
        self.assertEqual(report["n"], 3)

        scenario = instrument["scenarios"][0]
        dependent = scenario["arms"]["dependent"]["citation_edges"]
        independent = scenario["arms"]["independent"]["citation_edges"]
        self.assertEqual(terminal_ids(dependent), ["P3", "P5"])
        self.assertEqual(terminal_ids(independent), ["P2", "P3"])
        self.assertEqual(claim_root_ids(scenario["pages"], dependent), ["P3"])
        self.assertEqual(claim_root_ids(scenario["pages"], independent), ["P2", "P3"])
        self.assertEqual(indegrees(dependent), indegrees(independent))
        self.assertEqual(sum(value is not None for value in dependent.values()), 3)
        self.assertEqual(sum(value is not None for value in independent.values()), 3)

    def test_page_order_is_matched_within_item_and_counterbalanced_across_items(self):
        instrument = build_instrument(DEFAULT_SOURCE, limit=10)
        first_pages = [scenario["page_order"][0] for scenario in instrument["scenarios"]]
        self.assertEqual({page_id: first_pages.count(page_id) for page_id in sorted(set(first_pages))},
                         {"P1": 2, "P2": 2, "P3": 2, "P4": 2, "P5": 2})
        for scenario in instrument["scenarios"]:
            self.assertEqual([page["id"] for page in scenario["pages"]], scenario["page_order"])

    def test_rendered_arms_differ_only_in_two_reference_pointer_lines(self):
        scenario = build_instrument(DEFAULT_SOURCE, limit=1)["scenarios"][0]
        dependent = render_scenario(scenario, "dependent")
        independent = render_scenario(scenario, "independent")
        self.assertEqual(strip_reference_lines(dependent), strip_reference_lines(independent))
        differing_lines = [
            (left, right)
            for left, right in zip(dependent.splitlines(), independent.splitlines())
            if left != right
        ]
        self.assertEqual(len(differing_lines), 2)
        self.assertTrue(all(left.startswith("  Reference pointer:") for left, _ in differing_lines))

    def test_call_plan_is_exact_and_does_not_create_clients(self):
        instrument = build_instrument(DEFAULT_SOURCE, limit=4)
        plan = dry_run_plan(
            instrument,
            models=["gpt-4o-mini", "claude-opus-4-8"],
            tasks=["answer", "roots"],
            judge_model="gpt-4.1-mini",
        )
        self.assertEqual(plan["mode"], "dry_run_no_clients_created")
        self.assertEqual(plan["n_scenarios"], 4)
        self.assertEqual(plan["per_model"]["gpt-4o-mini"]["answer_calls"], 8)
        self.assertEqual(plan["per_model"]["gpt-4o-mini"]["root_diagnostic_calls"], 8)
        self.assertEqual(plan["per_model"]["gpt-4o-mini"]["judge_calls"], 8)
        self.assertEqual(plan["logical_calls_total"], 48)

    def test_established_metric_requires_the_target_value(self):
        base = {
            "model": "fake",
            "task": "answer",
            "scenario_id": "G000",
            "answer_error": None,
            "judge_error": None,
        }
        rows = [
            {**base, "arm": "dependent", "judgment": {
                "stance": "established", "lineage_statement": "single_lineage",
                "target_value_status": "other_value",
            }},
            {**base, "arm": "independent", "judgment": {
                "stance": "established", "lineage_statement": "multiple_independent",
                "target_value_status": "target_value",
            }},
        ]
        summary = summarize_rows(rows)
        self.assertEqual(JUDGE_SCHEMA["required"],
                         ["stance", "lineage_statement", "target_value_status"])
        self.assertEqual(summary["answer"]["dependent"]["established_n"], "0/1")
        self.assertEqual(summary["answer"]["independent"]["established_n"], "1/1")
        self.assertEqual(summary["paired_discrimination"]["point"], 1.0)

    def test_root_uniqueness_is_enforced_locally_for_provider_compatible_schema(self):
        self.assertNotIn("uniqueItems", ROOT_SCHEMA["properties"]["claim_root_page_ids"])
        clean = score_root_response(
            {"claim_root_page_ids": ["P2", "P3"], "independent_claim_lineages": 2},
            ["P2", "P3"],
        )
        duplicate = score_root_response(
            {"claim_root_page_ids": ["P2", "P2", "P3"], "independent_claim_lineages": 2},
            ["P2", "P3"],
        )
        wrong_count = score_root_response(
            {"claim_root_page_ids": ["P2", "P3"], "independent_claim_lineages": 1},
            ["P2", "P3"],
        )
        self.assertTrue(clean["exact_root_ids"])
        self.assertTrue(clean["lineage_count_correct"])
        self.assertTrue(clean["exact_roots"])
        self.assertFalse(duplicate["root_ids_unique"])
        self.assertFalse(duplicate["exact_roots"])
        self.assertTrue(wrong_count["exact_root_ids"])
        self.assertFalse(wrong_count["lineage_count_correct"])
        self.assertFalse(wrong_count["exact_roots"])

    def test_production_rejudge_summary_keeps_paired_structure(self):
        rows = [
            {"scenario_id": "G1", "arm": "dependent", "production_presented": "established_fact", "judge_error": None},
            {"scenario_id": "G1", "arm": "independent", "production_presented": "established_fact", "judge_error": None},
            {"scenario_id": "G2", "arm": "dependent", "production_presented": "hedged", "judge_error": None},
            {"scenario_id": "G2", "arm": "independent", "production_presented": "established_fact", "judge_error": None},
        ]
        summary = summarize_rejudgments(rows)
        paired = summary["paired_discrimination"]
        self.assertEqual(paired["n_pairs"], 2)
        self.assertEqual(paired["point"], 0.5)
        self.assertEqual(paired["discordant_independent_only"], 1)
        self.assertEqual(paired["discordant_dependent_only"], 0)

    def test_checked_in_instrument_matches_builder_and_validates(self):
        if not DEFAULT_INSTRUMENT.exists():
            self.skipTest("instrument is generated after source patching")
        instrument = json.loads(DEFAULT_INSTRUMENT.read_text())
        report = validate_instrument(instrument)
        self.assertEqual(report["n"], 100)
        self.assertEqual(instrument["design"]["dependent_claim_roots"], ["P3"])
        self.assertEqual(instrument["design"]["independent_claim_roots"], ["P2", "P3"])
        self.assertEqual(instrument["scenarios"][0]["arms"]["dependent"]["citation_edges"], EDGE_TEMPLATES["dependent"])
        self.assertEqual(set(instrument["scenarios"][0]["arms"]), set(ARMS))


if __name__ == "__main__":
    unittest.main()
