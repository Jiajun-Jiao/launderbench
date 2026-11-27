import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOCIETY = ROOT / "outputs" / "society"
HUMAN = ROOT / "outputs" / "human_eval"


class LaunderBenchArtifactTest(unittest.TestCase):
    def test_canonical100_panel_shape(self):
        path = SOCIETY / "citation_scenarios_100_v2.jsonl"
        scenarios = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

        self.assertEqual(len(scenarios), 100)
        self.assertEqual(sum(1 for item in scenarios if item.get("solid_pages")), 96)
        self.assertEqual({item["topology"] for item in scenarios},
                         {"linear", "star", "fake_corroboration"})

    def test_headline_result_rows_are_locked(self):
        checks = {
            "research_citation_multi_canonical100_gpt-4o-mini.json": {
                "trace": "96/100",
                "flagged": "4/100",
                "solid": "96/96",
            },
            "research_citation_multi_canonical100_claude-haiku-4-5.json": {
                "trace": "2/100",
                "flagged": "0/100",
                "solid": "96/96",
            },
        }

        for filename, expected in checks.items():
            with self.subTest(filename=filename):
                results = json.loads((SOCIETY / filename).read_text())["results"]
                self.assertEqual(results["trace"]["launder_n"], expected["trace"])
                self.assertEqual(results["flagged"]["launder_n"], expected["flagged"])
                self.assertEqual(results["solid"]["launder_n"], expected["solid"])

    def test_graph_only_completed_results_are_locked(self):
        analysis = json.loads((SOCIETY / "graph_only_rewire_analysis.json").read_text())
        expected_points = {
            "gpt-4o-mini": 0.0,
            "claude-opus-4-8": 0.5,
            "gpt-4.1": 0.11,
            "claude-haiku-4-5": 0.07,
            "claude-sonnet-4-6": 0.09,
            "gemini-2.5-flash": 0.0,
            "o3": 0.59,
            "gpt-5.5": 0.89,
            "gpt-4.1-mini": 0.0,
            "gpt-4o": 0.0,
            "qwen2.5:32b-instruct": -0.06,
            "qwen3:30b-a3b-instruct-2507-q4_K_M": -0.07,
        }
        self.assertEqual(set(analysis["models"]), set(expected_points))
        for model, point in expected_points.items():
            entry = analysis["models"][model]
            self.assertEqual(entry["integrity"]["raw_rows"], 400, model)
            self.assertEqual(entry["integrity"]["error_rows"], 0, model)
            self.assertAlmostEqual(
                entry["production_judge"]["paired_discrimination"]["point"], point,
                places=6, msg=model,
            )
        opus = analysis["models"]["claude-opus-4-8"]
        self.assertEqual(
            opus["root_diagnostic_detail"]["dependent"]["exact_root_id_set_n"], 100
        )
        self.assertEqual(
            opus["root_diagnostic_detail"]["dependent"]["joint_root_ids_and_count_n"], 88
        )
        self.assertEqual(
            opus["root_diagnostic_detail"]["independent"]["joint_root_ids_and_count_n"], 100
        )
        for model in ("o3", "gpt-5.5"):
            for arm in ("dependent", "independent"):
                self.assertEqual(
                    analysis["models"][model]["root_diagnostic_detail"][arm][
                        "joint_root_ids_and_count_n"
                    ],
                    100,
                    (model, arm),
                )
        self.assertEqual(analysis["unpriced_models"],
                         ["gemini-2.5-flash", "gpt-5.5", "o3", "qwen2.5:32b-instruct",
                          "qwen3:30b-a3b-instruct-2507-q4_K_M"])

    def test_register_factorial_results_are_locked(self):
        analysis = json.loads((SOCIETY / "graph_only_register_analysis.json").read_text())
        expected = {
            # (confessed dep, confessed ind, register effect dep, register effect ind,
            #  confessed-row agreement between the factorized and production judges)
            "gpt-4o-mini": ("2/100", "0/100", 0.97, 0.99, 1.0),
            "gemini-2.5-flash": ("0/100", "0/100", 0.98, 0.98, 1.0),
            "claude-haiku-4-5": ("0/100", "0/100", 0.68, 0.75, 1.0),
            "claude-sonnet-4-6": ("0/100", "0/100", 0.53, 0.62, 1.0),
            "gpt-4.1": ("0/100", "0/100", 0.89, 1.0, 1.0),
            "claude-opus-4-8": ("0/99", "0/99", 0.0, 0.505051, 1.0),
            "o3": ("0/100", "0/100", 0.12, 0.71, 1.0),
            "gpt-5.5": ("0/100", "0/100", 0.04, 0.93, 1.0),
            "gpt-4.1-mini": ("1/100", "0/100", 0.99, 1.0, 0.995),
            "gpt-4o": ("0/100", "0/100", 1.0, 1.0, 1.0),
            "qwen2.5:32b-instruct": ("1/100", "1/100", 0.71, 0.65, 1.0),
            "qwen3:30b-a3b-instruct-2507-q4_K_M": ("4/100", "2/100", 0.72, 0.67, 0.895),
        }
        self.assertEqual(set(analysis["models"]), set(expected))
        for model, (conf_dep, conf_ind, reg_dep, reg_ind, agreement) in expected.items():
            entry = analysis["models"][model]
            cells = entry["confessed"]["cells_production"]
            self.assertEqual(cells["dependent"]["established_n"], conf_dep, model)
            self.assertEqual(cells["independent"]["established_n"], conf_ind, model)
            effects = entry["register_effect_production"]
            self.assertAlmostEqual(effects["dependent"]["point"], reg_dep, places=6, msg=model)
            self.assertAlmostEqual(effects["independent"]["point"], reg_ind, places=6, msg=model)
            self.assertAlmostEqual(
                entry["confessed"]["binary_cross_judge_agreement"]["agreement_rate"], agreement,
                places=6, msg=model,
            )
        opus = analysis["models"]["claude-opus-4-8"]["confessed"]["integrity"]
        self.assertEqual(opus["refused_scenario_ids"], ["G001"])
        self.assertEqual(opus["raw_rows"], 400)
        self.assertEqual(opus["valid_answer_rows"], 198)
        self.assertEqual(analysis["unpriced_models"],
                         ["gemini-2.5-flash", "gpt-5.5", "o3", "qwen2.5:32b-instruct",
                          "qwen3:30b-a3b-instruct-2507-q4_K_M"])

    def test_schema2_typed_handoff_controls_are_complete(self):
        checks = {
            "prov_handoff_canonical100_trace_schema2_20260710.json": ("trace", 100, 0),
            "prov_handoff_canonical100_solid_schema2_20260710.json": ("solid", 96, 96),
        }
        for filename, (condition, n, established) in checks.items():
            with self.subTest(filename=filename):
                data = json.loads((SOCIETY / filename).read_text())
                self.assertEqual(data["output_schema_version"], 2)
                self.assertTrue(data["carrier_answers_saved"])
                self.assertEqual(data["config"]["condition"], condition)
                self.assertEqual(len(data["typer_fields"]), n)
                self.assertEqual(len(data["carriers"]), 4)
                for carrier in data["carriers"].values():
                    rows = carrier["rows"]
                    self.assertEqual(len(rows), n)
                    self.assertTrue(all(row["answers"]["typed_handoff"] for row in rows))
                    self.assertEqual(
                        sum(row["verdicts"]["typed_handoff"] == "established_fact"
                            for row in rows),
                        established,
                    )

    def test_scenario_construct_audit_is_locked(self):
        audit = json.loads((HUMAN / "scenario_audit_summary.json").read_text())
        sensitivity = json.loads((HUMAN / "scenario_audit_sensitivity.json").read_text())
        self.assertEqual(audit["primary_annotators"], ["annot1", "annot2", "annot4", "annot5"])
        self.assertEqual(audit["full_construct_majority_pass"], 39)
        self.assertEqual(audit["origin_majority_matches_gold"], 40)
        self.assertEqual(len(audit["one_rater_dissent_scenario_ids"]), 10)
        self.assertEqual(sensitivity["n_excluded"], 10)
        self.assertLessEqual(sensitivity["max_absolute_change"], 0.022)

    def test_explicit_prose_solid_control_is_complete(self):
        data = json.loads(
            (SOCIETY / "prov_handoff_prose_canonical100_solid_schema2_20260710.json").read_text()
        )
        expected = {
            "gpt-4o-mini": 85,
            "gpt-4.1-mini": 86,
            "gpt-4o": 86,
            "gpt-4.1": 87,
        }
        self.assertEqual(data["output_schema_version"], 2)
        self.assertEqual(data["config"]["condition"], "solid")
        for model, established in expected.items():
            rows = data["carriers"][model]["rows"]
            self.assertEqual(len(rows), 96)
            self.assertTrue(all(row["answers"]["prose_warning"] for row in rows))
            self.assertEqual(
                sum(row["verdicts"]["prose_warning"] == "established_fact" for row in rows),
                established,
            )


if __name__ == "__main__":
    unittest.main()
