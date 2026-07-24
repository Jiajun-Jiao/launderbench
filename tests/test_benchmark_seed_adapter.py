import json
import tempfile
import unittest
from pathlib import Path

from experiments.benchmark_seed_adapter import convert, load_records, record_to_seed


class BenchmarkSeedAdapterTest(unittest.TestCase):
    def test_expertqa_record_becomes_seed(self):
        record = {
            "question": "What clinical evidence supports using treatment X for condition Y?",
            "metadata": {
                "field": "medicine",
                "specific_field": "oncology",
                "question_type": "evidence synthesis",
            },
            "answers": {"gpt4": {"answer": "A trial-based answer."}},
        }

        seed = record_to_seed(record, 0, "expertqa")

        self.assertEqual(seed["source_benchmark"], "expertqa")
        self.assertEqual(seed["domain_hint"], "oncology")
        self.assertIn("clinical evidence", seed["question"])
        self.assertIn("answer_style", seed)

    def test_alce_like_record_infers_list_answer_style(self):
        record = {
            "id": "q1",
            "question": "Which authors wrote the listed novels?",
            "qa_pairs": [{"short_answers": ["A"]}, {"short_answers": ["B"]}],
            "docs": [{"title": "Reference", "text": "A retrieved passage."}],
        }

        seed = record_to_seed(record, 2, "auto")

        self.assertEqual(seed["source_benchmark"], "alce")
        self.assertEqual(seed["source_id"], "q1")
        self.assertIn("retrieved", seed["evidence_style"])

    def test_jsonl_loader_and_limit(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "items.jsonl"
            path.write_text(
                json.dumps({"question": "When was Foo founded?"}) + "\n"
                + json.dumps({"query": "Where is Bar located?"}) + "\n"
            )
            rows = convert(load_records(str(path)), benchmark="generic", limit=1)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["answer_style"], "date/year")


if __name__ == "__main__":
    unittest.main()
