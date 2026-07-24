"""Convert existing attribution/RAG benchmark files into LaunderBench seed specs.

The output is NOT a citation-laundering benchmark yet. It is a distributional seed
file used by citation_seeded.py: each seed preserves the external benchmark's
question style, field/domain hint, answer type, and evidence style, while the
downstream generator creates prior-free fictional claims and controlled citation
graphs. This avoids parametric-knowledge contamination from real benchmark facts.

Supported inputs are deliberately heuristic and dependency-free:
  - ExpertQA-like JSONL: question + metadata.field/specific_field/question_type
  - ALCE-like JSON/JSONL: question + docs/passages/qa_pairs/annotations
  - RAGTruth-like JSON/JSONL: query/prompt + source_info/documents/response
  - generic JSON/JSONL with a question/query/input field

Usage:
  python3 experiments/benchmark_seed_adapter.py \
    --benchmark expertqa --input data/r2_compiled_anon.jsonl \
    --output outputs/society/seeds_expertqa.jsonl --limit 100
"""
import argparse
import json
import os
import random
import re
from collections.abc import Iterable


QUESTION_PATHS = (
    "question",
    "query",
    "input",
    "prompt",
    "user_question",
    "instruction",
)
DOC_PATHS = (
    "docs",
    "documents",
    "passages",
    "retrieved_docs",
    "ctxs",
    "source_info",
    "references",
    "wikipages",
)
ANSWER_PATHS = (
    "answer",
    "answers",
    "output",
    "response",
    "gold_answer",
    "annotations",
    "qa_pairs",
)


def _get_path(obj, path):
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _first(obj, paths):
    for path in paths:
        val = _get_path(obj, path)
        if val not in (None, "", [], {}):
            return val
    return None


def _short_text(value, limit=360):
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    elif isinstance(value, dict):
        parts = []
        for key in ("title", "text", "content", "answer", "response", "question"):
            if key in value and value[key]:
                parts.append(str(value[key]))
        text = " | ".join(parts) if parts else json.dumps(value, ensure_ascii=False)
    elif isinstance(value, list):
        text = " | ".join(_short_text(v, 120) for v in value[:4])
    else:
        text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _iter_records_from_json(value):
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                yield item
    elif isinstance(value, dict):
        # Common wrappers used by dataset dumps.
        for key in ("data", "examples", "items", "instances", "records"):
            if isinstance(value.get(key), list):
                yield from _iter_records_from_json(value[key])
                return
        # Otherwise treat the dict itself as one record.
        yield value


def load_records(path):
    with open(path) as f:
        text = f.read()
        stripped = text.lstrip()
        if stripped.startswith("["):
            yield from _iter_records_from_json(json.loads(text))
            return
        if stripped.startswith("{"):
            try:
                yield from _iter_records_from_json(json.loads(text))
                return
            except json.JSONDecodeError:
                pass
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def infer_benchmark(record, explicit):
    if explicit != "auto":
        return explicit
    if "metadata" in record and ("field" in record["metadata"] or "specific_field" in record["metadata"]):
        return "expertqa"
    if any(k in record for k in ("qa_pairs", "annotations", "wikipages")):
        return "alce"
    if any(k in record for k in ("source_info", "hallucination", "response")):
        return "ragtruth"
    return "generic"


def infer_answer_style(question, answer_obj=None):
    q = (question or "").lower().strip()
    if q.startswith(("when ", "what year", "which year", "what date")):
        return "date/year"
    if q.startswith(("where ", "which city", "which country", "what city", "what country")):
        return "place/location"
    if q.startswith(("who ", "whose ", "which person")):
        return "person/name"
    if q.startswith(("how many", "how much", "what percentage", "what percent", "what number")):
        return "number/statistic"
    if "meaning" in q or "origin of" in q or "etymolog" in q:
        return "definition/etymology"
    if isinstance(answer_obj, list) and len(answer_obj) > 1:
        return "list of short answers"
    text = _short_text(answer_obj, 240).lower()
    if re.search(r"\b\d{3,4}\b", text):
        return "date/year or numeric value"
    if re.search(r"\b\d+(\.\d+)?\s?(%|percent|km|kg|metres?|meters?|years?)\b", text):
        return "number with unit"
    return "short factual answer"


def infer_domain_hint(record, benchmark):
    meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    docs = _first(record, DOC_PATHS)
    doc_hint = ""
    if isinstance(docs, list):
        titles = [_short_text(d.get("title"), 60) for d in docs[:3] if isinstance(d, dict) and d.get("title")]
        if titles:
            doc_hint = " / ".join(titles)
    elif isinstance(docs, dict):
        doc_hint = _short_text(docs.get("title") or docs.get("source") or docs.get("question"), 120)
    candidates = [
        meta.get("specific_field"),
        meta.get("field"),
        meta.get("question_type"),
        record.get("task_type"),
        record.get("source"),
        record.get("field"),
        record.get("domain"),
        record.get("category"),
        record.get("topic"),
        record.get("dataset"),
        doc_hint,
        benchmark,
    ]
    return _short_text(next((c for c in candidates if c), benchmark), 160)


def infer_evidence_style(record):
    docs = _first(record, DOC_PATHS)
    text = _short_text(docs, 480)
    if not text:
        return "retrieved passages or cited web snippets"
    # Keep only style cues, not enough content to use as an answer.
    if len(text) > 260:
        text = text[:260] + "..."
    return text


def source_id(record, idx):
    for key in ("id", "qid", "question_id", "example_id", "sample_id"):
        if key in record and record[key] not in (None, ""):
            return str(record[key])
    return str(idx)


def record_to_seed(record, idx, explicit_benchmark="auto"):
    benchmark = infer_benchmark(record, explicit_benchmark)
    question = _short_text(_first(record, QUESTION_PATHS), 500)
    if not question:
        return None
    answer_obj = _first(record, ANSWER_PATHS)
    seed = {
        "seed_id": f"{benchmark}:{source_id(record, idx)}",
        "source_benchmark": benchmark,
        "source_id": source_id(record, idx),
        "question": question,
        "domain_hint": infer_domain_hint(record, benchmark),
        "answer_style": infer_answer_style(question, answer_obj),
        "evidence_style": infer_evidence_style(record),
    }
    meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    if meta:
        seed["source_metadata"] = {
            key: _short_text(meta.get(key), 160)
            for key in ("field", "specific_field", "question_type")
            if meta.get(key)
        }
    return seed


def convert(records: Iterable[dict], benchmark="auto", limit=0, shuffle=False, seed=20260630):
    converted = []
    for idx, record in enumerate(records):
        item = record_to_seed(record, idx, benchmark)
        if item:
            converted.append(item)
    if shuffle:
        random.Random(seed).shuffle(converted)
    if limit:
        converted = converted[:limit]
    return converted


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="auto", choices=["auto", "alce", "expertqa", "ragtruth", "generic"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=20260630)
    args = ap.parse_args()

    rows = convert(load_records(args.input), benchmark=args.benchmark, limit=args.limit,
                   shuffle=args.shuffle, seed=args.seed)
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} seed specs -> {args.output}")
    by_bench = {}
    for row in rows:
        by_bench[row["source_benchmark"]] = by_bench.get(row["source_benchmark"], 0) + 1
    print("benchmarks:", by_bench)


if __name__ == "__main__":
    main()
