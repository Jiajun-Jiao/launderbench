"""Prepare native-control cases from external attribution/RAG benchmarks.

These controls are deliberately NOT LaunderBench scenarios. They keep each
source benchmark's original task semantics and provide a comparison panel:

  - ALCE: answer a real question from retrieved documents; compare to reference.
  - RAGTruth: audit whether a candidate response contains unsupported content.
  - ExpertQA: audit whether an annotated claim is supported by its evidence.

The goal is to show how models behave on the original distributions, while the
LaunderBench-seeded panels test controlled citation-laundering graphs.
"""
import argparse
import json
import os
import random
import re
from collections import defaultdict


def short_text(value, limit=2400):
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False)
    value = re.sub(r"\s+", " ", value).strip()
    return value[:limit]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json_or_jsonl(path):
    text = open(path).read().lstrip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def doc_snippets(docs, limit=5):
    out = []
    for idx, doc in enumerate((docs or [])[:limit]):
        if isinstance(doc, dict):
            title = doc.get("title") or doc.get("url") or f"doc-{idx+1}"
            text = doc.get("text") or doc.get("content") or doc.get("summary") or ""
        else:
            title = f"doc-{idx+1}"
            text = str(doc)
        out.append({"id": f"D{idx+1}", "title": short_text(title, 160), "text": short_text(text, 1000)})
    return out


def prepare_alce(paths, per_file, rng):
    rows = []
    for path in paths:
        data = load_json_or_jsonl(path)
        rng.shuffle(data)
        subset = data[:per_file]
        name = os.path.basename(path).split("_eval_")[0]
        for idx, rec in enumerate(subset):
            question = short_text(rec.get("question"), 800)
            answer = short_text(rec.get("answer"), 1600)
            if not question or not answer:
                continue
            rows.append({
                "id": f"alce:{name}:{rec.get('sample_id') or rec.get('id') or idx}",
                "benchmark": "alce",
                "task": "answer_from_sources",
                "dataset_slice": name,
                "question": question,
                "sources": doc_snippets(rec.get("docs"), limit=5),
                "reference_answer": answer,
                "gold": "reference_answer",
            })
    return rows


def prepare_ragtruth(source_path, response_path, n, rng):
    sources = {}
    for line in open(source_path):
        row = json.loads(line)
        sources[str(row["source_id"])] = row

    supported, unsupported = [], []
    for line in open(response_path):
        row = json.loads(line)
        if row.get("quality") not in (None, "good"):
            continue
        src = sources.get(str(row.get("source_id")))
        if not src:
            continue
        target = unsupported if row.get("labels") else supported
        labels = row.get("labels") or []
        target.append({
            "id": f"ragtruth:{row.get('id')}",
            "benchmark": "ragtruth",
            "task": "audit_response_support",
            "dataset_slice": src.get("task_type") or src.get("source") or "ragtruth",
            "question": short_text(src.get("prompt") or "", 900),
            "sources": [{"id": "S1", "title": short_text(src.get("source") or "source", 160),
                         "text": short_text(src.get("source_info"), 3000)}],
            "candidate_answer": short_text(row.get("response"), 2200),
            "gold": "has_unsupported" if labels else "fully_supported",
            "gold_label_types": sorted({lab.get("label_type", "") for lab in labels if lab.get("label_type")}),
        })

    rng.shuffle(supported)
    rng.shuffle(unsupported)
    half = n // 2
    return supported[:half] + unsupported[:n - half]


NEG_SUPPORT = {"missing", "incomplete", "partial", "none"}


def prepare_expertqa(path, n, rng):
    supported, unsupported = [], []
    for line in open(path):
        rec = json.loads(line)
        question = short_text(rec.get("question"), 900)
        answers = rec.get("answers") or {}
        for answer_name, answer in answers.items():
            for cidx, claim in enumerate(answer.get("claims") or []):
                support = str(claim.get("support") or "").strip()
                if support.lower() == "n/a":
                    continue
                evidence = claim.get("evidence") or []
                row = {
                    "id": f"expertqa:{rec.get('question')[:40]}:{answer_name}:{cidx}",
                    "benchmark": "expertqa",
                    "task": "audit_claim_support",
                    "dataset_slice": (rec.get("metadata") or {}).get("specific_field")
                    or (rec.get("metadata") or {}).get("field") or "expertqa",
                    "question": question,
                    "sources": [{"id": f"E{i+1}", "title": f"evidence-{i+1}", "text": short_text(ev, 1400)}
                                for i, ev in enumerate(evidence[:4])],
                    "candidate_answer": short_text(claim.get("claim_string"), 1800),
                    "gold": "fully_supported" if support.lower() == "complete" else "has_unsupported",
                    "source_support_label": support,
                }
                if support.lower() == "complete":
                    supported.append(row)
                elif support.lower() in NEG_SUPPORT:
                    unsupported.append(row)

    rng.shuffle(supported)
    rng.shuffle(unsupported)
    half = n // 2
    return supported[:half] + unsupported[:n - half]


def summarize(rows):
    by_bench = defaultdict(int)
    by_task = defaultdict(int)
    by_gold = defaultdict(int)
    for row in rows:
        by_bench[row["benchmark"]] += 1
        by_task[row["task"]] += 1
        by_gold[row["gold"]] += 1
    print("benchmarks:", dict(by_bench))
    print("tasks:", dict(by_task))
    print("gold:", dict(by_gold))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["alce", "ragtruth", "expertqa", "all"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260630)
    ap.add_argument("--alce-files", nargs="*", default=[
        "outputs/external/alce_data/ALCE-data/asqa_eval_gtr_top100_reranked_oracle.json",
        "outputs/external/alce_data/ALCE-data/qampari_eval_gtr_top100_reranked_oracle.json",
        "outputs/external/alce_data/ALCE-data/eli5_eval_bm25_top100_reranked_oracle.json",
    ])
    ap.add_argument("--ragtruth-source", default="outputs/external/ragtruth_source_info.jsonl")
    ap.add_argument("--ragtruth-response", default="outputs/external/ragtruth_response.jsonl")
    ap.add_argument("--expertqa", default="outputs/external/expertqa_r2_compiled_anon.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    if args.benchmark in ("alce", "all"):
        per_file = max(1, args.n // len(args.alce_files))
        rows.extend(prepare_alce(args.alce_files, per_file, rng)[:args.n])
    if args.benchmark in ("ragtruth", "all"):
        rows.extend(prepare_ragtruth(args.ragtruth_source, args.ragtruth_response, args.n, rng))
    if args.benchmark in ("expertqa", "all"):
        rows.extend(prepare_expertqa(args.expertqa, args.n, rng))

    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} native-control cases -> {args.output}")
    summarize(rows)


if __name__ == "__main__":
    main()
