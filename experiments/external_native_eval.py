"""Run native-control cases prepared from original external benchmarks.

This script intentionally reports native metrics, not LaunderBench laundering:

  - answer_from_sources: model answers from original retrieved docs; a judge
    compares to the benchmark reference answer.
  - audit_*_support: model classifies whether the provided answer/claim is
    fully supported by the original sources/evidence.
"""
import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from society_reflective import client_for  # noqa: E402

NATIVE_RENDER_STYLE = os.environ.get("NATIVE_RENDER_STYLE", "plain")


SUPPORT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "support": {
            "type": "string",
            "enum": ["fully_supported", "partially_supported", "unsupported", "insufficient_information"],
        },
        "rationale": {"type": "string"},
    },
    "required": ["support", "rationale"],
}


ANSWER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


MATCH_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "match": {"type": "string", "enum": ["match", "partial", "mismatch", "abstain"]},
        "rationale": {"type": "string"},
    },
    "required": ["match", "rationale"],
}


def load_cases(path, limit=0):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return rows[:limit] if limit else rows


def render_sources(sources):
    lines = []
    for idx, src in enumerate(sources):
        title = src.get("title") or src.get("id")
        if NATIVE_RENDER_STYLE == "plain":
            lines.append(f"[{src.get('id')}] {title}\n{src.get('text')}")
        elif NATIVE_RENDER_STYLE == "citation_wrapper":
            next_src = sources[idx + 1] if idx + 1 < len(sources) else None
            next_title = (next_src.get("title") or next_src.get("id")) if next_src else "none listed"
            lines.append(
                f"Source: {title}\n"
                f"Verbatim excerpt: {src.get('text')}\n"
                f"References listed by this source: {next_title}"
            )
        else:
            raise ValueError(f"unknown NATIVE_RENDER_STYLE={NATIVE_RENDER_STYLE!r}")
    return "\n\n".join(lines)


def ask_answer(client, case):
    dev = (
        "You are a source-grounded research assistant. Answer the question using only the provided "
        "sources. If the sources do not support an answer, say so. Return JSON {\"answer\":\"...\"}."
    )
    user = f"Question: {case['question']}\n\nSources:\n{render_sources(case['sources'])}"
    if hasattr(client, "temperature"):
        client.temperature = 0.0
    try:
        return client.complete_json(
            [{"role": "developer", "content": dev}, {"role": "user", "content": user}],
            response_schema=ANSWER_SCHEMA,
        )["answer"]
    except Exception as exc:  # noqa: BLE001
        return f"[error: {exc}]"


def judge_answer(judge, case, answer):
    dev = (
        "You compare a model's source-grounded answer to a benchmark reference answer. "
        "Use match if it gives the same substantive answer, partial if it captures only part of the "
        "reference answer, mismatch if it contradicts or misses the answer, and abstain if it refuses "
        "or says the sources are insufficient. Return JSON."
    )
    user = json.dumps({
        "question": case["question"],
        "reference_answer": case.get("reference_answer"),
        "model_answer": answer,
    }, ensure_ascii=False)
    if hasattr(judge, "temperature"):
        judge.temperature = 0.0
    try:
        return judge.complete_json(
            [{"role": "developer", "content": dev}, {"role": "user", "content": user}],
            response_schema=MATCH_SCHEMA,
        )
    except Exception as exc:  # noqa: BLE001
        return {"match": "abstain", "rationale": f"judge error: {exc}"}


def audit_support(client, case):
    dev = (
        "You audit source support. Decide whether the candidate answer/claim is fully supported by the "
        "provided sources. Use partially_supported if some important parts are supported and some are not; "
        "unsupported if it contains clear unsupported or conflicting information; insufficient_information "
        "if the sources do not contain enough information to decide. Return JSON."
    )
    user = json.dumps({
        "question_or_prompt": case.get("question"),
        "sources": render_sources(case.get("sources") or []),
        "candidate_answer_or_claim": case.get("candidate_answer"),
    }, ensure_ascii=False)
    if hasattr(client, "temperature"):
        client.temperature = 0.0
    try:
        return client.complete_json(
            [{"role": "developer", "content": dev}, {"role": "user", "content": user}],
            response_schema=SUPPORT_SCHEMA,
        )
    except Exception as exc:  # noqa: BLE001
        return {"support": "insufficient_information", "rationale": f"error: {exc}"}


def is_has_unsupported(label):
    return label in {"partially_supported", "unsupported", "insufficient_information", "has_unsupported"}


def output_path(cases_path, model, tag):
    if not tag:
        stem = os.path.splitext(os.path.basename(cases_path))[0]
        tag = stem.replace("native_cases_", "")
    return os.path.join(
        _ROOT,
        "outputs",
        "society",
        "external_native_"
        + tag
        + ("_" + NATIVE_RENDER_STYLE if NATIVE_RENDER_STYLE != "plain" else "")
        + f"_{model.replace('/', '_').replace(':', '_')}.json",
    )


def summarize(rows):
    by_bench = defaultdict(Counter)
    by_task = defaultdict(Counter)
    for row in rows:
        bench = row["benchmark"]
        task = row["task"]
        if task == "answer_from_sources":
            by_bench[bench][row["judgment"]["match"]] += 1
            by_task[task][row["judgment"]["match"]] += 1
        else:
            pred_bad = is_has_unsupported(row["prediction"]["support"])
            gold_bad = row["gold"] == "has_unsupported"
            key = ("tp" if pred_bad and gold_bad else
                   "tn" if (not pred_bad and not gold_bad) else
                   "fp" if pred_bad and not gold_bad else "fn")
            by_bench[bench][key] += 1
            by_task[task][key] += 1

    def stats(counter):
        if any(k in counter for k in ("tp", "tn", "fp", "fn")):
            tp, tn, fp, fn = counter["tp"], counter["tn"], counter["fp"], counter["fn"]
            n = tp + tn + fp + fn
            return {
                "n": n,
                "accuracy": round((tp + tn) / n, 2) if n else None,
                "unsupported_recall": round(tp / (tp + fn), 2) if tp + fn else None,
                "supported_specificity": round(tn / (tn + fp), 2) if tn + fp else None,
                "confusion": dict(counter),
            }
        n = sum(counter.values())
        return {
            "n": n,
            "match": round(counter["match"] / n, 2) if n else None,
            "match_or_partial": round((counter["match"] + counter["partial"]) / n, 2) if n else None,
            "dist": dict(counter),
        }

    return {
        "by_benchmark": {k: stats(v) for k, v in by_bench.items()},
        "by_task": {k: stats(v) for k, v in by_task.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--model", default=os.environ.get("MODEL", "gpt-4o-mini"))
    ap.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "gpt-4.1-mini"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260630)
    ap.add_argument("--tag", default="")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    cases = load_cases(args.cases, args.limit)
    random.Random(args.seed).shuffle(cases)
    client = client_for(args.model)
    judge = client_for(args.judge_model)
    rows = []
    print(
        f"native control: model={args.model} cases={len(cases)} "
        f"render={NATIVE_RENDER_STYLE}",
        flush=True,
    )
    for idx, case in enumerate(cases, 1):
        row = dict(case)
        if case["task"] == "answer_from_sources":
            answer = ask_answer(client, case)
            row["model_answer"] = answer
            row["judgment"] = judge_answer(judge, case, answer)
        else:
            row["prediction"] = audit_support(client, case)
        rows.append(row)
        if idx % 10 == 0 or idx == len(cases):
            print(f"  done {idx}/{len(cases)}", flush=True)

    out = {
        "config": {"model": args.model, "judge_model": args.judge_model, "cases": args.cases,
                   "n": len(cases), "native_render_style": NATIVE_RENDER_STYLE},
        "summary": summarize(rows),
        "rows": rows,
    }
    path = args.output or output_path(args.cases, args.model, args.tag)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(out["summary"], indent=2, ensure_ascii=False))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
