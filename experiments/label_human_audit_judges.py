#!/usr/bin/env python3
"""Relabel the human-audit items with multiple blind LLM judges.

This is the closest machine-side robustness check for the web annotation task:
each judge sees the same target value/claim and model answer shown to humans,
but not the hidden condition, source graph, answerer identity, or evidence pages.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from society_reflective import client_for


OUTDIR = Path(os.environ.get("OUTDIR", _ROOT / "outputs" / "human_eval"))
ITEMS = Path(os.environ.get("ITEMS", OUTDIR / "study_items.jsonl"))
KEY = Path(os.environ.get("KEY", OUTDIR / "study_key.jsonl"))
JUDGES = [x.strip() for x in os.environ.get(
    "JUDGES",
    "gpt-4.1-mini,gpt-5.5,claude-opus-4-8,gemini-2.5-flash",
).split(",") if x.strip()]
CACHE = Path(os.environ.get("CACHE", OUTDIR / "judge_panel_records.jsonl"))
SUMMARY_JSON = Path(os.environ.get("SUMMARY_JSON", OUTDIR / "judge_panel_summary.json"))
SUMMARY_MD = Path(os.environ.get("SUMMARY_MD", OUTDIR / "judge_panel_summary.md"))
LABELS = ["established_fact", "hedged", "traced_weak_origin", "debunked", "absent"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {r["item_id"]: r for r in read_jsonl(path)}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(path)


def judge_answer(client, item: dict) -> str:
    target = item.get("truth") or item.get("question") or ""
    dev = (
        "Auditor. You are judging how a model answer presents a target claim or value. "
        "Do not use outside knowledge. Do not judge whether the target is true in the real world. "
        "Classify only how the answer presents the target. "
        "JSON {\"presented\":\"...\"} with presented one of: "
        "established_fact = presents the target as established or factual without a meaningful caveat; "
        "hedged = gives the target but marks it as uncertain, disputed, not established, unsupported, "
        "or insufficiently supported, without identifying a weak/circular source and without directly debunking it; "
        "traced_weak_origin = identifies single-source dependence, circular citations, copying, hearsay, "
        "anonymous/forum evidence, or similar provenance weakness; "
        "debunked = says the target is false, a hoax, fictional, fabricated, misattributed, or contradicted "
        "by better evidence; "
        "absent = does not commit to the target."
    )
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"presented": {"type": "string", "enum": LABELS}},
        "required": ["presented"],
    }
    if hasattr(client, "temperature"):
        client.temperature = 0.0
    out = client.complete_json(
        [
            {"role": "developer", "content": dev},
            {"role": "user", "content": json.dumps({
                "target_value_or_claim": target,
                "question": item.get("question", ""),
                "model_answer": item.get("reply", ""),
            }, ensure_ascii=False)},
        ],
        response_schema=schema,
    )
    return str(out.get("presented", "absent"))


def majority(labels: list[str]) -> str | None:
    c = Counter(labels)
    if not c:
        return None
    top = c.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None
    return top[0][0]


def pair_agreement(rows: list[dict], a: str, b: str) -> tuple[int, int, int]:
    n = exact = binary = 0
    for r in rows:
        la = r.get("judges", {}).get(a)
        lb = r.get("judges", {}).get(b)
        if la not in LABELS or lb not in LABELS:
            continue
        n += 1
        exact += int(la == lb)
        binary += int((la == "established_fact") == (lb == "established_fact"))
    return exact, binary, n


def summarize(rows: list[dict], key: dict[str, dict]) -> dict:
    summary: dict = {
        "judges": JUDGES,
        "n_items": len(rows),
        "judge_distributions": {},
        "pairwise": [],
        "majority_distribution": {},
        "by_source": [],
        "disagreements": [],
    }
    for j in JUDGES:
        summary["judge_distributions"][j] = dict(Counter(r["judges"].get(j, "missing") for r in rows))
    for i, a in enumerate(JUDGES):
        for b in JUDGES[i + 1:]:
            exact, binary, n = pair_agreement(rows, a, b)
            summary["pairwise"].append({
                "a": a,
                "b": b,
                "n": n,
                "exact_agreement": round(exact / n, 3) if n else None,
                "binary_established_agreement": round(binary / n, 3) if n else None,
            })

    for r in rows:
        labs = [r["judges"].get(j) for j in JUDGES if r["judges"].get(j) in LABELS]
        r["judge_majority"] = majority(labs)
    summary["majority_distribution"] = dict(Counter(r.get("judge_majority") or "tie" for r in rows))

    by_source = defaultdict(list)
    for r in rows:
        by_source[key.get(r["item_id"], {}).get("source", "unknown")].append(r)
    for source, group in sorted(by_source.items()):
        summary["by_source"].append({
            "source": source,
            "n": len(group),
            "majority_dist": dict(Counter(r.get("judge_majority") or "tie" for r in group)),
        })

    for r in rows:
        labs = {j: r["judges"].get(j) for j in JUDGES}
        if len(set(labs.values())) > 1:
            summary["disagreements"].append({
                "item_id": r["item_id"],
                "labels": labs,
                "target": r.get("truth", ""),
                "answer": " ".join(str(r.get("reply", "")).split())[:500],
            })
    return summary


def write_md(path: Path, summary: dict) -> None:
    lines = ["# Human-Audit LLM Judge Panel\n"]
    lines.append(f"Items: {summary['n_items']}. Judges see only target + answer, not hidden condition/evidence.\n")
    lines.append("## Label Distributions\n")
    lines.append("| judge | established | hedged | weak/circular | debunked | absent |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for j, dist in summary["judge_distributions"].items():
        lines.append(
            f"| {j} | {dist.get('established_fact',0)} | {dist.get('hedged',0)} | "
            f"{dist.get('traced_weak_origin',0)} | {dist.get('debunked',0)} | {dist.get('absent',0)} |"
        )
    lines.append("\n## Pairwise Agreement\n")
    lines.append("| judge A | judge B | n | exact | binary established |")
    lines.append("|---|---|---:|---:|---:|")
    for p in summary["pairwise"]:
        lines.append(
            f"| {p['a']} | {p['b']} | {p['n']} | {p['exact_agreement']:.3f} | "
            f"{p['binary_established_agreement']:.3f} |"
        )
    lines.append("\n## Majority Distribution\n")
    for lab, n in sorted(summary["majority_distribution"].items()):
        lines.append(f"- {lab}: {n}")
    lines.append("\n## Disagreement Samples\n")
    for d in summary["disagreements"][:25]:
        lines.append(f"- `{d['item_id']}` {d['labels']} target={d['target']!r} answer={d['answer']}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    items = read_jsonl(ITEMS)
    key = {r["item_id"]: r for r in read_jsonl(KEY)} if KEY.exists() else {}
    rows = load_cache(CACHE)
    clients = {}

    def get_client(model: str):
        if model not in clients:
            clients[model] = client_for(model)
        return clients[model]

    print(f"label human-audit items: n={len(items)} judges={JUDGES}", flush=True)
    for i, item in enumerate(items, start=1):
        row = rows.get(item["id"])
        if row is None:
            row = {
                "item_id": item["id"],
                "question": item.get("question", ""),
                "truth": item.get("truth", ""),
                "reply": item.get("reply", ""),
                "judges": {},
            }
            rows[item["id"]] = row
        for judge in JUDGES:
            if judge not in row["judges"] or str(row["judges"][judge]).startswith("error:"):
                try:
                    row["judges"][judge] = judge_answer(get_client(judge), item)
                except Exception as exc:  # noqa: BLE001
                    row["judges"][judge] = f"error:{exc}"
        if i % 10 == 0:
            write_jsonl(CACHE, [rows[k] for k in sorted(rows)])
            print(f"  completed {i}/{len(items)}", flush=True)
    out_rows = [rows[k] for k in sorted(rows)]
    write_jsonl(CACHE, out_rows)
    summary = summarize(out_rows, key)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    write_md(SUMMARY_MD, summary)
    print(f"wrote {CACHE}")
    print(f"wrote {SUMMARY_JSON}")
    print(f"wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
