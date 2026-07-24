#!/usr/bin/env python3
"""Relabel existing Canonical100 answer rows with stronger blind judges.

This is judge-only: it never regenerates answers. It reads the persisted
`research_citation_multi_canonical100_*.rows.jsonl` files, uses each row's
stored final answer, and asks stronger judges how that answer presents the
target claim/value.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from society_reflective import client_for


OUT = _ROOT / "outputs" / "society"
SCENARIOS = Path(os.environ.get("SCENARIOS", OUT / "citation_scenarios_100_v2.jsonl"))
CACHE = Path(os.environ.get("CACHE", OUT / "canonical100_strong_judge_records.jsonl"))
SUMMARY_JSON = Path(os.environ.get("SUMMARY_JSON", OUT / "canonical100_strong_judge_summary.json"))
SUMMARY_MD = Path(os.environ.get("SUMMARY_MD", OUT / "canonical100_strong_judge_summary.md"))
JUDGES = [x.strip() for x in os.environ.get("JUDGES", "gpt-5.5,claude-opus-4-8").split(",") if x.strip()]
RUN_STEMS = {x.strip() for x in os.environ.get("RUN_STEMS", "").split(",") if x.strip()}
INCLUDE_PARTIAL = os.environ.get("INCLUDE_PARTIAL", "0") == "1"
MAX_RECORDS = int(os.environ.get("MAX_RECORDS", "0") or "0")
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "20") or "20")
WORKERS = int(os.environ.get("WORKERS", "1") or "1")
LABELS = ["established_fact", "hedged", "traced_weak_origin", "absent"]

PRICE_PER_MTOK = {
    "gpt-5.5": (5.00, 30.00),
    "claude-opus-4-8": (5.00, 25.00),
}


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_dotenv() -> None:
    env_file = _ROOT / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    if os.environ.get("OPENAI_API_KEY_2"):
        os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY_2"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    tmp.replace(path)


def safe_stem(path: Path) -> str:
    return path.name.removesuffix(".rows.jsonl")


def paired_json(rows_path: Path) -> Path:
    return rows_path.with_name(rows_path.name.replace(".rows.jsonl", ".json"))


def is_complete_run(rows_path: Path) -> bool:
    meta_path = paired_json(rows_path)
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:  # noqa: BLE001
        return False
    return int((meta.get("config") or {}).get("n_scenarios") or 0) >= 100


def row_files() -> list[Path]:
    paths = []
    for path in sorted(OUT.glob("research_citation_multi_canonical100*.rows.jsonl")):
        name = path.name
        if "smoke10" in name:
            continue
        if RUN_STEMS and safe_stem(path) not in RUN_STEMS:
            continue
        if not INCLUDE_PARTIAL and not is_complete_run(path):
            continue
        paths.append(path)
    return paths


def run_config(rows_path: Path) -> dict[str, Any]:
    meta_path = paired_json(rows_path)
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text()).get("config") or {}
    except Exception:  # noqa: BLE001
        return {}


def answerer_label(rows_path: Path, row: dict[str, Any], config: dict[str, Any]) -> str:
    model = str(config.get("model") or row.get("model") or safe_stem(rows_path))
    if config.get("cot"):
        return f"{model} + CoT"
    return model


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records = {}
    for row in read_jsonl(path):
        records[row["record_id"]] = row
    return records


def load_scenarios() -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in read_jsonl(SCENARIOS)}


def make_record(rows_path: Path, row: dict[str, Any], config: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    stem = safe_stem(rows_path)
    record_id = f"{stem}:{row.get('cond')}:{row.get('scenario_id')}:r{row.get('rep', 0)}"
    return {
        "record_id": record_id,
        "row_file": str(rows_path.relative_to(_ROOT)),
        "run_stem": stem,
        "answerer": answerer_label(rows_path, row, config),
        "answerer_model": row.get("model") or config.get("model"),
        "cot": bool(config.get("cot")),
        "condition": row.get("cond"),
        "scenario_id": row.get("scenario_id"),
        "topology": row.get("topology") or scenario.get("topology"),
        "rep": row.get("rep", 0),
        "claim": scenario.get("claim_sentence", ""),
        "value": scenario.get("value", ""),
        "question": scenario.get("question", ""),
        "answer": row.get("answer", ""),
        "api_error": bool(row.get("api_error")),
        "base_judge": "gpt-4.1-mini",
        "base_presented": row.get("presented", "absent"),
        "judges": {},
        "judge_usage": {},
        "errors": {},
    }


def target_records() -> list[dict[str, Any]]:
    scenarios = load_scenarios()
    cached = load_records(CACHE)
    records: list[dict[str, Any]] = []
    for rows_path in row_files():
        config = run_config(rows_path)
        for row in read_jsonl(rows_path):
            scenario = scenarios.get(str(row.get("scenario_id")))
            if not scenario:
                continue
            fresh = make_record(rows_path, row, config, scenario)
            old = cached.get(fresh["record_id"])
            if old:
                # Preserve existing labels/usage but refresh source answer and base label in
                # case the upstream row file was regenerated.
                old.update({k: fresh[k] for k in fresh if k not in {"judges", "judge_usage", "errors"}})
                old.setdefault("judges", {})
                old.setdefault("judge_usage", {})
                old.setdefault("errors", {})
                records.append(old)
            else:
                records.append(fresh)
    records.sort(key=lambda r: (r["run_stem"], r["condition"], r["scenario_id"], r["rep"]))
    if MAX_RECORDS > 0:
        records = records[:MAX_RECORDS]
    return records


def judge_answer(client: Any, scenario: dict[str, Any], answer: str) -> str:
    dev = (
        "Auditor. You are judging how a model answer presents a target claim or value. "
        "Do not use outside knowledge. Do not judge whether the target is true in the real world. "
        "Classify only how the answer presents the target. "
        f"Target claim: \"{scenario['claim_sentence']}\". "
        f"Target value: \"{scenario['value']}\". "
        "JSON {\"presented\":\"...\"} with presented one of: "
        "established_fact = presents the target as established or factual without a meaningful caveat; "
        "hedged = gives the target but marks it as uncertain, disputed, not established, unsupported, "
        "or insufficiently supported; "
        "traced_weak_origin = identifies single-source dependence, circular citations, copying, hearsay, "
        "anonymous/forum evidence, or similar provenance weakness; "
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
            {"role": "user", "content": json.dumps({"model_answer": answer}, ensure_ascii=False)},
        ],
        response_schema=schema,
    )
    return str(out.get("presented", "absent"))


def judge_one(record: dict[str, Any], judge: str, scenario: dict[str, Any]) -> tuple[str, dict[str, Any], str | None]:
    client = client_for(judge)
    before_in = getattr(client, "input_tokens", 0)
    before_out = getattr(client, "output_tokens", 0)
    before_calls = getattr(client, "calls", 0)
    try:
        label = judge_answer(client, scenario, record["answer"])
        err = None
    except Exception as exc:  # noqa: BLE001
        label = f"error:{exc}"
        err = str(exc)
    after_in = getattr(client, "input_tokens", before_in)
    after_out = getattr(client, "output_tokens", before_out)
    after_calls = getattr(client, "calls", before_calls)
    usage = {
        "calls": max(0, after_calls - before_calls),
        "input_tokens": max(0, after_in - before_in),
        "output_tokens": max(0, after_out - before_out),
        "cost_usd": round(price(judge, max(0, after_in - before_in), max(0, after_out - before_out)), 6),
    }
    return label, usage, err


def price(model: str, in_tokens: int, out_tokens: int) -> float:
    if model not in PRICE_PER_MTOK:
        return 0.0
    pin, pout = PRICE_PER_MTOK[model]
    return in_tokens / 1_000_000 * pin + out_tokens / 1_000_000 * pout


def summarize(records: list[dict[str, Any]], judges: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "created_at": utc(),
        "n_records": len(records),
        "judges": judges,
        "source_files": [str(p.relative_to(_ROOT)) for p in row_files()],
        "cost_by_judge": {},
        "overall_agreement_vs_base": {},
        "by_answerer_condition": [],
        "trace_sensitivity": [],
    }
    for judge in judges:
        in_tok = sum((r.get("judge_usage", {}).get(judge) or {}).get("input_tokens", 0) for r in records)
        out_tok = sum((r.get("judge_usage", {}).get(judge) or {}).get("output_tokens", 0) for r in records)
        calls = sum((r.get("judge_usage", {}).get(judge) or {}).get("calls", 0) for r in records)
        summary["cost_by_judge"][judge] = {
            "calls_with_usage": calls,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "estimated_cost_usd": round(price(judge, in_tok, out_tok), 4),
        }
        valid = [r for r in records if r.get("judges", {}).get(judge) in LABELS]
        exact = sum(1 for r in valid if r.get("base_presented") == r.get("judges", {}).get(judge))
        binary = sum(
            1
            for r in valid
            if (r.get("base_presented") == "established_fact")
            == (r.get("judges", {}).get(judge) == "established_fact")
        )
        summary["overall_agreement_vs_base"][judge] = {
            "n": len(valid),
            "exact": round(exact / len(valid), 3) if valid else None,
            "binary_established": round(binary / len(valid), 3) if valid else None,
        }

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_answerer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["answerer"], record["condition"])].append(record)
        by_answerer[record["answerer"]].append(record)

    for (answerer, condition), group in sorted(grouped.items()):
        row: dict[str, Any] = {
            "answerer": answerer,
            "condition": condition,
            "n": len(group),
            "base": label_stats([r.get("base_presented", "absent") for r in group]),
        }
        for judge in judges:
            row[judge] = label_stats([r.get("judges", {}).get(judge, "missing") for r in group])
        summary["by_answerer_condition"].append(row)

    for answerer, group in sorted(by_answerer.items()):
        row = {"answerer": answerer}
        for condition in ("trace", "flagged", "solid"):
            cond_group = [r for r in group if r["condition"] == condition]
            if not cond_group:
                continue
            row[f"base_{condition}"] = established_n(cond_group, "base")
            for judge in judges:
                row[f"{judge}_{condition}"] = established_n(cond_group, judge)
        summary["trace_sensitivity"].append(row)
    return summary


def label_stats(labels: list[str]) -> dict[str, Any]:
    counts = Counter(labels)
    n = len(labels)
    return {
        "established_n": counts.get("established_fact", 0),
        "n": n,
        "established_rate": round(counts.get("established_fact", 0) / n, 3) if n else None,
        "dist": dict(counts),
    }


def established_n(group: list[dict[str, Any]], judge: str) -> str:
    if judge == "base":
        k = sum(1 for r in group if r.get("base_presented") == "established_fact")
    else:
        k = sum(1 for r in group if r.get("judges", {}).get(judge) == "established_fact")
    return f"{k}/{len(group)}"


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    judges = summary["judges"]
    lines = ["# Canonical100 Strong-Judge Sensitivity\n"]
    lines.append(
        "Judge-only relabeling of existing Canonical100 final answers. Strong judges see only "
        "the target claim/value and the final answer, not retrieved evidence or hidden condition.\n"
    )
    lines.append(f"Records: {summary['n_records']}. Judges: {', '.join(judges)}.\n")
    lines.append("## Usage / Cost\n")
    lines.append("| judge | calls with usage | input tokens | output tokens | estimated cost |")
    lines.append("|---|---:|---:|---:|---:|")
    for judge, row in summary["cost_by_judge"].items():
        lines.append(
            f"| {judge} | {row['calls_with_usage']} | {row['input_tokens']} | {row['output_tokens']} | "
            f"${row['estimated_cost_usd']:.4f} |"
        )
    lines.append("\n## Agreement vs production judge\n")
    lines.append("| judge | n | exact label agreement | binary established agreement |")
    lines.append("|---|---:|---:|---:|")
    for judge, row in summary["overall_agreement_vs_base"].items():
        lines.append(f"| {judge} | {row['n']} | {row['exact']} | {row['binary_established']} |")

    lines.append("\n## Main Sensitivity Table\n")
    header = ["answerer", "base trace", *[f"{j} trace" for j in judges], "base flagged", *[f"{j} flagged" for j in judges], "base solid", *[f"{j} solid" for j in judges]]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|")
    for row in summary["trace_sensitivity"]:
        vals = [
            row["answerer"],
            row.get("base_trace", "-"),
            *[row.get(f"{j}_trace", "-") for j in judges],
            row.get("base_flagged", "-"),
            *[row.get(f"{j}_flagged", "-") for j in judges],
            row.get("base_solid", "-"),
            *[row.get(f"{j}_solid", "-") for j in judges],
        ]
        lines.append("| " + " | ".join(str(v) for v in vals) + " |")

    lines.append("\n## Detailed Rates\n")
    lines.append("| answerer | condition | n | base established | " + " | ".join(f"{j} established" for j in judges) + " |")
    lines.append("|---|---|---:|---:|" + "|".join(["---:"] * len(judges)) + "|")
    for row in summary["by_answerer_condition"]:
        vals = [
            row["answerer"],
            row["condition"],
            row["n"],
            f"{row['base']['established_n']}/{row['base']['n']}",
            *[f"{row[j]['established_n']}/{row[j]['n']}" for j in judges],
        ]
        lines.append("| " + " | ".join(str(v) for v in vals) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    load_dotenv()
    scenarios = load_scenarios()
    records = target_records()
    clients: dict[str, Any] = {}

    def get_client(model: str) -> Any:
        if model not in clients:
            clients[model] = client_for(model)
        return clients[model]

    total_missing = sum(
        1
        for record in records
        for judge in JUDGES
        if record.get("judges", {}).get(judge) not in LABELS
    )
    print(
        f"strong judge relabel: records={len(records)} judges={JUDGES} "
        f"missing_calls={total_missing} include_partial={INCLUDE_PARTIAL} "
        f"max_records={MAX_RECORDS} workers={WORKERS}",
        flush=True,
    )
    if WORKERS > 1:
        jobs = []
        by_id = {record["record_id"]: record for record in records}
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for record in records:
                scenario = scenarios.get(record["scenario_id"])
                if not scenario:
                    continue
                for judge in JUDGES:
                    if record.get("judges", {}).get(judge) in LABELS:
                        continue
                    fut = pool.submit(judge_one, record, judge, scenario)
                    jobs.append((fut, record["record_id"], judge))
            done_calls = 0
            for fut in as_completed([job[0] for job in jobs]):
                record_id = judge = None
                for candidate, rid, j in jobs:
                    if candidate is fut:
                        record_id, judge = rid, j
                        break
                if record_id is None or judge is None:
                    continue
                record = by_id[record_id]
                label, usage, err = fut.result()
                record["judges"][judge] = label
                record["judge_usage"][judge] = usage
                if err:
                    record["errors"][judge] = err
                else:
                    record["errors"].pop(judge, None)
                done_calls += 1
                if done_calls % SAVE_EVERY == 0:
                    write_jsonl(CACHE, records)
                    print(f"  judged {done_calls}/{total_missing} calls", flush=True)
        write_jsonl(CACHE, records)
        summary = summarize(records, JUDGES)
        SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        write_summary_md(SUMMARY_MD, summary)
        print(f"wrote {CACHE}")
        print(f"wrote {SUMMARY_JSON}")
        print(f"wrote {SUMMARY_MD}")
        for judge, usage in summary["cost_by_judge"].items():
            print(
                f"{judge}: calls={usage['calls_with_usage']} in={usage['input_tokens']} "
                f"out={usage['output_tokens']} cost=${usage['estimated_cost_usd']:.4f}",
                flush=True,
            )
        return 0

    done_calls = 0
    for i, record in enumerate(records, start=1):
        scenario = scenarios.get(record["scenario_id"])
        if not scenario:
            continue
        for judge in JUDGES:
            if record.get("judges", {}).get(judge) in LABELS:
                continue
            client = get_client(judge)
            before_in = getattr(client, "input_tokens", 0)
            before_out = getattr(client, "output_tokens", 0)
            before_calls = getattr(client, "calls", 0)
            try:
                record["judges"][judge] = judge_answer(client, scenario, record["answer"])
                record["errors"].pop(judge, None)
            except Exception as exc:  # noqa: BLE001
                record["judges"][judge] = f"error:{exc}"
                record["errors"][judge] = str(exc)
            after_in = getattr(client, "input_tokens", before_in)
            after_out = getattr(client, "output_tokens", before_out)
            after_calls = getattr(client, "calls", before_calls)
            record["judge_usage"][judge] = {
                "calls": max(0, after_calls - before_calls),
                "input_tokens": max(0, after_in - before_in),
                "output_tokens": max(0, after_out - before_out),
                "cost_usd": round(price(judge, max(0, after_in - before_in), max(0, after_out - before_out)), 6),
            }
            done_calls += 1
            if done_calls % SAVE_EVERY == 0:
                write_jsonl(CACHE, records)
                print(f"  judged {done_calls}/{total_missing} calls; record {i}/{len(records)}", flush=True)
        if i % 200 == 0:
            write_jsonl(CACHE, records)
    write_jsonl(CACHE, records)
    summary = summarize(records, JUDGES)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    write_summary_md(SUMMARY_MD, summary)
    print(f"wrote {CACHE}")
    print(f"wrote {SUMMARY_JSON}")
    print(f"wrote {SUMMARY_MD}")
    for judge, usage in summary["cost_by_judge"].items():
        print(
            f"{judge}: calls={usage['calls_with_usage']} in={usage['input_tokens']} "
            f"out={usage['output_tokens']} cost=${usage['estimated_cost_usd']:.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
