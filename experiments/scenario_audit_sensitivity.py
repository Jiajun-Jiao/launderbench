"""Any-rater-disagreement sensitivity for the Canonical100 trace panel.

This script removes every audited scenario on which any complete annotator
dissented from the majority-audited construct (whatever the current annotator
count in scenario_audit_summary.json) and recomputes the headline trace rates.
It is a strict sensitivity analysis, not a relabeling of the benchmark.
"""
import glob
import json
import os


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(ROOT, "outputs", "human_eval", "scenario_audit_summary.json")
OUT = os.path.join(ROOT, "outputs", "human_eval", "scenario_audit_sensitivity.json")
MODELS = (
    "gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "qwen2.5:32b-instruct", "gpt-4.1",
    "qwen3:30b-a3b-instruct-2507-q4_K_M", "gemini-2.5-flash", "o3",
    "claude-haiku-4-5", "gpt-5.5", "claude-sonnet-4-6", "claude-opus-4-8",
)


def load_rows():
    rows = {}
    pattern = os.path.join(ROOT, "outputs", "society", "research_citation_multi_canonical100_*.rows.jsonl")
    for path in glob.glob(pattern):
        first = None
        data = []
        for line in open(path):
            row = json.loads(line)
            first = first or row
            data.append(row)
        if first and first.get("model") in MODELS and all(token not in os.path.basename(path)
                                                          for token in ("_cot_", "_instr_")):
            rows[first["model"]] = data
    return rows


def rate(rows, excluded):
    valid = [row for row in rows if row.get("cond") == "trace" and not row.get("api_error")
             and row.get("scenario_id") not in excluded]
    k = sum(row.get("presented") == "established_fact" for row in valid)
    return {"established": k, "n": len(valid), "rate": round(k / len(valid), 4) if valid else None}


def main():
    summary = json.load(open(SUMMARY))
    excluded = set(summary.get("one_rater_dissent_scenario_ids", []))
    by_model = {}
    for model, rows in load_rows().items():
        full = rate(rows, set())
        filtered = rate(rows, excluded)
        by_model[model] = {"full": full, "without_one_rater_dissents": filtered,
                           "absolute_change": round(filtered["rate"] - full["rate"], 4)}
    result = {"excluded_scenario_ids": sorted(excluded), "n_excluded": len(excluded),
              "models": by_model,
              "max_absolute_change": max((abs(v["absolute_change"]) for v in by_model.values()), default=0.0)}
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
