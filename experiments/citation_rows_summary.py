"""Recompute citation-panel rates from persisted row records without model API calls.

The denominator is always the number of valid answers. Rows marked ``api_error`` (including
model-level refusals) remain visible as attempted/excluded counts, but are never silently scored as
the substantive ``absent`` class.

Example:
  python3 experiments/citation_rows_summary.py \
    --scenarios outputs/society/citation_scenarios_expertqa_v2.jsonl \
    --rows outputs/society/research_citation_multi_expertqa_genuine_claude-opus-4-8.rows.jsonl
"""
import argparse
import hashlib
import json
from pathlib import Path

try:  # direct script execution puts experiments/ on sys.path; package import uses the relative form
    from research_citation_multi import aggregate_rows
except ModuleNotFoundError:  # pragma: no cover - exercised by unit-test/package import
    from .research_citation_multi import aggregate_rows


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def unique_rows(rows):
    """Reject duplicate run keys instead of choosing one invisibly."""
    seen = set()
    for row in rows:
        key = (row.get("model"), row["cond"], row["scenario_id"], int(row.get("rep", 0)))
        if key in seen:
            raise ValueError(f"duplicate row key: {key}")
        seen.add(key)
    return rows


def build_summary(rows_path, scenarios_path):
    rows = unique_rows(read_jsonl(rows_path))
    scenarios = read_jsonl(scenarios_path)
    scenario_by_id = {scenario["id"]: scenario for scenario in scenarios}
    missing = sorted({row["scenario_id"] for row in rows} - set(scenario_by_id))
    if missing:
        raise ValueError(f"{len(missing)} row scenario ids missing from scenario file: {missing[:5]}")
    preferred = ["trace", "flagged", "solid"]
    present = {row["cond"] for row in rows}
    conds = [cond for cond in preferred if cond in present] + sorted(present - set(preferred))
    results, _ = aggregate_rows(rows, conds, scenario_by_id)
    return {
        "schema_version": 1,
        "denominator_policy": "valid_answers_excluding_api_errors",
        "rows": str(rows_path),
        "rows_sha256": sha256(rows_path),
        "scenarios": str(scenarios_path),
        "scenarios_sha256": sha256(scenarios_path),
        "row_count": len(rows),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True, help="persisted *.rows.jsonl file")
    parser.add_argument("--scenarios", type=Path, required=True, help="scenario JSONL used by the run")
    parser.add_argument("--output", type=Path, help="optional JSON output; stdout is always printed")
    args = parser.parse_args()
    summary = build_summary(args.rows, args.scenarios)
    rendered = json.dumps(summary, indent=2, ensure_ascii=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
