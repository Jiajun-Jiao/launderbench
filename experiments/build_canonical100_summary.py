#!/usr/bin/env python3
"""Rebuild the Canonical100 summary from completed research_citation_multi outputs."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs" / "society"
SCENARIOS = OUTDIR / "citation_scenarios_100_v2.jsonl"
DEFAULT_MODELS = [
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4.1",
    "gpt-5.5",
    "o3",
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-8",
    "gemini-2.5-flash",
    "qwen2.5:32b-instruct",
    "qwen3:30b-a3b-instruct-2507-q4_K_M",
]


def safe(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def panel_info() -> dict:
    if not SCENARIOS.exists():
        return {"rows": 0, "solid_ok": 0, "solid_missing": []}
    rows = [json.loads(line) for line in SCENARIOS.read_text().splitlines() if line.strip()]
    missing = [row["id"] for row in rows if not row.get("solid_pages")]
    return {"rows": len(rows), "solid_ok": len(rows) - len(missing), "solid_missing": missing}


def paired_discrimination(model: str) -> dict | None:
    """Compute the preferred score directly from valid trace/solid row pairs."""
    rows_path = OUTDIR / f"research_citation_multi_canonical100_{safe(model)}.rows.jsonl"
    if not rows_path.exists() or not SCENARIOS.exists():
        return None
    candidate_ids = {
        row["id"]
        for row in (json.loads(line) for line in SCENARIOS.read_text().splitlines() if line.strip())
        if row.get("solid_pages")
    }
    values = {}
    for row in (json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()):
        if row.get("api_error") or row.get("cond") not in {"trace", "solid"}:
            continue
        if row["scenario_id"] not in candidate_ids:
            continue
        key = (row["cond"], row["scenario_id"], int(row.get("rep", 0)))
        values[key] = int(row["presented"] == "established_fact")
    trace_keys = {(sid, rep) for cond, sid, rep in values if cond == "trace"}
    solid_keys = {(sid, rep) for cond, sid, rep in values if cond == "solid"}
    pairs = sorted(trace_keys & solid_keys)
    if not pairs:
        return None
    point = sum(
        values[("solid", sid, rep)] - values[("trace", sid, rep)] for sid, rep in pairs
    ) / len(pairs)
    return {"point": round(point, 6), "n_pairs": len(pairs), "method": "paired_matched_pairs"}


def model_rows(models: list[str]) -> list[dict]:
    out = []
    for model in models:
        path = OUTDIR / f"research_citation_multi_canonical100_{safe(model)}.json"
        data = load_json(path)
        if not data:
            continue
        res = data["results"]
        paired = paired_discrimination(model)
        legacy_discrimination = res.get("discrimination")
        row = {
            "model": data["config"]["model"],
            "trace_launder": res["trace"]["launder"],
            "trace_n": res["trace"]["launder_n"],
            "trace_ci95": res["trace"].get("ci95"),
            "flagged_launder": res["flagged"]["launder"],
            "flagged_n": res["flagged"]["launder_n"],
            "solid_established": res["solid"]["launder"],
            "solid_n": res["solid"]["launder_n"],
            "discrimination": paired["point"] if paired else legacy_discrimination,
            "discrimination_method": paired["method"] if paired else "legacy_unmatched_rates",
            "discrimination_pair_n": paired["n_pairs"] if paired else None,
            "discrimination_independent_legacy": legacy_discrimination,
            "api_errors": {
                "trace": res["trace"].get("api_errors", 0),
                "flagged": res["flagged"].get("api_errors", 0),
                "solid": res["solid"].get("api_errors", 0),
            },
            "trace_by_topology": res["trace"].get("by_topology", {}),
            "source_file": str(path.relative_to(ROOT)),
        }
        out.append(row)
    out.sort(key=lambda r: (-float(r.get("discrimination") or 0), float(r["trace_launder"])))
    return out


def write_md(path: Path, summary: dict) -> None:
    lines = ["# Canonical100 extension summary", ""]
    panel = summary["panel"]
    lines.append(
        "Panel: {rows} validated weak-origin scenarios; {solid_ok} have validated solid positive "
        "controls. Solid missing: {missing}.".format(
            rows=panel["rows"],
            solid_ok=panel["solid_ok"],
            missing=", ".join(panel["solid_missing"]) or "none",
        )
    )
    lines.extend([
        "",
        "| model | trace launder | trace 95% CI | flagged launder | solid established | discrimination (paired) | API errors |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary["models"]:
        lines.append(
            f"| {row['model']} | {row['trace_launder']:.2f} ({row['trace_n']}) | "
            f"{row['trace_ci95']} | {row['flagged_launder']:.2f} ({row['flagged_n']}) | "
            f"{row['solid_established']:.2f} ({row['solid_n']}) | {row['discrimination']:.2f} "
            f"(n={row['discrimination_pair_n']}) | "
            f"{row['api_errors']} |"
        )
    lines.append("")
    lines.append("Trace laundering by topology:")
    for row in summary["models"]:
        lines.append("")
        lines.append(f"## {row['model']}")
        for topology, entry in row.get("trace_by_topology", {}).items():
            lines.append(f"- {topology}: {entry['launder_n']} ({entry['rate']:.2f})")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    models = [x.strip() for x in os.environ.get("MODELS", ",".join(DEFAULT_MODELS)).split(",") if x.strip()]
    summary = {"panel": panel_info(), "models": model_rows(models)}
    json_path = OUTDIR / "canonical100_extension_summary.json"
    md_path = OUTDIR / "canonical100_extension_summary.md"
    json_path.write_text(json.dumps(summary, indent=2))
    write_md(md_path, summary)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
