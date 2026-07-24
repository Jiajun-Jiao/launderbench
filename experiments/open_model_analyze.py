#!/usr/bin/env python3
"""Summarize local/open-weight citation-laundering runs."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs" / "society"

CANONICAL100_MODELS = [
    "qwen2.5:32b-instruct",
    "qwen3:30b-a3b-instruct-2507-q4_K_M",
]

APPENDIX30_MODELS = [
    "qwen2.5:7b-instruct",
    "qwen2.5:32b-instruct",
    "llama3.1:8b",
    "qwen3:30b-a3b-instruct-2507-q4_K_M",
]

SMALL_CANONICAL100_MODELS = [
    "qwen2.5:7b-instruct",
    "llama3.1:8b",
    "mistral:7b",
    "gemma2:9b",
]


def safe(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def read_run(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    res = data["results"]
    row = {
        "model": data["config"]["model"],
        "n_scenarios": data["config"].get("n_scenarios"),
        "source_file": str(path.relative_to(ROOT)),
        "trace": res["trace"]["launder_n"],
        "trace_rate": res["trace"]["launder"],
        "flagged": res["flagged"]["launder_n"],
        "flagged_rate": res["flagged"]["launder"],
        "solid": res["solid"]["launder_n"],
        "solid_rate": res["solid"]["launder"],
        "discrimination": res.get("discrimination"),
        "trace_dist": res["trace"].get("dist", {}),
        "solid_dist": res["solid"].get("dist", {}),
    }
    row["read"] = interpretation(row)
    return row


def interpretation(row: dict) -> str:
    if row["solid_rate"] < 0.9:
        return "adherence/solid-control confounded"
    if row["flagged_rate"] > 0.05:
        return "blanket-trust / weak explicit-warning control"
    return "valid discriminator; trace laundering remains"


def collect(models: list[str], prefix: str) -> list[dict]:
    rows = []
    for model in models:
        row = read_run(OUTDIR / f"{prefix}_{safe(model)}.json")
        if row:
            rows.append(row)
    return rows


def write_table(lines: list[str], rows: list[dict]) -> None:
    lines.append("| model | n | trace | flagged | solid | discrimination | read |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['n_scenarios']} | {row['trace']} | {row['flagged']} | "
            f"{row['solid']} | {row['discrimination']:.2f} | {row['read']} |"
        )


def write_md(path: Path, summary: dict) -> None:
    lines = [
        "# Open-Weight Citation Panel Summary",
        "",
        "Local Ollama models as answerers, `gpt-4.1-mini` as the output classifier.",
        "",
        "## Canonical100 Open-Weight Rows",
        "",
    ]
    write_table(lines, summary["canonical100"])
    lines.extend([
        "",
        "Main-read rows are those that pass the explicit-warning and solid controls cleanly enough to "
        "compare against closed hosted models.",
        "",
        "## Smaller / Legacy Canonical100 Rows",
        "",
    ])
    write_table(lines, summary["small_canonical100"])
    lines.extend([
        "",
        "These are useful controls but should be read cautiously when solid or flagged controls are not clean.",
        "",
        "## Canonical30 Legacy Appendix",
        "",
    ])
    write_table(lines, summary["canonical30_appendix"])
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    canonical100 = collect(CANONICAL100_MODELS, "research_citation_multi_canonical100")
    small100 = collect(SMALL_CANONICAL100_MODELS, "research_citation_multi_canonical100")
    appendix30 = collect(APPENDIX30_MODELS, "research_citation_multi")
    summary = {
        "canonical100": canonical100,
        "small_canonical100": small100,
        "canonical30_appendix": appendix30,
    }
    json_path = OUTDIR / "open_model_summary.json"
    md_path = OUTDIR / "open_model_summary.md"
    json_path.write_text(json.dumps(summary, indent=2))
    write_md(md_path, summary)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
