"""Aggregate provenance_mechanism_diagnostic.py outputs into a mechanism-gradient table."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs" / "society"
DEFAULT_INPUTS = [
    OUTDIR / "provenance_mechanism_activation_gpt4omini.json",
    OUTDIR / "provenance_mechanism_activation_gpt41mini.json",
    OUTDIR / "provenance_mechanism_activation_claude_haiku.json",
]
DEFAULT_OUT_JSON = OUTDIR / "provenance_mechanism_gradient.json"
DEFAULT_OUT_MD = OUTDIR / "provenance_mechanism_gradient.md"


def env_paths(name: str, defaults: list[Path]) -> list[Path]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return defaults
    out = []
    for part in raw.split(","):
        p = Path(part.strip())
        out.append(p if p.is_absolute() else ROOT / p)
    return out


def read_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        data = json.loads(path.read_text())
        for model, summary in data["summaries"].items():
            step = summary["step_accuracy"]
            ints = summary["interventions"]
            rows.append(
                {
                    "model": model,
                    "n": summary["n"],
                    "cited_edges": step["cited_edge_accuracy"],
                    "root_set": step["root_set_correct"],
                    "source_type": step["source_type_correct"],
                    "raw": ints["raw"]["launder_n"],
                    "instruction": ints.get("instruction", {}).get("launder_n", "n/a"),
                    "diagnose_answer": ints.get("diagnose_answer", {}).get("launder_n", "n/a"),
                    "self_structured": ints.get("self_structured", {}).get("launder_n", "n/a"),
                    "graph": ints["graph"]["launder_n"],
                    "roots": ints["roots"]["launder_n"],
                    "type": ints["type"]["launder_n"],
                    "license": ints["license"]["launder_n"],
                    "raw_launder_with_stepwise_success": summary["raw_launder_with_stepwise_success"],
                    "first_rescue_arm": summary["first_rescue_arm"],
                    "source_file": str(path.relative_to(ROOT)),
                }
            )
    return rows


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = ["# Provenance Mechanism Gradient", ""]
    lines.append("| model | n | cited-edge acc | root-set acc | source-type acc | raw | instruction | diagnose+answer | self-structured | graph | roots | type | license |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['n']} | {r['cited_edges']:.3f} | {r['root_set']:.3f} | "
            f"{r['source_type']:.3f} | {r['raw']} | {r['instruction']} | {r['diagnose_answer']} | "
            f"{r['self_structured']} | {r['graph']} | {r['roots']} | {r['type']} | {r['license']} |"
        )
    lines.append("")
    lines.append("## Raw Laundering Despite Fully Correct Explicit Diagnostic")
    lines.append("")
    lines.append("| model | raw laundering with explicit diagnostic success | first rescue arm counts |")
    lines.append("|---|---:|---|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['raw_launder_with_stepwise_success']} | "
            f"{json.dumps(r['first_rescue_arm'], sort_keys=True)} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    paths = env_paths("INPUTS", DEFAULT_INPUTS)
    rows = read_rows(paths)
    out_json = Path(os.environ.get("OUT_JSON", str(DEFAULT_OUT_JSON)))
    out_md = Path(os.environ.get("OUT_MD", str(DEFAULT_OUT_MD)))
    if not out_json.is_absolute():
        out_json = ROOT / out_json
    if not out_md.is_absolute():
        out_md = ROOT / out_md
    out_json.write_text(json.dumps({"rows": rows}, indent=2))
    write_markdown(out_md, rows)
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
