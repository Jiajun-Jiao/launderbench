#!/usr/bin/env python3
"""Summarize the 100-item external benchmark extension runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs" / "society"
BENCHES = ["expertqa", "alce", "ragtruth"]
MODELS = ["gpt-4o-mini", "claude-opus-4-8"]


def safe(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def sha(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def citation_summary() -> list[dict]:
    rows = []
    for bench in BENCHES:
        for model in MODELS:
            data = read_json(OUTDIR / f"research_citation_multi_{bench}100_{safe(model)}.json")
            if not data:
                continue
            res = data["results"]
            rows.append(
                {
                    "benchmark": f"{bench}100",
                    "model": model,
                    "trace": res["trace"]["launder_n"],
                    "flagged": res["flagged"]["launder_n"],
                    "solid": res["solid"]["launder_n"],
                    "discrimination": res.get("discrimination"),
                }
            )
    return rows


def handoff_summary() -> list[dict]:
    rows = []
    for bench in BENCHES:
        data = read_json(OUTDIR / f"prov_handoff_{bench}100.json")
        if not data or "gpt-4o-mini" not in data.get("carriers", {}):
            continue
        summary = data["carriers"]["gpt-4o-mini"]["summary"]
        typer_fields = data.get("typer_fields", {})
        rows.append(
            {
                "benchmark": f"{bench}100",
                "typer_weak_circular": sum(
                    1
                    for row in typer_fields.values()
                    if row.get("source_type") in {"single_unverified_origin", "circular_reporting"}
                ),
                "n": len(typer_fields),
                "self_downstream": summary["self_downstream"]["launder_n"],
                "self_carry": summary["self_carry"]["launder_n"],
                "typed_handoff": summary["typed_handoff"]["launder_n"],
                "own_field_correct": data["carriers"]["gpt-4o-mini"].get("own_field_correct"),
            }
        )
    return rows


def native_summary() -> list[dict]:
    rows = []
    for bench in BENCHES:
        for model in MODELS:
            for wrapper in (False, True):
                suffix = "_citation_wrapper" if wrapper else ""
                data = read_json(OUTDIR / f"external_native_{bench}100{suffix}_{safe(model)}.json")
                if not data:
                    continue
                bsum = data["summary"]["by_benchmark"][bench]
                row = {
                    "benchmark": f"{bench}100",
                    "model": model,
                    "render": "citation_wrapper" if wrapper else "plain",
                    "n": bsum["n"],
                }
                if "match_or_partial" in bsum:
                    row.update({"metric": "match_or_partial", "value": bsum["match_or_partial"], "exact_match": bsum["match"]})
                else:
                    row.update(
                        {
                            "metric": "accuracy",
                            "value": bsum["accuracy"],
                            "unsupported_recall": bsum["unsupported_recall"],
                            "supported_specificity": bsum["supported_specificity"],
                        }
                    )
                rows.append(row)
    return rows


def write_md(path: Path, summary: dict) -> None:
    lines = ["# External Benchmark Extension Summary (n=100)", ""]
    lines.extend(["## Seeded LaunderBench Panels", ""])
    lines.append("| benchmark | model | trace | flagged | solid | discrimination |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in summary["citation"]:
        lines.append(f"| {row['benchmark']} | {row['model']} | {row['trace']} | {row['flagged']} | {row['solid']} | {row['discrimination']:.2f} |")
    lines.extend(["", "## Type-Once-Carry Handoff", ""])
    lines.append("| benchmark | Opus typer weak/circular | mini self-downstream | mini self-carry | Opus typed-handoff | mini own field correct |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in summary["handoff"]:
        lines.append(
            f"| {row['benchmark']} | {row['typer_weak_circular']}/{row['n']} | {row['self_downstream']} | "
            f"{row['self_carry']} | {row['typed_handoff']} | {row['own_field_correct']} |"
        )
    lines.extend(["", "## Original Native-Control Panels", ""])
    lines.append("| benchmark | model | render | metric | value | extra |")
    lines.append("|---|---|---|---|---:|---|")
    for row in summary["native"]:
        extra = (
            f"exact={row['exact_match']:.2f}"
            if row["metric"] == "match_or_partial"
            else f"recall={row['unsupported_recall']:.2f}, specificity={row['supported_specificity']:.2f}"
        )
        lines.append(f"| {row['benchmark']} | {row['model']} | {row['render']} | {row['metric']} | {row['value']:.2f} | {extra} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    generated = []
    for bench in BENCHES:
        generated.extend(
            [
                OUTDIR / f"citation_scenarios_{bench}_100.jsonl",
                OUTDIR / f"citation_scenarios_{bench}_100_v2.jsonl",
                OUTDIR / f"native_cases_{bench}_100.jsonl",
            ]
        )
    summary = {
        "citation": citation_summary(),
        "handoff": handoff_summary(),
        "native": native_summary(),
        "sha256": {str(path.relative_to(ROOT)): digest for path in generated if (digest := sha(path))},
    }
    json_path = OUTDIR / "external100_extension_summary.json"
    md_path = OUTDIR / "external100_extension_summary.md"
    json_path.write_text(json.dumps(summary, indent=2))
    write_md(md_path, summary)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
