"""Summarize external-benchmark extension results."""
import json
import os


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "outputs", "society")
BENCHES = ["expertqa", "alce", "ragtruth"]
MODELS = ["gpt-4o-mini", "claude-opus-4-8"]


def read_json(path):
    with open(path) as f:
        return json.load(f)


def citation_path(bench, model):
    return os.path.join(OUTDIR, f"research_citation_multi_{bench}_{model}.json")


def handoff_path(bench):
    return os.path.join(OUTDIR, f"prov_handoff_{bench}.json")


def native_path(bench, model):
    return os.path.join(OUTDIR, f"external_native_{bench}_{model}.json")


def shas(paths):
    import hashlib

    out = {}
    for path in paths:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        out[os.path.relpath(path, ROOT)] = h.hexdigest()
    return out


def citation_summary():
    rows = []
    for bench in BENCHES:
        for model in MODELS:
            data = read_json(citation_path(bench, model))["results"]
            rows.append({
                "benchmark": bench,
                "model": model,
                "trace": data["trace"]["launder_n"],
                "flagged": data["flagged"]["launder_n"],
                "solid": data["solid"]["launder_n"],
                "discrimination": data.get("discrimination"),
            })
    return rows


def handoff_summary():
    rows = []
    for bench in BENCHES:
        data = read_json(handoff_path(bench))
        summary = data["carriers"]["gpt-4o-mini"]["summary"]
        typer_fields = data["typer_fields"].values() if isinstance(data["typer_fields"], dict) else data["typer_fields"]
        rows.append({
            "benchmark": bench,
            "typer_weak_circular": sum(
                1 for row in typer_fields
                if row.get("source_type") in {"single_unverified_origin", "circular_reporting"}
                or row.get("field") in {"single_unverified_origin", "circular_reporting"}
            ),
            "n": len(data["typer_fields"]),
            "self_downstream": summary["self_downstream"]["launder_n"],
            "self_carry": summary["self_carry"]["launder_n"],
            "typed_handoff": summary["typed_handoff"]["launder_n"],
            "own_field_correct": data["carriers"]["gpt-4o-mini"].get("own_field_correct"),
        })
    return rows


def native_summary():
    rows = []
    for bench in BENCHES:
        for model in MODELS:
            data = read_json(native_path(bench, model))["summary"]["by_benchmark"][bench]
            row = {"benchmark": bench, "model": model, "n": data["n"]}
            if "match_or_partial" in data:
                row.update({"metric": "match_or_partial", "value": data["match_or_partial"],
                            "exact_match": data["match"]})
            else:
                row.update({"metric": "accuracy", "value": data["accuracy"],
                            "unsupported_recall": data["unsupported_recall"],
                            "supported_specificity": data["supported_specificity"]})
            rows.append(row)
    return rows


def write_md(md_path, summary):
    lines = []
    lines.append("# External Benchmark Extension Summary\n")
    lines.append("## Seeded LaunderBench Panels\n")
    lines.append("| benchmark | model | trace | flagged | solid | discrimination |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in summary["citation"]:
        lines.append(
            f"| {row['benchmark']} | {row['model']} | {row['trace']} | {row['flagged']} | "
            f"{row['solid']} | {row['discrimination']:.2f} |"
        )
    lines.append("\n## Type-Once-Carry Handoff\n")
    lines.append("| benchmark | Opus typer weak/circular | mini self-downstream | mini self-carry | Opus typed-handoff | mini own field correct |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in summary["handoff"]:
        lines.append(
            f"| {row['benchmark']} | {row['typer_weak_circular']}/{row['n']} | "
            f"{row['self_downstream']} | {row['self_carry']} | {row['typed_handoff']} | "
            f"{row['own_field_correct']} |"
        )
    lines.append("\n## Original Native-Control Panels\n")
    lines.append("| benchmark | model | metric | value | extra |")
    lines.append("|---|---|---|---:|---|")
    for row in summary["native"]:
        if row["metric"] == "match_or_partial":
            extra = f"exact={row['exact_match']:.2f}"
        else:
            extra = f"recall={row['unsupported_recall']:.2f}, specificity={row['supported_specificity']:.2f}"
        lines.append(f"| {row['benchmark']} | {row['model']} | {row['metric']} | {row['value']:.2f} | {extra} |")
    lines.append("\n## Frozen Inputs\n")
    for frozen_path, digest in summary["sha256"].items():
        lines.append(f"- `{frozen_path}`: `{digest}`")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    generated = []
    for bench in BENCHES:
        generated.extend([
            os.path.join(OUTDIR, f"seeds_{bench}.jsonl"),
            os.path.join(OUTDIR, f"citation_scenarios_{bench}.jsonl"),
            os.path.join(OUTDIR, f"citation_scenarios_{bench}_v2.jsonl"),
            os.path.join(OUTDIR, f"native_cases_{bench}.jsonl"),
        ])
    summary = {
        "citation": citation_summary(),
        "handoff": handoff_summary(),
        "native": native_summary(),
        "sha256": shas(generated),
    }
    json_path = os.path.join(OUTDIR, "external_extension_summary.json")
    md_path = os.path.join(OUTDIR, "external_extension_summary.md")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    write_md(md_path, summary)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
