"""Summarize citation-rendering ablation results."""
import json
import os


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "outputs", "society")
BENCHES = ["expertqa", "alce", "ragtruth"]
MODELS = ["gpt-4o-mini", "claude-opus-4-8"]
STYLES = {
    "explicit": "explicit",
    "domain_refs": "domain_refs",
}


def path_for(bench, model, style):
    suffix = "" if style == "explicit" else f"_{style}"
    return os.path.join(OUTDIR, f"research_citation_multi_{bench}{suffix}_{model}.json")


def load(path):
    with open(path) as f:
        return json.load(f)["results"]


def summarize():
    rows = []
    for bench in BENCHES:
        for model in MODELS:
            row = {"benchmark": bench, "model": model}
            for style in STYLES:
                data = load(path_for(bench, model, style))
                row[style] = {
                    "trace": data["trace"]["launder_n"],
                    "flagged": data["flagged"]["launder_n"],
                    "solid": data["solid"]["launder_n"],
                    "discrimination": data.get("discrimination"),
                }
            rows.append(row)
    return rows


def write_md(path, rows):
    lines = [
        "# Rendering Ablation Summary",
        "",
        "`explicit` is the original `[P1] ... (cites P2)` rendering. `domain_refs` removes page-id citation syntax and renders each result as a source domain, excerpt, and `References listed by this page: <domain>`.",
        "",
        "| benchmark | model | explicit trace | domain_refs trace | explicit flagged | domain_refs flagged | explicit solid | domain_refs solid |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        e, d = row["explicit"], row["domain_refs"]
        lines.append(
            f"| {row['benchmark']} | {row['model']} | {e['trace']} | {d['trace']} | "
            f"{e['flagged']} | {d['flagged']} | {e['solid']} | {d['solid']} |"
        )
    lines.extend([
        "",
        "Main read: replacing explicit page-id cites with domain-level references does not remove the core split. GPT-4o-mini still launders circular sourcing; Opus still launders 0/30 on trace across all three panels.",
    ])
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    rows = summarize()
    json_path = os.path.join(OUTDIR, "rendering_ablation_summary.json")
    md_path = os.path.join(OUTDIR, "rendering_ablation_summary.md")
    with open(json_path, "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    write_md(md_path, rows)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
