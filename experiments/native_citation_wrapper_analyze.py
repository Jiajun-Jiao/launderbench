"""Compare original native-control inputs with a verbatim citation wrapper."""
import json
import os


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "outputs", "society")
BENCHES = ["expertqa", "alce", "ragtruth"]
MODELS = ["gpt-4o-mini", "claude-opus-4-8"]


def load(path):
    with open(path) as f:
        return json.load(f)


def path_for(bench, model, wrapper=False):
    extra = "_citation_wrapper" if wrapper else ""
    return os.path.join(OUTDIR, f"external_native_{bench}{extra}_{model}.json")


def bench_summary(data, bench):
    s = data["summary"]["by_benchmark"][bench]
    if "match_or_partial" in s:
        return {"metric": "match_or_partial", "value": s["match_or_partial"], "extra": f"exact={s['match']:.2f}"}
    return {
        "metric": "accuracy",
        "value": s["accuracy"],
        "extra": f"recall={s['unsupported_recall']:.2f}, specificity={s['supported_specificity']:.2f}",
    }


def summarize():
    rows = []
    for bench in BENCHES:
        for model in MODELS:
            plain = bench_summary(load(path_for(bench, model, wrapper=False)), bench)
            wrapped = bench_summary(load(path_for(bench, model, wrapper=True)), bench)
            rows.append({"benchmark": bench, "model": model, "plain": plain, "citation_wrapper": wrapped})
    return rows


def write_md(path, rows):
    lines = [
        "# Native Citation Wrapper Summary",
        "",
        "Original benchmark text is kept verbatim. The wrapper only adds `References listed by this source: ...` metadata to each original source/evidence item. Metrics remain each benchmark's native control metric, not LaunderBench laundering.",
        "",
        "| benchmark | model | metric | plain | citation_wrapper | extra plain | extra wrapper |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        p, w = row["plain"], row["citation_wrapper"]
        lines.append(
            f"| {row['benchmark']} | {row['model']} | {p['metric']} | {p['value']:.2f} | "
            f"{w['value']:.2f} | {p['extra']} | {w['extra']} |"
        )
    lines.append("")
    lines.append("Main read: adding lightweight citation metadata to original, non-rephrased benchmark inputs does not create the seeded LaunderBench split; native performance remains broadly similar.")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    rows = summarize()
    json_path = os.path.join(OUTDIR, "native_citation_wrapper_summary.json")
    md_path = os.path.join(OUTDIR, "native_citation_wrapper_summary.md")
    with open(json_path, "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    write_md(md_path, rows)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
