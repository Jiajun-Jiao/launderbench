"""origin_kind_breakdown.py — reviewer-requested slice: trace laundering by ORIGIN KIND.

The canonical 100 scenarios draw their weak origin from five kinds (fiction/satire later taken as
fact; anonymous forum guess that admits having no source; unsourced anonymous wiki edit; content-farm
SEO article stating it flatly; personal-blog hunch). This script joins each model's canonical100
row records with the scenario origin_kind and reports trace laundering per kind — does how
self-undermining the origin is change whether the echo chamber above it gets laundered?

Reads outputs/society/research_citation_multi_canonical100_<model>.rows.jsonl (skips _cot_ /
_genuine_ variants). Zero API calls.
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

KIND_KEY = [
    ("clearly-labelled fiction", "fiction_satire"),
    ("forum/message-board", "forum_guess"),
    ("community wiki", "wiki_edit"),
    ("content-farm", "content_farm"),
    ("personal blog", "blog_hunch"),
]
KIND_ORDER = ["forum_guess", "blog_hunch", "wiki_edit", "content_farm", "fiction_satire"]
# panel display order (matches tab:panel roughly: launderers -> mid -> cheap reasoning -> frontier -> open)
MODEL_ORDER = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "gpt-4.1", "gemini-2.5-flash",
               "o4-mini", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5-mini",
               "o3", "gpt-5.5", "claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-8",
               "qwen2.5_7b-instruct", "qwen2.5_32b-instruct", "qwen3_30b-a3b-instruct-2507-q4_K_M",
               "llama3.1_8b", "mistral_7b", "gemma2_9b"]


def kind_of(origin_kind):
    for needle, key in KIND_KEY:
        if needle in origin_kind:
            return key
    raise ValueError(f"unmapped origin_kind: {origin_kind!r}")


def main():
    scenarios = {}
    for line in open(os.path.join(_ROOT, "outputs", "society", "citation_scenarios_100_v2.jsonl")):
        if line.strip():
            sc = json.loads(line)
            scenarios[sc["id"]] = kind_of(sc["origin_kind"])

    per_model = {}
    pattern = os.path.join(_ROOT, "outputs", "society", "research_citation_multi_canonical100_*.rows.jsonl")
    for path in sorted(glob.glob(pattern)):
        base = os.path.basename(path)
        if "_cot_" in base or "_genuine" in base:
            continue
        model = re.sub(r"^research_citation_multi_canonical100_", "", base)[: -len(".rows.jsonl")]
        agg = defaultdict(lambda: [0, 0])  # kind -> [launder, n]
        for line in open(path):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("cond") != "trace" or int(row.get("rep", 0)) != 0:
                continue
            if row.get("scenario_id") not in scenarios:
                continue
            kind = scenarios[row["scenario_id"]]
            agg[kind][1] += 1
            agg[kind][0] += int(row.get("presented") == "established_fact")
        if agg:
            per_model[model] = {k: tuple(v) for k, v in agg.items()}

    ordered = [m for m in MODEL_ORDER if m in per_model] + sorted(set(per_model) - set(MODEL_ORDER))
    kind_n = defaultdict(int)
    for m in ordered:
        for k, (_, n) in per_model[m].items():
            kind_n[k] = max(kind_n[k], n)
    header = f"{'model':38s}" + "".join(f"{k:>16s}" for k in KIND_ORDER) + f"{'total':>10s}"
    print(header)
    out = {"kinds": {k: kind_n[k] for k in KIND_ORDER}, "models": {}}
    pooled = defaultdict(lambda: [0, 0])
    for m in ordered:
        cells = []
        tot_k = tot_n = 0
        for k in KIND_ORDER:
            kk, nn = per_model[m].get(k, (0, 0))
            cells.append(f"{kk}/{nn}" if nn else "-")
            tot_k += kk
            tot_n += nn
            pooled[k][0] += kk
            pooled[k][1] += nn
        print(f"{m:38s}" + "".join(f"{c:>16s}" for c in cells) + f"{f'{tot_k}/{tot_n}':>10s}")
        out["models"][m] = {k: f"{per_model[m].get(k, (0, 0))[0]}/{per_model[m].get(k, (0, 0))[1]}"
                            for k in KIND_ORDER}
        out["models"][m]["total"] = f"{tot_k}/{tot_n}"
    print(f"{'POOLED (all models)':38s}"
          + "".join(f"{f'{pooled[k][0]}/{pooled[k][1]}':>16s}" for k in KIND_ORDER))
    pooled_rates = {k: round(pooled[k][0] / pooled[k][1], 3) if pooled[k][1] else None for k in KIND_ORDER}
    print(f"{'pooled rate':38s}" + "".join(f"{pooled_rates[k]:>16.3f}" for k in KIND_ORDER))
    out["pooled"] = {k: f"{pooled[k][0]}/{pooled[k][1]}" for k in KIND_ORDER}
    out["pooled_rate"] = pooled_rates

    path = os.path.join(_ROOT, "outputs", "society", "origin_kind_breakdown.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    sys.exit(main())
