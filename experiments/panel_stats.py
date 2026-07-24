"""Paired and clustered statistics for the canonical citation panel.

All models answer the SAME 100 trace scenarios, so model comparisons should exploit the paired
structure, and scenarios share a generation pipeline balanced over 20 domains x 3 topologies, so
scenario-level intervals should be checked against domain-clustered uncertainty. This script adds:

  1. Exact McNemar tests (two-sided binomial on discordant pairs) for the tier claims:
     each mini vs. its full-size counterpart, and each cheap/mid model vs. each zero-rate model.
  2. Paired percentile bootstrap CIs (10k resamples over the 96 scenarios that have both a
     trace treatment and its matched solid control) for the discrimination score
     P(est|solid) - P(est|trace) per model. The superseded independent-set bootstrap is retained
     under an explicitly named legacy key so old reports remain auditable.
  3. Domain-clustered bootstrap CIs (10k resamples of the 20 domains) for every trace rate,
     to compare against the scenario-level Wilson intervals in Table 2.
  4. A per-domain laundering breakdown (pooled non-reasoning launderers and per model).

Usage: python3 experiments/panel_stats.py
Output: outputs/society/panel_stats.json (+ printed summary)
"""
import json
import math
import os
import random

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get(
    "PANEL_STATS_OUT",
    os.path.join(_ROOT, "outputs", "society", "panel_stats.json"),
)
SCEN = os.path.join(_ROOT, "outputs", "society", "citation_scenarios_100_v2.jsonl")
B = 10000
SEED = 20260704
HEADLINE = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "qwen2.5_32b-instruct", "gpt-4.1", "qwen3_30b-a3b-instruct-2507-q4_K_M",
            "gemini-2.5-flash", "o3", "claude-haiku-4-5", "gpt-5.5", "claude-sonnet-4-6", "claude-opus-4-8"]
LAUNDERERS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "gpt-4.1", "qwen2.5_32b-instruct",
              "qwen3_30b-a3b-instruct-2507-q4_K_M"]
ZERO_ROWS = ["gpt-5.5", "claude-sonnet-4-6", "claude-opus-4-8", "claude-haiku-4-5", "o3"]
MINI_FULL = [("gpt-4o-mini", "gpt-4o"), ("gpt-4.1-mini", "gpt-4.1")]


def load(tag):
    path = os.path.join(_ROOT, "outputs", "society",
                        f"research_citation_multi_canonical100_{tag}.rows.jsonl")
    out = {}
    with open(path) as handle:
        for line in handle:
            r = json.loads(line)
            if r.get("api_error"):
                continue
            out[(r["cond"], r["scenario_id"])] = 1 if r["presented"] == "established_fact" else 0
    return out


def mcnemar_exact(a, b, ids):
    """Two-sided exact McNemar on paired binary outcomes over shared scenario ids."""
    n01 = sum(1 for i in ids if a[("trace", i)] == 0 and b[("trace", i)] == 1)
    n10 = sum(1 for i in ids if a[("trace", i)] == 1 and b[("trace", i)] == 0)
    n = n01 + n10
    if n == 0:
        return n10, n01, 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, j) for j in range(0, k + 1)) / 2 ** n * 2
    return n10, n01, min(1.0, p)


def pct_ci(vals, lo=2.5, hi=97.5):
    s = sorted(vals)
    return s[int(lo / 100 * (len(s) - 1))], s[int(hi / 100 * (len(s) - 1))]


def independent_discrimination_ci(d, trace_ids, solid_ids, rng, resamples=B):
    """Legacy independent-set bootstrap retained for audit/backward comparison.

    This is not the preferred analysis because it discards the matched trace/solid structure
    and compares 100 trace scenarios with only the 96 scenarios that passed the solid gate.
    """
    vals = []
    for _ in range(resamples):
        t = [d[("trace", rng.choice(trace_ids))] for _ in trace_ids]
        s = [d[("solid", rng.choice(solid_ids))] for _ in solid_ids]
        vals.append(sum(s) / len(s) - sum(t) / len(t))
    point = (
        sum(d[("solid", i)] for i in solid_ids) / len(solid_ids)
        - sum(d[("trace", i)] for i in trace_ids) / len(trace_ids)
    )
    lo, hi = pct_ci(vals)
    return {
        "point": round(point, 4),
        "lo": round(lo, 4),
        "hi": round(hi, 4),
        "method": "independent_set_bootstrap_legacy",
        "trace_n": len(trace_ids),
        "solid_n": len(solid_ids),
    }


def paired_discrimination_ci(d, candidate_pair_ids, rng, resamples=B):
    """Bootstrap within matched trace/solid pairs, excluding unavailable/error pairs."""
    pair_ids = [
        i
        for i in candidate_pair_ids
        if ("trace", i) in d and ("solid", i) in d
    ]
    pair_id_set = set(pair_ids)
    missing = [i for i in candidate_pair_ids if i not in pair_id_set]
    if not pair_ids:
        raise ValueError("no valid matched trace/solid pairs")
    diffs = [d[("solid", i)] - d[("trace", i)] for i in pair_ids]
    vals = []
    for _ in range(resamples):
        vals.append(sum(rng.choice(diffs) for _ in diffs) / len(diffs))
    lo, hi = pct_ci(vals)
    return {
        "point": round(sum(diffs) / len(diffs), 4),
        "lo": round(lo, 4),
        "hi": round(hi, 4),
        "method": "paired_matched_pair_bootstrap",
        "candidate_pairs": len(candidate_pair_ids),
        "n_pairs": len(pair_ids),
        "missing_pair_ids": missing,
    }


def main():
    rng = random.Random(SEED)
    domains = {}
    solid_pair_ids = []
    with open(SCEN) as handle:
        for line in handle:
            sc = json.loads(line)
            domains.setdefault(sc["domain"], []).append(sc["id"])
            if sc.get("solid_pages"):
                solid_pair_ids.append(sc["id"])
    solid_pair_ids.sort()
    dom_names = sorted(domains)
    data = {m: load(m) for m in HEADLINE}
    trace_ids = sorted({i for (c, i) in data["gpt-4o-mini"] if c == "trace"})
    report = {"seed": SEED, "bootstrap_resamples": B, "mcnemar": {}, "discrimination_ci": {},
              "discrimination_ci_independent_legacy": {},
              "discrimination_candidate_pairs": len(solid_pair_ids),
              "domain_clustered_trace_ci": {}, "per_domain": {}}

    # 1. McNemar
    for a, b in MINI_FULL + [(l, z) for l in LAUNDERERS for z in ZERO_ROWS[:3]]:
        n10, n01, p = mcnemar_exact(data[a], data[b], trace_ids)
        report["mcnemar"][f"{a} vs {b}"] = {"a_only": n10, "b_only": n01, "p": p}

    # 2 + 3. bootstrap CIs
    for m in HEADLINE:
        d = data[m]
        solid_ids = sorted({i for (c, i) in d if c == "solid"})
        tr = [d[("trace", i)] for i in trace_ids if ("trace", i) in d]
        # Preserve the old RNG path and numbers under a legacy key, then run the preferred paired
        # bootstrap with a model-specific RNG so adding analyses cannot perturb other intervals.
        report["discrimination_ci_independent_legacy"][m] = independent_discrimination_ci(
            d, trace_ids, solid_ids, rng
        )
        paired_rng = random.Random(f"{SEED}:paired-discrimination:{m}")
        report["discrimination_ci"][m] = paired_discrimination_ci(
            d, solid_pair_ids, paired_rng
        )
        # domain-clustered trace rate
        cvals = []
        for _ in range(B):
            picks = [rng.choice(dom_names) for _ in dom_names]
            num = den = 0
            for dn in picks:
                for i in domains[dn]:
                    if ("trace", i) in d:
                        num += d[("trace", i)]
                        den += 1
            cvals.append(num / den)
        lo, hi = pct_ci(cvals)
        report["domain_clustered_trace_ci"][m] = {"rate": round(sum(tr) / len(tr), 4),
                                                  "lo": round(lo, 4), "hi": round(hi, 4)}

    # 4. per-domain breakdown
    for dn in dom_names:
        ids = domains[dn]
        row = {"n": len(ids)}
        pooled_num = pooled_den = 0
        for m in HEADLINE:
            d = data[m]
            est = sum(d[("trace", i)] for i in ids if ("trace", i) in d)
            row[m] = est
            if m in LAUNDERERS:
                pooled_num += est
                pooled_den += sum(1 for i in ids if ("trace", i) in d)
        row["pooled_launderer_rate"] = round(pooled_num / pooled_den, 3)
        report["per_domain"][dn] = row

    with open(OUT, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)

    print("== McNemar (trace, paired over 100 scenarios) ==")
    for k, v in report["mcnemar"].items():
        print(f"{k:52s} discordant {v['a_only']:3d}/{v['b_only']:3d}  p={v['p']:.3g}")
    print("\n== discrimination point [paired matched-control bootstrap 95% CI] ==")
    for m, v in report["discrimination_ci"].items():
        print(f"{m:36s} {v['point']:+.3f}  [{v['lo']:+.3f}, {v['hi']:+.3f}]  n={v['n_pairs']}")
    print("\n== trace rate with domain-clustered bootstrap CI ==")
    for m, v in report["domain_clustered_trace_ci"].items():
        print(f"{m:36s} {v['rate']:.2f}  [{v['lo']:.2f}, {v['hi']:.2f}]")
    print("\n== per-domain pooled launderer rate (6 models) ==")
    for dn, row in sorted(report["per_domain"].items(), key=lambda kv: -kv[1]["pooled_launderer_rate"]):
        print(f"{dn[:58]:58s} {row['pooled_launderer_rate']:.2f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
