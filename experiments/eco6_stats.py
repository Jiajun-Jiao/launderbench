"""Statistical rigor (reviewer #7): the paper reports mean±std with no CIs or significance tests,
and re-narrates cases where structured ties/loses the best prompt. This computes, on aligned
per-query data, each controller's EC with a 95% bootstrap CI, and the paired structured-vs-best-prompt
difference with a bootstrap CI + an exact-binomial McNemar test — so we can state honestly whether the
gap exceeds noise. Local qwen, free."""
import json
import math
import os
import random
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, "/path/to/launderbench")
from query_generator import generate_dataset
from llm_controller import (
    OpenAICompatibleChatClient,
    StructuredSourceCalibratedController,
    make_llm_controllers,
)
from evaluator import evaluate_response

SEEDS = [int(x) for x in os.environ.get("SEEDS", "7,11,17").split(",")]
WORLDS = int(os.environ.get("WORLDS", "8"))
QPW = int(os.environ.get("QPW", "5"))
GEN_MODEL = os.environ.get("GEN_MODEL", "qwen2.5:32b-instruct")
B = 4000


def boot_ci(xs):
    rng = random.Random(0)
    n = len(xs)
    means = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return means[int(0.025 * B)], means[int(0.975 * B)]


def mcnemar(pairs):  # pairs of (a_correct, b_correct); exact two-sided binomial
    b = sum(1 for a, bb in pairs if a and not bb)
    c = sum(1 for a, bb in pairs if bb and not a)
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return b, c, min(1.0, p)


def main():
    gen = OpenAICompatibleChatClient(model=GEN_MODEL, base_url="http://localhost:11434/v1",
                                     temperature=0.0, use_structured_output=True)
    ladder = make_llm_controllers(gen, ["local_state", "careful", "source_calibrated"])
    ctrls = {c.name: c for c in ladder}
    ctrls["llm_structured"] = StructuredSourceCalibratedController(client=gen, name="llm_structured")
    ec = defaultdict(list)
    olr = defaultdict(list)
    for seed in SEEDS:
        worlds, queries = generate_dataset(WORLDS, QPW, seed)
        by = {w.world_id: w for w in worlds}
        for q in queries:
            w = by[q.world_id]
            for name, c in ctrls.items():
                m = evaluate_response(w, q, c.respond(w, q)).metrics
                ec[name].append(m["epistemic_consistency"])
                olr[name].append(m["omniscience_leakage"])
        print(f"seed {seed} done", flush=True)

    n = len(ec["llm_structured"])
    print(f"\n=== per-controller EC with 95% bootstrap CI (n={n}, {GEN_MODEL}) ===")
    summ = {}
    for name in ctrls:
        lo, hi = boot_ci(ec[name])
        summ[name] = {"EC": round(statistics.mean(ec[name]), 3), "EC_CI": [round(lo, 3), round(hi, 3)],
                      "OLR": round(statistics.mean(olr[name]), 3)}
        print(f"  {name:24s} EC={statistics.mean(ec[name]):.3f}  95%CI[{lo:.3f},{hi:.3f}]  OLR={statistics.mean(olr[name]):.3f}")

    best = max(ladder, key=lambda c: statistics.mean(ec[c.name])).name
    diffs = [a - b for a, b in zip(ec["llm_structured"], ec[best])]
    lo, hi = boot_ci(diffs)
    b_, c_, p = mcnemar([(a == 1.0, b == 1.0) for a, b in zip(ec["llm_structured"], ec[best])])
    sig = "CI excludes 0 -> SIGNIFICANT" if (lo > 0 or hi < 0) else "CI includes 0 -> NOT significant (report honestly)"
    print(f"\n=== structured vs best prompt ({best}) ===")
    print(f"  ΔEC = {statistics.mean(diffs):+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  | McNemar b={b_} c={c_} p={p:.4f}  | {sig}")
    summ["_comparison"] = {"best_prompt": best, "delta_EC": round(statistics.mean(diffs), 3),
                           "delta_CI": [round(lo, 3), round(hi, 3)], "mcnemar_b": b_, "mcnemar_c": c_,
                           "mcnemar_p": round(p, 4), "significant": bool(lo > 0 or hi < 0)}
    json.dump(summ, open("outputs/stats_results.json", "w"), indent=2)
    print("\nwrote outputs/stats_results.json")


if __name__ == "__main__":
    main()
