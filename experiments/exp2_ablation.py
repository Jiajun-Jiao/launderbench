"""Exp2: component ablation of the structured controller (2x2: enforce x deterministic-attr)
across 3 models x 5 seeds. Shows how much enum-enforcement and deterministic attribution
each contribute. Also reports mode_allowed_rate (fraction of responses using a licensed mode)."""
import json
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, "/path/to/launderbench")
from query_generator import generate_dataset
from llm_controller import OpenAICompatibleChatClient, StructuredSourceCalibratedController
from evaluator import evaluate_all, aggregate_results

SEEDS = [7, 11, 17, 23, 29]
WORLDS, QPW = 8, 5
MODELS = [("qwen7b", "qwen2.5:7b-instruct"),
          ("qwen32b", "qwen2.5:32b-instruct"),
          ("qwen3-30b", "qwen3:30b-a3b-instruct-2507-q4_K_M")]
CONFIGS = [
    ("enforce+det", dict(enforce_modes=True, deterministic_attribution=True)),
    ("hint+det", dict(enforce_modes=False, deterministic_attribution=True)),
    ("enforce+llmattr", dict(enforce_modes=True, deterministic_attribution=False)),
    ("hint+llmattr", dict(enforce_modes=False, deterministic_attribution=False)),
]
BASE = "http://localhost:11434/v1"
METRICS = ["epistemic_consistency", "omniscience_leakage", "source_attribution_accuracy",
           "warranted_answer_accuracy", "false_witness", "mode_allowed_rate"]
OUT = "/path/to/launderbench/outputs"
prog = open(f"{OUT}/exp2_progress.jsonl", "w")
per_seed = defaultdict(list)

for seed in SEEDS:
    worlds, queries = generate_dataset(num_worlds=WORLDS, queries_per_world=QPW, seed=seed)
    for mlabel, model in MODELS:
        client = OpenAICompatibleChatClient(model=model, base_url=BASE, temperature=0.0,
                                            use_structured_output=True)
        for clabel, kw in CONFIGS:
            try:
                ctrl = StructuredSourceCalibratedController(client=client, name="s", **kw)
                recs = evaluate_all(worlds, queries, [ctrl])
                agg = aggregate_results(recs)["s"]
                key = f"{mlabel}/{clabel}"
                for m, v in agg.items():
                    if v is not None:
                        per_seed[(key, m)].append(v)
                mar = sum(1 for r in recs if r.details["mode_allowed"]) / len(recs)
                per_seed[(key, "mode_allowed_rate")].append(mar)
                prog.write(json.dumps({"label": key, "seed": seed,
                                       "ec": agg["epistemic_consistency"], "mar": mar}) + "\n")
                prog.flush()
            except Exception as exc:
                print(f"[seed {seed}] {mlabel}/{clabel} FAILED: {exc}", flush=True)
        print(f"[seed {seed}] {mlabel} ablation done", flush=True)
prog.close()

summary = {}
for label in sorted(set(k[0] for k in per_seed)):
    summary[label] = {}
    for m in METRICS:
        vals = per_seed.get((label, m), [])
        if vals:
            summary[label][m] = {"mean": round(statistics.mean(vals), 3),
                                 "std": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0}
json.dump(summary, open(f"{OUT}/exp2_ablation_results.json", "w"), indent=2)


def c(label, m):
    s = summary.get(label, {}).get(m)
    return f"{s['mean']:.2f}±{s['std']:.2f}" if s else "n/a"


print("\n==== component ablation (mean±std, 5 seeds) ====")
for mdl in ["qwen7b", "qwen32b", "qwen3-30b"]:
    print(f"\n[{mdl}]    enforce = enum-restrict modes ; det = deterministic attribution")
    print(f"  {'config':20s}{'EC':>13}{'OLR':>13}{'SAA':>13}{'mode_ok':>13}")
    for clabel, _ in CONFIGS:
        lab = f"{mdl}/{clabel}"
        print(f"  {clabel:20s}{c(lab,'epistemic_consistency'):>13}{c(lab,'omniscience_leakage'):>13}"
              f"{c(lab,'source_attribution_accuracy'):>13}{c(lab,'mode_allowed_rate'):>13}")
print(f"\nseeds={SEEDS}; {WORLDS*QPW} q/seed; enforce+det is the full proposed method")
