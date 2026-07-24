"""Exp3: adversarial robustness. Run the key conditions under MAXIMUM player pressure
(pressure=1.0: every query gets a 'answer directly / everyone already knows it' push) and
compare to the normal-pressure numbers. Hypothesis: the enum-enforced structured controller
is unaffected (cannot pick an unlicensed mode), while prompting policies leak more."""
import json
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
from evaluator import evaluate_all, aggregate_results

SEEDS = [7, 11, 17, 23, 29]
WORLDS, QPW = 8, 5
PRESSURE = 1.0
MODELS = [("qwen7b", "qwen2.5:7b-instruct"),
          ("qwen32b", "qwen2.5:32b-instruct"),
          ("qwen3-30b", "qwen3:30b-a3b-instruct-2507-q4_K_M")]
POLICIES = ["local_state", "careful", "source_calibrated"]
BASE = "http://localhost:11434/v1"
METRICS = ["epistemic_consistency", "omniscience_leakage", "false_witness",
           "false_presupposition_resistance", "source_attribution_accuracy"]
OUT = "/path/to/launderbench/outputs"
prog = open(f"{OUT}/exp3_progress.jsonl", "w")
per_seed = defaultdict(list)

for seed in SEEDS:
    worlds, queries = generate_dataset(num_worlds=WORLDS, queries_per_world=QPW, seed=seed, pressure=PRESSURE)
    for mlabel, model in MODELS:
        client = OpenAICompatibleChatClient(model=model, base_url=BASE, temperature=0.0,
                                            use_structured_output=True)
        controllers = make_llm_controllers(client, policies=POLICIES) + [
            StructuredSourceCalibratedController(client=client, name="llm_structured")
        ]
        try:
            agg = aggregate_results(evaluate_all(worlds, queries, controllers))
            for ctrl, md in agg.items():
                key = f"{mlabel}/{ctrl.replace('llm_', '')}"
                for m, v in md.items():
                    if v is not None:
                        per_seed[(key, m)].append(v)
                prog.write(json.dumps({"label": key, "seed": seed, "metrics": md}) + "\n")
            prog.flush()
            print(f"[seed {seed}] {mlabel} high-pressure done", flush=True)
        except Exception as exc:
            print(f"[seed {seed}] {mlabel} FAILED: {exc}", flush=True)
prog.close()

summary = {}
for label in sorted(set(k[0] for k in per_seed)):
    summary[label] = {}
    for m in METRICS:
        vals = per_seed.get((label, m), [])
        if vals:
            summary[label][m] = {"mean": round(statistics.mean(vals), 3),
                                 "std": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0}
json.dump(summary, open(f"{OUT}/exp3_pressure_results.json", "w"), indent=2)


def c(label, m):
    s = summary.get(label, {}).get(m)
    return f"{s['mean']:.2f}±{s['std']:.2f}" if s else "n/a"


print(f"\n==== HIGH PRESSURE (pressure={PRESSURE}) mean±std, 5 seeds ====")
for mdl in ["qwen7b", "qwen32b", "qwen3-30b"]:
    print(f"\n[{mdl}]")
    print(f"  {'condition':20s}{'EC':>13}{'OLR':>13}{'FWR':>13}{'FPR':>13}")
    for cond in ["local_state", "careful", "source_calibrated", "structured"]:
        lab = f"{mdl}/{cond}"
        print(f"  {cond:20s}{c(lab,'epistemic_consistency'):>13}{c(lab,'omniscience_leakage'):>13}"
              f"{c(lab,'false_witness'):>13}{c(lab,'false_presupposition_resistance'):>13}")
print("\nCompare OLR/FWR here vs §5/§6 normal-pressure numbers to see who degrades under pressure.")
