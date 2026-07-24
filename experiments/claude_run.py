"""Modest Claude-API run: ladder + structured on Haiku 4.5 and Sonnet 4.6.
Key read from env (never persisted). Reports actual token cost."""
import json
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, "/path/to/launderbench")
from query_generator import generate_dataset
from llm_controller import (
    AnthropicChatClient,
    StructuredSourceCalibratedController,
    make_llm_controllers,
)
from evaluator import evaluate_all, aggregate_results

SEEDS = [7, 11, 17]
WORLDS, QPW = 8, 5
POLICIES = ["local_state", "careful", "source_calibrated"]
MODELS = [("haiku", "claude-haiku-4-5"), ("sonnet", "claude-sonnet-4-6")]
PRICE = {"claude-haiku-4-5": (1.0, 5.0), "claude-sonnet-4-6": (3.0, 15.0)}
OUT = "/path/to/launderbench/outputs"
key = os.environ["ANTHROPIC_API_KEY"]

rows = defaultdict(list)
cost_total = 0.0
prog = open(f"{OUT}/claude_progress.jsonl", "w")
for mlabel, model in MODELS:
    client = AnthropicChatClient(model=model, api_key=key)
    for seed in SEEDS:
        worlds, queries = generate_dataset(WORLDS, QPW, seed)
        ctrls = make_llm_controllers(client, policies=POLICIES) + [
            StructuredSourceCalibratedController(client=client, name="llm_structured")
        ]
        agg = aggregate_results(evaluate_all(worlds, queries, ctrls))
        for ctrl, md in agg.items():
            lab = f"{mlabel}/{ctrl.replace('llm_', '')}"
            for m, v in md.items():
                if v is not None:
                    rows[(lab, m)].append(v)
            prog.write(json.dumps({"label": lab, "seed": seed, "metrics": md}) + "\n")
        prog.flush()
        print(f"[{mlabel}] seed {seed} done", flush=True)
    pin, pout = PRICE[model]
    c = client.input_tokens / 1e6 * pin + client.output_tokens / 1e6 * pout
    cost_total += c
    print(f"[{mlabel}] calls={client.calls} in={client.input_tokens} out={client.output_tokens} cost=${c:.3f}", flush=True)
prog.close()

summary = {}
for lab in sorted(set(k[0] for k in rows)):
    summary[lab] = {}
    for m in ["warranted_answer_accuracy", "epistemic_consistency", "omniscience_leakage",
              "false_witness", "source_attribution_accuracy", "rumor_as_fact"]:
        vals = rows.get((lab, m), [])
        if vals:
            summary[lab][m] = {"mean": round(statistics.mean(vals), 3),
                               "std": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0}
json.dump(summary, open(f"{OUT}/claude_results.json", "w"), indent=2)


def cell(lab, m):
    s = summary.get(lab, {}).get(m)
    return f"{s['mean']:.2f}±{s['std']:.2f}" if s else "n/a"


print("\n#### Claude API: ladder + structured (3 seeds, mean±std) ####")
for mlabel, _ in MODELS:
    print(f"[{mlabel}]")
    for cond in ["local_state", "careful", "source_calibrated", "structured"]:
        lab = f"{mlabel}/{cond}"
        print(f"  {cond:18s} EC={cell(lab,'epistemic_consistency'):>11}  OLR={cell(lab,'omniscience_leakage'):>11}  "
              f"SAA={cell(lab,'source_attribution_accuracy'):>11}")
print(f"\nTOTAL COST = ${cost_total:.3f}")
