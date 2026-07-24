"""GPT-4.1-mini as a 6th controller family for the cross-family table (§8.1).
Ladder + structured, 3 seeds, default 4 scenarios, gold-free. Key read from env (.env),
never persisted. Reports actual token cost. Resumable via outputs/gpt41_progress.jsonl."""
import json
import os
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
from evaluator import aggregate_results, evaluate_all

SEEDS = [int(x) for x in os.environ.get("SEEDS", "7,11,17").split(",")]
WORLDS = int(os.environ.get("WORLDS", "8"))
QPW = int(os.environ.get("QPW", "5"))
POLICIES = ["local_state", "careful", "source_calibrated"]
MODELS = [("gpt41mini", "gpt-4.1-mini")]
PRICE = {"gpt-4.1-mini": (0.40, 1.60)}  # $/1M in, out
BASE = os.environ.get("BASE", "https://api.openai.com/v1")
OUT = "/path/to/launderbench/outputs"
key = os.environ["OPENAI_API_KEY"]

rows = defaultdict(list)
done = set()
prog_path = f"{OUT}/gpt41_progress.jsonl"
if os.path.exists(prog_path):
    for line in open(prog_path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        done.add((r["label"], r["seed"]))
        for m, v in r["metrics"].items():
            if v is not None:
                rows[(r["label"], m)].append(v)
    print(f"resumed: {len(done)} (label,seed) cells already on disk", flush=True)

prog = open(prog_path, "a")
cost_total = 0.0
for mlabel, model in MODELS:
    client = OpenAICompatibleChatClient(model=model, base_url=BASE, api_key=key,
                                        temperature=0.0, use_structured_output=True)
    for seed in SEEDS:
        if (f"{mlabel}/structured", seed) in done:
            print(f"[{mlabel}] seed {seed} cached", flush=True)
            continue
        worlds, queries = generate_dataset(WORLDS, QPW, seed)
        ctrls = make_llm_controllers(client, policies=POLICIES) + [
            StructuredSourceCalibratedController(client=client, name="llm_structured")]
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
    print(f"[{mlabel}] calls={client.calls} in={client.input_tokens} out={client.output_tokens} "
          f"cost(this run)=${c:.3f}", flush=True)
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
json.dump(summary, open(f"{OUT}/gpt41_results.json", "w"), indent=2)


def cell(lab, m):
    s = summary.get(lab, {}).get(m)
    return f"{s['mean']:.2f}±{s['std']:.2f}" if s else "n/a"


print("\n#### GPT-4.1-mini: ladder + structured (3 seeds, mean±std) ####")
for mlabel, _ in MODELS:
    print(f"[{mlabel}]")
    for cond in ["local_state", "careful", "source_calibrated", "structured"]:
        lab = f"{mlabel}/{cond}"
        print(f"  {cond:18s} EC={cell(lab,'epistemic_consistency'):>11}  "
              f"OLR={cell(lab,'omniscience_leakage'):>11}  SAA={cell(lab,'source_attribution_accuracy'):>11}")
print(f"\nTOTAL COST (this run) = ${cost_total:.3f}")
