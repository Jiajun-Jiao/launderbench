"""Clean ablation: fair ladder (all share per-NPC local evidence) across 3 models x 5 seeds.
local_state (naive) < careful (strong NL careful) < source_calibrated_nohint (taxonomy, no gold hint)
< source_calibrated (taxonomy + gold acceptable_modes, upper reference)."""
import json
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, "/path/to/launderbench")
from query_generator import generate_dataset
from baselines import default_controllers
from llm_controller import OpenAICompatibleChatClient, make_llm_controllers
from evaluator import evaluate_all, aggregate_results

SEEDS = [7, 11, 17, 23, 29]
WORLDS, QPW = 8, 5
POLICIES = ["local_state", "careful", "source_calibrated_nohint", "source_calibrated"]
MODELS = [
    ("qwen7b", "qwen2.5:7b-instruct"),
    ("qwen32b", "qwen2.5:32b-instruct"),
    ("qwen3-30b", "qwen3:30b-a3b-instruct-2507-q4_K_M"),
]
BASE = "http://localhost:11434/v1"
METRICS = ["warranted_answer_accuracy", "epistemic_consistency", "omniscience_leakage",
           "false_witness", "source_attribution_accuracy", "rumor_as_fact",
           "false_presupposition_resistance", "information_seeking_utility"]
OUT = "/path/to/launderbench/outputs"
prog = open(f"{OUT}/multiseed_ablation_progress.jsonl", "w")
per_seed = defaultdict(list)


def record(label_tmpl, seed, agg):
    for ctrl, md in agg.items():
        key = label_tmpl.format(ctrl=ctrl)
        for m, v in md.items():
            if v is not None:
                per_seed[(key, m)].append(v)
        prog.write(json.dumps({"label": key, "seed": seed, "metrics": md}) + "\n")
    prog.flush()


for seed in SEEDS:
    worlds, queries = generate_dataset(num_worlds=WORLDS, queries_per_world=QPW, seed=seed)
    try:
        record("{ctrl}", seed, aggregate_results(evaluate_all(worlds, queries, default_controllers())))
    except Exception as exc:
        print(f"[seed {seed}] baselines FAILED: {exc}", flush=True)
    for label, model in MODELS:
        try:
            client = OpenAICompatibleChatClient(model=model, base_url=BASE,
                                                temperature=0.0, use_structured_output=True)
            agg = aggregate_results(evaluate_all(worlds, queries, make_llm_controllers(client, policies=POLICIES)))
            record(label + "/{ctrl}", seed, {c.replace("llm_", ""): md for c, md in agg.items()})
            print(f"[seed {seed}] {label} done", flush=True)
        except Exception as exc:
            print(f"[seed {seed}] {label} FAILED: {exc}", flush=True)

prog.close()

summary = {}
for label in sorted(set(k[0] for k in per_seed)):
    summary[label] = {}
    for m in METRICS:
        vals = per_seed.get((label, m), [])
        if vals:
            summary[label][m] = {"mean": round(statistics.mean(vals), 3),
                                 "std": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0,
                                 "n": len(vals)}
json.dump(summary, open(f"{OUT}/multiseed_ablation_results.json", "w"), indent=2)

def cell(label, m):
    s = summary.get(label, {}).get(m)
    return f"{s['mean']:.2f}±{s['std']:.2f}" if s else "n/a"

print("\n==== fair ladder: mean +/- std across seeds (all share per-NPC local evidence) ====")
print(f"{'model / policy':30s}{'EC':>12}{'OLR':>12}{'FWR':>12}{'RAF':>12}{'SAA':>12}{'ISU':>12}")
for _, _label in [(0, f"{m}/{p}") for p in POLICIES for m in ["qwen7b", "qwen32b", "qwen3-30b"]]:
    print(f"{_label:30s}" + "".join(cell(_label, m) .rjust(12) for m in
          ["epistemic_consistency", "omniscience_leakage", "false_witness",
           "rumor_as_fact", "source_attribution_accuracy", "information_seeking_utility"]))
print(f"\n(oracle source_calibrated EC={cell('source_calibrated','epistemic_consistency')})")
print(f"seeds={SEEDS} worlds={WORLDS} qpw={QPW} -> {WORLDS*QPW} q/seed; policies={POLICIES}")
