"""Multi-seed, multi-model non-oracle run -> mean/std per (controller, metric).
Writes incremental progress (resumable view) and a final results JSON."""
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
POLICIES = ["global_state", "local_state", "source_calibrated"]
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
prog = open(f"{OUT}/multiseed_progress.jsonl", "w")
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
    # oracle/deterministic baselines (model-independent) once per seed
    try:
        agg = aggregate_results(evaluate_all(worlds, queries, default_controllers()))
        record("{ctrl}", seed, agg)  # label = controller name e.g. source_calibrated
    except Exception as exc:
        print(f"[seed {seed}] baselines FAILED: {exc}", flush=True)
    for label, model in MODELS:
        try:
            client = OpenAICompatibleChatClient(model=model, base_url=BASE,
                                                temperature=0.0, use_structured_output=True)
            ctrls = make_llm_controllers(client, policies=POLICIES)
            agg = aggregate_results(evaluate_all(worlds, queries, ctrls))
            # ctrl like 'llm_source_calibrated' -> 'qwen7b/source_calibrated'
            record(label + "/{ctrl}", seed, {c.replace("llm_", ""): md for c, md in agg.items()})
            ec = agg.get("llm_source_calibrated", {}).get("epistemic_consistency")
            print(f"[seed {seed}] {label} done (source_calibrated EC={ec})", flush=True)
        except Exception as exc:
            print(f"[seed {seed}] {label} FAILED: {exc}", flush=True)

prog.close()

summary = {}
for label in sorted(set(k[0] for k in per_seed)):
    summary[label] = {}
    for m in METRICS:
        vals = per_seed.get((label, m), [])
        if vals:
            summary[label][m] = {
                "mean": round(statistics.mean(vals), 3),
                "std": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }
json.dump(summary, open(f"{OUT}/multiseed_results.json", "w"), indent=2)

print("\n==== mean +/- std across seeds ====")
print(f"{'controller':28s} {'EC':>13} {'OLR':>13} {'SAA':>13} {'ISU':>13}")
for label in sorted(summary):
    s = summary[label]
    def cell(m):
        if m not in s:
            return "n/a"
        return f"{s[m]['mean']:.2f}±{s[m]['std']:.2f}"
    print(f"{label:28s} {cell('epistemic_consistency'):>13} {cell('omniscience_leakage'):>13} "
          f"{cell('source_attribution_accuracy'):>13} {cell('information_seeking_utility'):>13}")
print(f"\nseeds={SEEDS} worlds={WORLDS} qpw={QPW} -> {WORLDS*QPW} queries/seed")
