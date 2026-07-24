"""Combined model-major runner for exp1 (cross-family), exp3 (high pressure), exp4 (new
scenarios). Model-major loop => each model loads ~once (minimal ollama reload thrash).
3 seeds for these additional experiments (the 5-seed headline Qwen results are already done)."""
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

SEEDS = [7, 11, 17]
WORLDS, QPW = 8, 5
BASE = "http://localhost:11434/v1"
QWEN = [("qwen7b", "qwen2.5:7b-instruct"),
        ("qwen32b", "qwen2.5:32b-instruct"),
        ("qwen3-30b", "qwen3:30b-a3b-instruct-2507-q4_K_M")]
NEW = [("llama3.1", "llama3.1:8b"), ("mistral", "mistral:7b"), ("gemma2", "gemma2:9b")]
ALL = QWEN + NEW
NEW_SCEN = ["conflicting_evidence", "multi_hop_rumor"]
OUT = "/path/to/launderbench/outputs"
import os
rows = defaultdict(list)
done_models = set()
_progress = f"{OUT}/exp134_progress.jsonl"
if os.path.exists(_progress):
    for _line in open(_progress):
        try:
            _d = json.loads(_line)
            done_models.add(_d["label"].split("/")[0])
            for _m, _v in _d.get("seed_metrics", {}).items():
                if _v is not None:
                    rows[(_d["exp"], _d["label"], _m)].append(_v)
        except Exception:
            pass
prog = open(_progress, "a")
if done_models:
    print(f"resuming; skipping done models: {sorted(done_models)}", flush=True)


def build(client, policies, structured):
    ctrls = make_llm_controllers(client, policies=policies)
    if structured:
        ctrls = ctrls + [StructuredSourceCalibratedController(client=client, name="llm_structured")]
    return ctrls


def run(exp, mlabel, ctrls, worlds, queries):
    agg = aggregate_results(evaluate_all(worlds, queries, ctrls))
    for ctrl, md in agg.items():
        lab = f"{mlabel}/{ctrl.replace('llm_', '')}"
        for m, v in md.items():
            if v is not None:
                rows[(exp, lab, m)].append(v)
        prog.write(json.dumps({"exp": exp, "label": lab, "seed_metrics": md}) + "\n")
    prog.flush()


for mlabel, model in ALL:
    if mlabel in done_models:
        print(f"[{mlabel}] skip (already in progress file)", flush=True)
        continue
    client = OpenAICompatibleChatClient(model=model, base_url=BASE, temperature=0.0,
                                        use_structured_output=True, max_retries=1)
    for seed in SEEDS:
        # exp4: new scenarios (all models)
        try:
            w, q = generate_dataset(WORLDS, QPW, seed, scenario_types=NEW_SCEN)
            run("exp4_newscen", mlabel, build(client, ["local_state", "careful"], True), w, q)
        except Exception as e:
            print(f"{mlabel} s{seed} exp4 FAIL: {str(e)[:120]}", flush=True)
        # exp1: cross-family default scenarios (new models only; Qwen default already have 5-seed)
        if (mlabel, model) in NEW:
            try:
                w, q = generate_dataset(WORLDS, QPW, seed)
                run("exp1_crossfam", mlabel,
                    build(client, ["local_state", "careful", "source_calibrated_nohint", "source_calibrated"], True), w, q)
            except Exception as e:
                print(f"{mlabel} s{seed} exp1 FAIL: {str(e)[:120]}", flush=True)
        # exp3: high-pressure default (Qwen) — also test enum-enforce vs hint-only under pressure
        if (mlabel, model) in QWEN:
            try:
                w, q = generate_dataset(WORLDS, QPW, seed, pressure=1.0)
                ctrls = make_llm_controllers(client, ["local_state", "careful"]) + [
                    StructuredSourceCalibratedController(client=client, name="llm_structured",
                                                        enforce_modes=True, deterministic_attribution=True),
                    StructuredSourceCalibratedController(client=client, name="llm_structured_hint",
                                                        enforce_modes=False, deterministic_attribution=True),
                ]
                run("exp3_pressure", mlabel, ctrls, w, q)
            except Exception as e:
                print(f"{mlabel} s{seed} exp3 FAIL: {str(e)[:120]}", flush=True)
    print(f"[{mlabel}] done", flush=True)
prog.close()

summary = defaultdict(dict)
for (exp, lab, m), vals in rows.items():
    summary[exp].setdefault(lab, {})[m] = {
        "mean": round(statistics.mean(vals), 3),
        "std": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0,
    }
json.dump(summary, open(f"{OUT}/exp134_results.json", "w"), indent=2)


def c(exp, lab, m):
    s = summary.get(exp, {}).get(lab, {}).get(m)
    return f"{s['mean']:.2f}±{s['std']:.2f}" if s else "n/a"


print("\n#### exp1 CROSS-FAMILY (default scenarios, 3 seeds) ####")
for mlabel, _ in NEW:
    print(f"[{mlabel}]")
    for cond in ["local_state", "careful", "source_calibrated_nohint", "source_calibrated", "structured"]:
        lab = f"{mlabel}/{cond}"
        print(f"  {cond:24s} EC={c('exp1_crossfam',lab,'epistemic_consistency'):>11}  "
              f"OLR={c('exp1_crossfam',lab,'omniscience_leakage'):>11}  SAA={c('exp1_crossfam',lab,'source_attribution_accuracy'):>11}")

print("\n#### exp4 NEW SCENARIOS (conflicting + multi-hop, 3 seeds) ####")
for mlabel, _ in ALL:
    print(f"[{mlabel}]")
    for cond in ["local_state", "careful", "structured"]:
        lab = f"{mlabel}/{cond}"
        print(f"  {cond:24s} EC={c('exp4_newscen',lab,'epistemic_consistency'):>11}  "
              f"OLR={c('exp4_newscen',lab,'omniscience_leakage'):>11}  RAF={c('exp4_newscen',lab,'rumor_as_fact'):>11}")

print("\n#### exp3 HIGH PRESSURE (pressure=1.0, Qwen, 3 seeds) ####")
for mlabel, _ in QWEN:
    print(f"[{mlabel}]")
    for cond in ["local_state", "careful", "structured", "structured_hint"]:
        lab = f"{mlabel}/{cond}"
        print(f"  {cond:24s} EC={c('exp3_pressure',lab,'epistemic_consistency'):>11}  "
              f"OLR={c('exp3_pressure',lab,'omniscience_leakage'):>11}  FWR={c('exp3_pressure',lab,'false_witness'):>11}")
