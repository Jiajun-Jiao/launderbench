"""#1 Human-validation kit — STEP 1: sample annotation items.
Draws a stratified sample of (scenario, NPC-reply) items from the scaled benchmark
(outputs/eco_worlds_full.jsonl), across diverse controllers (so every property has positives
& negatives), and emits:
  outputs/human_eval/items.csv  -> annotator-facing (blank Y/N columns; NO auto labels shown)
  outputs/human_eval/key.jsonl  -> HIDDEN key: item_id + auto-metric verdicts + (optional) LLM-judge
Optionally runs the same gpt-4.1-mini judge as §11 on each item (RUN_JUDGE=1, ~ $0.05) so the
scoring step can compare humans vs LLM-judge vs auto. Key from .env, never persisted."""
import csv
import json
import os
import random
import re
import sys

sys.path.insert(0, "/path/to/launderbench")
from eco_worlds import build_queries, validate_and_build
from eco3_metric_validity import _JUDGE_DEV, JUDGE_SCHEMA, render_knowledge
from baselines import default_controllers
from llm_controller import (
    OpenAICompatibleChatClient,
    StructuredSourceCalibratedController,
    make_llm_controllers,
)
from evaluator import evaluate_response

ECO_PATH = os.environ.get("ECO_PATH", "outputs/eco_worlds_full.jsonl")
NWORLDS = int(os.environ.get("NWORLDS", "40"))
N_ITEMS = int(os.environ.get("N_ITEMS", "120"))
GEN_MODEL = os.environ.get("GEN_MODEL", "qwen2.5:32b-instruct")
RUN_JUDGE = os.environ.get("RUN_JUDGE", "1") == "1"
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4.1-mini")
OUTDIR = "/path/to/launderbench/outputs/human_eval"


def load(path):
    with open(path) as fh:
        txt = fh.read().strip()
    return json.loads(txt) if txt.startswith("[") else [json.loads(l) for l in txt.splitlines() if l.strip()]


def clean_q(q):
    """Turn the templated 'About this: <prop> What do you actually know…' into a natural question;
    rewrite terse code-type labels (e.g. 'vault_code') into 'What is the vault code?'."""
    m = re.search(r"About this:\s*(.*?)\s*What do you actually know", q)
    prop = (m.group(1).strip() if m else q.strip())
    if not prop:
        return q
    if prop.endswith("?"):
        return prop
    if "_" in prop or len(prop.split()) <= 2:
        return "What is the " + prop.replace("_", " ").strip() + "?"
    return prop + "?"


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    raws = load(ECO_PATH)[:NWORLDS]
    worlds, queries = [], []
    for i, raw in enumerate(raws):
        w, _ = validate_and_build(raw, f"eco_{i:04d}")
        if w is None:
            continue
        worlds.append(w)
        queries.extend(build_queries(w))
    by = {w.world_id: w for w in worlds}

    gen = OpenAICompatibleChatClient(model=GEN_MODEL, base_url="http://localhost:11434/v1",
                                     temperature=0.0, use_structured_output=True)
    det = {c.name: c for c in default_controllers()}
    controllers = [det["global_state"], det["rag_npc"], det["source_calibrated"], det["prompt_only"]] + \
        make_llm_controllers(gen, ["local_state", "source_calibrated"]) + \
        [StructuredSourceCalibratedController(client=gen, name="llm_structured")]

    recs = []
    for q in queries:
        w = by[q.world_id]
        fact = w.facts[q.proposition_id]
        for ctrl in controllers:
            resp = ctrl.respond(w, q)
            m = evaluate_response(w, q, resp).metrics
            recs.append({
                "ctrl": ctrl.name, "question": clean_q(q.question), "true": fact.true_value,
                "knows": render_knowledge(w, q), "reply": resp.text,
                "ec_auto": m["epistemic_consistency"] == 1.0, "olr_auto": m["omniscience_leakage"] == 1.0,
                "raf_auto": m["rumor_as_fact"] == 1.0, "fwr_auto": m["false_witness"] == 1.0,
            })
    print(f"built {len(recs)} candidate items from {len(worlds)} worlds", flush=True)

    # stratified sample: guarantee positives for leak / rumor / inconsistent, plus consistent negatives
    rng = random.Random(0)
    seen, chosen = set(), []
    per = max(1, N_ITEMS // 5)
    buckets = [[r for r in recs if r["olr_auto"]],
               [r for r in recs if r["raf_auto"] or r["fwr_auto"]],
               [r for r in recs if not r["ec_auto"]],
               [r for r in recs if r["ec_auto"]],
               [r for r in recs if r["ctrl"] == "llm_structured"]]  # ensure the method's own replies are judged
    for pool in buckets:
        rng.shuffle(pool)
        for r in pool[:per]:
            if id(r) not in seen:
                seen.add(id(r)); chosen.append(r)
    rng.shuffle(recs)
    for r in recs:
        if len(chosen) >= N_ITEMS:
            break
        if id(r) not in seen:
            seen.add(id(r)); chosen.append(r)
    rng.shuffle(chosen)  # mix controllers so annotators don't see runs of the same behavior
    chosen = chosen[:N_ITEMS]

    judge = None
    if RUN_JUDGE:
        judge = OpenAICompatibleChatClient(model=JUDGE_MODEL, base_url="https://api.openai.com/v1",
                                           api_key=os.environ["OPENAI_API_KEY"], temperature=0.0,
                                           use_structured_output=True)

    items_path = f"{OUTDIR}/items.csv"
    key_path = f"{OUTDIR}/key.jsonl"
    kh = open(key_path, "w")
    with open(items_path, "w", newline="") as ch:
        cw = csv.writer(ch)
        cw.writerow(["item_id", "question", "what_the_npc_knows", "the_true_answer", "npc_reply",
                     "consistent_YN", "leaks_truth_YN", "rumor_as_fact_YN", "scenario_ok_YN", "notes"])
        for i, r in enumerate(chosen):
            iid = f"H{i:03d}"
            cw.writerow([iid, r["question"], r["knows"], r["true"], r["reply"], "", "", "", "", ""])
            jd = None
            if judge is not None:
                ctx = {"question": r["question"], "the_true_answer": r["true"],
                       "what_this_npc_actually_knows": r["knows"], "npc_reply": r["reply"]}
                try:
                    jd = judge.complete_json([{"role": "developer", "content": _JUDGE_DEV},
                                              {"role": "user", "content": json.dumps(ctx, ensure_ascii=True)}],
                                             response_schema=JUDGE_SCHEMA)
                except Exception as exc:  # noqa: BLE001
                    jd = {"error": str(exc)[:60]}
            kh.write(json.dumps({"item_id": iid, "ctrl": r["ctrl"],
                                 "auto": {"ec": r["ec_auto"], "olr": r["olr_auto"],
                                          "raf": r["raf_auto"], "fwr": r["fwr_auto"]},
                                 "llm_judge": jd}) + "\n")
            if (i + 1) % 25 == 0:
                print(f"  wrote {i+1}/{len(chosen)} items", flush=True)
    kh.close()

    # blank per-annotator copies (3) for convenience
    import shutil
    for a in (1, 2, 3):
        shutil.copyfile(items_path, f"{OUTDIR}/annot_{a}.csv")

    if judge is not None:
        cost = judge.input_tokens / 1e6 * 0.40 + judge.output_tokens / 1e6 * 1.60
        print(f"LLM-judge: calls={judge.calls} cost=${cost:.3f}")
    print(f"\nwrote {len(chosen)} items -> {items_path}")
    print(f"hidden key -> {key_path}")
    print(f"blank annotator copies -> {OUTDIR}/annot_1.csv, annot_2.csv, annot_3.csv")
    from collections import Counter
    print("controller mix:", dict(Counter(r["ctrl"] for r in chosen)))
    print("auto positives: leaks=%d rumor/fw=%d inconsistent=%d" % (
        sum(r["olr_auto"] for r in chosen),
        sum(r["raf_auto"] or r["fwr_auto"] for r in chosen),
        sum(not r["ec_auto"] for r in chosen)))


if __name__ == "__main__":
    main()
