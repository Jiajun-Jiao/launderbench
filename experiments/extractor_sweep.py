"""Multi-model extractor sweep: is the METHOD (narrative_struct = decouple graph-reconstruction +
propagation) robustly better than the joint baseline (narrative_graph) ACROSS extractor models?
Runs each model end-to-end on the SAME worlds and tabulates source_type / EC / OLR for both methods,
plus Δ(struct − graph) = the method's net gain. Local models free; gpt-* via OpenAI.

Env: MODELS (comma list), MAXW (8), NARR_PATH, OLLAMA_BASE.
"""
import dataclasses
import json
import os
import statistics
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eco_worlds import build_queries, validate_and_build
from epistemic_state import EpistemicState
from llm_controller import AnthropicChatClient, OpenAICompatibleChatClient, StructuredSourceCalibratedController
from evaluator import _answer_matches, evaluate_response
from graph_extract3 import narrative_graph_states
from graph_extract4 import narrative_struct_states

PATH = os.environ.get("NARR_PATH", os.path.join(_ROOT, "outputs", "eco_narrative.jsonl"))
MAXW = int(os.environ.get("MAXW", "8"))
OLLAMA = os.environ.get("OLLAMA_BASE", "http://localhost:11434/v1")
DEFAULT_MODELS = ("qwen2.5:7b-instruct,llama3.1:8b,mistral:7b,gemma2:9b,"
                  "qwen3:30b-a3b-instruct-2507-q4_K_M,qwen2.5:32b-instruct,"
                  "gpt-4.1-nano,gpt-4.1-mini,gpt-4.1")
MODELS = [m.strip() for m in os.environ.get("MODELS", DEFAULT_MODELS).split(",") if m.strip()]
OUT = os.environ.get("OUT", os.path.join(_ROOT, "outputs", "extractor_sweep.json"))
PRICE = {"gpt-4.1": (2.0, 8.0), "gpt-4.1-mini": (0.40, 1.60), "gpt-4.1-nano": (0.10, 0.40),
         "claude-sonnet-4-6": (3.0, 15.0), "claude-haiku-4-5": (0.80, 4.0)}  # claude prices approx
METHODS = ["narrative_graph", "narrative_struct"]


def client_for(model):
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        return OpenAICompatibleChatClient(model=model, base_url="https://api.openai.com/v1",
                                          api_key=os.environ.get("OPENAI_API_KEY"),
                                          temperature=0.0, use_structured_output=True)
    if model.startswith("claude-"):
        # native Anthropic /v1/messages; JSON via instruction-following (schema not enum-enforced)
        return AnthropicChatClient(model=model, api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
                                   max_tokens=2048)
    return OpenAICompatibleChatClient(model=model, base_url=OLLAMA, temperature=0.0,
                                      use_structured_output=True)


def run_model(model, raws):
    client = client_for(model)
    ctrl = StructuredSourceCalibratedController(client=client, name="structured")
    agg = {m: defaultdict(int) for m in METHODS}
    rows = {m: defaultdict(list) for m in METHODS}
    for i, raw in enumerate(raws):
        w, _ = validate_and_build(raw, f"sw_{i:03d}")
        if w is None:
            continue
        narrative = raw.get("narrative", "")
        cache = {}
        for q in build_queries(w):
            gold = w.epistemic_states[q.target_npc_id].best_claim(q.proposition_id)
            for m in METHODS:
                key = (m, q.proposition_id)
                if key not in cache:
                    cache[key] = (narrative_graph_states(w, narrative, q.proposition_id, client)
                                  if m == "narrative_graph"
                                  else narrative_struct_states(w, narrative, q.proposition_id, client))
                es = cache[key].get(q.target_npc_id, EpistemicState(npc_id=q.target_npc_id))
                eb = es.best_claim(q.proposition_id)
                a = agg[m]
                a["tot"] += 1
                gh, eh = gold is not None, eb is not None
                a["pres"] += int(gh == eh)
                if gh and eh:
                    a["both"] += 1
                    a["src"] += int(eb.source_type == gold.source_type)
                    a["val"] += int(_answer_matches(gold.value, eb.value))
                ew = dataclasses.replace(w, epistemic_states={**w.epistemic_states, q.target_npc_id: es})
                for k, v in evaluate_response(w, q, ctrl.respond(ew, q)).metrics.items():
                    if v is not None:
                        rows[m][k].append(v)
    ci, co = PRICE.get(model, (0.0, 0.0))
    cost = client.input_tokens / 1e6 * ci + client.output_tokens / 1e6 * co
    out = {"_cost": round(cost, 3)}
    for m in METHODS:
        a = agg[m]
        ec = rows[m].get("epistemic_consistency", [])
        olr = rows[m].get("omniscience_leakage", [])
        out[m] = {"src": round(a["src"] / max(1, a["both"]), 3),
                  "EC": round(statistics.mean(ec), 3) if ec else None,
                  "OLR": round(statistics.mean(olr), 3) if olr else None,
                  "pres": round(a["pres"] / max(1, a["tot"]), 3),
                  "val": round(a["val"] / max(1, a["both"]), 3)}
    return out


def main():
    raws = [json.loads(l) for l in open(PATH) if l.strip()][:MAXW]
    print(f"extractor sweep: {len(MODELS)} models on {len(raws)} worlds\nmodels: {MODELS}\n", flush=True)
    results = {}
    for model in MODELS:
        try:
            r = run_model(model, raws)
            results[model] = r
            g, s = r["narrative_graph"], r["narrative_struct"]
            print(f"[done] {model:34s} struct src={s['src']:.2f} EC={s['EC']} OLR={s['OLR']} "
                  f"| graph src={g['src']:.2f} EC={g['EC']} | cost=${r['_cost']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {model}: {str(exc)[:120]}", flush=True)
            results[model] = {"error": str(exc)[:200]}

    print("\n================== EXTRACTOR SWEEP (method robustness) ==================")
    print(f"{'model':34s}| {'graph(src/EC/OLR)':>20} | {'STRUCT(src/EC/OLR)':>21} | {'Δsrc':>6}{'ΔEC':>7}")
    total = 0.0
    for model in MODELS:
        r = results.get(model, {})
        if "error" in r:
            print(f"{model:34s}| ERROR: {r['error'][:60]}")
            continue
        g, s = r["narrative_graph"], r["narrative_struct"]
        dsrc = s["src"] - g["src"]
        dec = (s["EC"] or 0) - (g["EC"] or 0)
        print(f"{model:34s}| {g['src']:.2f}/{g['EC']}/{g['OLR']:>4} | "
              f"{s['src']:.2f}/{s['EC']}/{s['OLR']:>4} | {dsrc:+.2f} {dec:+.2f}")
        total += r.get("_cost", 0.0)
    print(f"\nreference: gold_structural ≈ src .56 / EC .64 / OLR .00 (same 8 worlds, model-agnostic graph)")
    print(f"Δ = narrative_struct − narrative_graph (our method's net gain). OpenAI cost ≈ ${total:.2f}")
    json.dump(results, open(OUT, "w"), indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
