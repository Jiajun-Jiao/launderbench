"""Real method (v1): graph-grounded neuro-symbolic provenance extraction.
The §9 one-shot LLM extractor guesses presence + source_type + value in a single call
(source-type acc .46). Here we SEPARATE the sub-problems: presence and source_type are decided by
the information-flow STRUCTURE (event observers + who-told-whom edges) deterministically, and the LLM
is used ONLY to ground the believed VALUE from the NPC's own memory note(s). We compare graph-grounded
vs one-shot on the same data: extraction accuracy (presence / source_type / value) and downstream
(the structured controller run on the EXTRACTED state, judged vs ground truth). Local qwen by default."""
import dataclasses
import json
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, "/path/to/launderbench")
from query_generator import generate_dataset
from epistemic_state import Claim, EpistemicState
from llm_controller import OpenAICompatibleChatClient, StructuredSourceCalibratedController
from evaluator import _answer_matches, evaluate_response
from provenance_experiment import candidates_for, extract_state  # one-shot LLM baseline

SEEDS = [int(x) for x in os.environ.get("SEEDS", "7,11").split(",")]
WORLDS = int(os.environ.get("WORLDS", "5"))
QPW = int(os.environ.get("QPW", "4"))
MODEL = os.environ.get("MODEL", "qwen2.5:32b-instruct")
BASE = os.environ.get("BASE", "http://localhost:11434/v1")
STRONG = {"direct_observation", "role_knowledge", "public_record", "episodic_memory", "testimony"}

_VAL_DEV = (
    "You are given ONE NPC's own memory note(s) about a question, and the list of possible answer "
    "values. Output the single value (copied verbatim from the list) that THIS NPC believes from its "
    "notes — even if it may be wrong/distorted; null if the notes express no value. Return JSON.")


def _val_schema(cands):
    return {"type": "object", "additionalProperties": False,
            "properties": {"reasoning": {"type": "string"},
                           "value": {"type": ["string", "null"], "enum": [*cands, None]}},
            "required": ["reasoning", "value"]}


def llm_value(client, prop_text, mems, cands):
    try:
        p = client.complete_json(
            [{"role": "developer", "content": _VAL_DEV},
             {"role": "user", "content": json.dumps({"question": prop_text, "npc_memory_notes": mems,
                                                      "possible_values": cands}, ensure_ascii=True)}],
            response_schema=_val_schema(cands))
        return p.get("value")
    except Exception:
        return None


def graph_extract_state(world, npc_id, prop_id, client):
    """Structure decides presence + source_type; LLM grounds only the value."""
    es = EpistemicState(npc_id=npc_id)
    mems = [c.proposition for c in world.epistemic_states[npc_id].claims_for(prop_id)]  # raw notes = legit input
    if not mems:
        return es, {"source_type": "unknown", "value": None}  # no note -> no knowledge (presence by construction)
    observers = set()
    for e in world.events:
        observers |= set(e.observers)
    incoming = [g.get("source") for g in world.source_graph
                if g.get("target") == npc_id and g.get("proposition_id") in (None, prop_id)]
    if npc_id in observers:
        st = "direct_observation"
    elif incoming:
        src = incoming[0]
        st = "testimony" if (src and src not in ("?", "crowd", "")) else "hearsay"
    else:
        st = "role_knowledge"  # has a note but no observation/edge -> role/inference; default role
    val = llm_value(client, world.facts[prop_id].proposition, mems, candidates_for(world, prop_id))
    if val:
        rel = 0.85 if st in STRONG else 0.4
        es.add_claim(Claim(proposition_id=prop_id, proposition=mems[0], value=val, source_type=st,
                           source_agent=(incoming[0] if incoming else None), time_obtained=1,
                           reliability=rel, verification_status="verified" if rel >= 0.7 else "unverified"))
    return es, {"source_type": st, "value": val}


def main():
    client = OpenAICompatibleChatClient(model=MODEL, base_url=BASE, api_key=os.environ.get("OPENAI_API_KEY"),
                                        temperature=0.0, use_structured_output=True)
    ctrl = StructuredSourceCalibratedController(client=client, name="structured")
    agg = {"oneshot": defaultdict(int), "graph": defaultdict(int)}
    rows = {"oneshot": defaultdict(list), "graph": defaultdict(list)}
    for seed in SEEDS:
        worlds, queries = generate_dataset(WORLDS, QPW, seed)
        by = {w.world_id: w for w in worlds}
        for q in queries:
            gw = by[q.world_id]
            gold = gw.epistemic_states[q.target_npc_id].best_claim(q.proposition_id)
            for method, fn in (("oneshot", extract_state), ("graph", graph_extract_state)):
                es, _ = fn(gw, q.target_npc_id, q.proposition_id, client)
                eb = es.best_claim(q.proposition_id)
                a = agg[method]
                a["tot"] += 1
                gh, eh = gold is not None, eb is not None
                a["pres"] += int(gh == eh)
                if gh and eh:
                    a["both"] += 1
                    a["src"] += int(eb.source_type == gold.source_type)
                    a["val"] += int(_answer_matches(gold.value, eb.value))
                ew = dataclasses.replace(gw, epistemic_states={**gw.epistemic_states, q.target_npc_id: es})
                for m, v in evaluate_response(gw, q, ctrl.respond(ew, q)).metrics.items():
                    if v is not None:
                        rows[method][m].append(v)
        print(f"seed {seed} done", flush=True)

    def mean(method, k):
        v = rows[method].get(k, [])
        return statistics.mean(v) if v else float("nan")

    print(f"\n=== provenance EXTRACTION accuracy: one-shot LLM vs graph-grounded ({MODEL}) ===")
    print(f"  {'method':10s}{'presence':>10}{'source_type':>13}{'value':>8}")
    for m in ("oneshot", "graph"):
        a = agg[m]
        print(f"  {m:10s}{a['pres']/a['tot']:>10.2f}{a['src']/max(1,a['both']):>13.2f}{a['val']/max(1,a['both']):>8.2f}")
    print(f"\n  downstream (structured controller on the EXTRACTED state, judged vs gold):")
    print(f"  {'method':10s}{'EC':>7}{'OLR↓':>7}{'SAA':>7}")
    for m in ("oneshot", "graph"):
        print(f"  {m:10s}{mean(m,'epistemic_consistency'):>7.2f}{mean(m,'omniscience_leakage'):>7.2f}{mean(m,'source_attribution_accuracy'):>7.2f}")
    out = {m: {"presence": round(agg[m]['pres']/agg[m]['tot'], 3), "src": round(agg[m]['src']/max(1, agg[m]['both']), 3),
               "val": round(agg[m]['val']/max(1, agg[m]['both']), 3), "EC": round(mean(m, 'epistemic_consistency'), 3),
               "OLR": round(mean(m, 'omniscience_leakage'), 3), "SAA": round(mean(m, 'source_attribution_accuracy'), 3)}
           for m in ("oneshot", "graph")}
    json.dump(out, open("outputs/graph_extract_results.json", "w"), indent=2)
    print("\nwrote outputs/graph_extract_results.json")


if __name__ == "__main__":
    main()
