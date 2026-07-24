"""Improved from-narrative extractor: DECOUPLE graph-reconstruction from propagation.
The weak part of narrative_graph (one joint call) is source_type (.35 vs gold .61); its values are
already good (.95). So we keep values from the joint call, and fix the TYPE: Stage 1 extracts the
information-flow GRAPH from the narrative (eyewitnesses / who-told-whom / who-only-heard-a-rumor);
Stage 2 applies the SAME deterministic propagation as gold_structural on the EXTRACTED graph to assign
source_type. Compares narrative_graph (old) vs narrative_struct (new) vs gold_structural (ceiling).
Local qwen by default."""
import dataclasses
import json
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, "/path/to/launderbench")
from eco_worlds import build_queries, validate_and_build
from epistemic_state import EpistemicState
from llm_controller import OpenAICompatibleChatClient, StructuredSourceCalibratedController
from evaluator import _answer_matches, evaluate_response
from graph_extract import STRONG
from graph_extract3 import _mkclaim, gold_structural_state, name2id, narrative_graph_states

PATH = os.environ.get("NARR_PATH", "outputs/eco_narrative.jsonl")
MODEL = os.environ.get("MODEL", "qwen2.5:32b-instruct")
BASE = os.environ.get("BASE", "http://localhost:11434/v1")
MAXW = int(os.environ.get("MAXW", "0"))

_FLOW_DEV = (
    "From the NARRATIVE, extract the information-flow graph about the central event. Return: "
    "`observers` = characters who witnessed the event FIRSTHAND; `telling` = pairs (teller, hearer) "
    "where one character told another about it; `rumor_hearers` = characters who only heard a "
    "crowd/market RUMOR (not a specific person). Use the exact character names. Return JSON.")


def _flow_schema():
    return {"type": "object", "additionalProperties": False,
            "properties": {
                "observers": {"type": "array", "items": {"type": "string"}},
                "telling": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                            "properties": {"teller": {"type": "string"}, "hearer": {"type": "string"}},
                            "required": ["teller", "hearer"]}},
                "rumor_hearers": {"type": "array", "items": {"type": "string"}}},
            "required": ["observers", "telling", "rumor_hearers"]}


def extract_flow_graph(narrative, names, client):
    try:
        return client.complete_json(
            [{"role": "developer", "content": _FLOW_DEV},
             {"role": "user", "content": json.dumps({"characters": names, "narrative": narrative}, ensure_ascii=True)}],
            response_schema=_flow_schema())
    except Exception:
        return {"observers": [], "telling": [], "rumor_hearers": []}


def narrative_struct_states(world, narrative, prop_id, client):
    """value from the joint call (already good); type from the EXTRACTED flow graph + propagation."""
    nm = name2id(world)
    flow = extract_flow_graph(narrative, [n.name for n in world.npcs], client)
    obs = {nm.get(str(x).strip().lower()) for x in flow.get("observers", [])}
    told_by = {}
    for t in flow.get("telling", []):
        if isinstance(t, dict):
            teller, hearer = t.get("teller", ""), t.get("hearer", "")
        elif isinstance(t, (list, tuple)) and len(t) >= 2:
            teller, hearer = t[0], t[1]
        else:
            continue
        h = nm.get(str(hearer).strip().lower())
        if h:
            told_by[h] = nm.get(str(teller).strip().lower()) or teller
    rumor = {nm.get(str(x).strip().lower()) for x in flow.get("rumor_hearers", [])}
    joint = narrative_graph_states(world, narrative, prop_id, client)  # for values
    states = {}
    for n in world.npcs:
        es = EpistemicState(npc_id=n.npc_id)
        jc = joint.get(n.npc_id).best_claim(prop_id) if joint.get(n.npc_id) else None
        if jc is None or not jc.value:
            states[n.npc_id] = es
            continue
        if n.npc_id in obs:
            st, agent = "direct_observation", None
        elif n.npc_id in told_by:
            st, agent = "testimony", told_by[n.npc_id]
        elif n.npc_id in rumor:
            st, agent = "hearsay", None
        else:
            st, agent = "role_knowledge", None  # has a value but no graph evidence -> role/inference default
        es.add_claim(_mkclaim(prop_id, narrative[:120], st, agent, jc.value))
        states[n.npc_id] = es
    return states


def main():
    raws = [json.loads(l) for l in open(PATH) if l.strip()]
    if MAXW:
        raws = raws[:MAXW]
    client = OpenAICompatibleChatClient(model=MODEL, base_url=BASE, api_key=os.environ.get("OPENAI_API_KEY"),
                                        temperature=0.0, use_structured_output=True)
    ctrl = StructuredSourceCalibratedController(client=client, name="structured")
    methods = ["narrative_graph", "narrative_struct", "gold_structural"]
    agg = {m: defaultdict(int) for m in methods}
    rows = {m: defaultdict(list) for m in methods}
    for i, raw in enumerate(raws):
        w, _ = validate_and_build(raw, f"nar_{i:03d}")
        if w is None:
            continue
        narrative = raw.get("narrative", "")
        cache = {}
        for q in build_queries(w):
            gold = w.epistemic_states[q.target_npc_id].best_claim(q.proposition_id)
            for m in methods:
                if m == "gold_structural":
                    es = gold_structural_state(w, q.target_npc_id, q.proposition_id, client)
                else:
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
        print(f"world {i} done", flush=True)

    def mean(m, k):
        v = rows[m].get(k, [])
        return statistics.mean(v) if v else float("nan")

    print(f"\n=== improved from-narrative extraction ({MODEL}, {len(raws)} worlds) ===")
    print(f"  {'method':18s}{'presence':>10}{'source_type':>13}{'value':>8} | {'EC':>6}{'OLR↓':>6}{'SAA':>6}")
    out = {}
    for m in methods:
        a = agg[m]
        out[m] = {"presence": round(a['pres']/max(1, a['tot']), 3), "src": round(a['src']/max(1, a['both']), 3),
                  "val": round(a['val']/max(1, a['both']), 3), "EC": round(mean(m, 'epistemic_consistency'), 3),
                  "OLR": round(mean(m, 'omniscience_leakage'), 3), "SAA": round(mean(m, 'source_attribution_accuracy'), 3)}
        print(f"  {m:18s}{a['pres']/max(1,a['tot']):>10.2f}{a['src']/max(1,a['both']):>13.2f}{a['val']/max(1,a['both']):>8.2f} | "
              f"{mean(m,'epistemic_consistency'):>6.2f}{mean(m,'omniscience_leakage'):>6.2f}{mean(m,'source_attribution_accuracy'):>6.2f}")
    json.dump(out, open("outputs/graph_extract4_results.json", "w"), indent=2)
    print("\nwrote outputs/graph_extract4_results.json")
    print("Goal: narrative_struct source_type/EC/OLR beats narrative_graph and approaches gold_structural.")


if __name__ == "__main__":
    main()
