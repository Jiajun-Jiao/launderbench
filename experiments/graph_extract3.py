"""Real step-2 part B: reconstruct the info-flow graph from NARRATIVE PROSE, then propagate provenance.
On narrative worlds (eco_narrative.jsonl) compares three extractors (input = the narrative text only):
  narrative_oneshot : per-NPC LLM call from the narrative -> type + value (isolated / naive)
  narrative_graph   : ONE joint LLM call over the whole narrative -> every character's
                      (how-they-know / who-told-them / value), i.e. reconstruct the relational
                      who-saw/who-told structure from text, then map to provenance (the novel method)
  gold_structural   : use the GOLD structured graph (upper bound — if extraction were perfect)
Reports source_type accuracy + downstream EC/OLR (structured controller on the extracted state, judged
vs ground truth). Local qwen by default."""
import dataclasses
import json
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, "/path/to/launderbench")
from eco_worlds import build_queries, validate_and_build
from epistemic_state import Claim, EpistemicState
from llm_controller import OpenAICompatibleChatClient, StructuredSourceCalibratedController
from evaluator import _answer_matches, evaluate_response
from provenance_experiment import candidates_for
from graph_extract import STRONG, llm_value
from graph_extract2 import _structural_type

PATH = os.environ.get("NARR_PATH", "outputs/eco_narrative.jsonl")
MODEL = os.environ.get("MODEL", "qwen2.5:32b-instruct")
BASE = os.environ.get("BASE", "http://localhost:11434/v1")
MAXW = int(os.environ.get("MAXW", "0"))  # 0 = all
HOW2ST = {"observed": "direct_observation", "told": "testimony", "rumor": "hearsay",
          "inferred": "inference", "role": "role_knowledge", "none": None}

_GRAPH_DEV = (
    "Read the NARRATIVE about an event. For EACH listed character, determine — using ONLY the narrative "
    "— whether it knows the answer to the question, and HOW it came to know: 'observed' (saw it "
    "firsthand), 'told' (a specific named person told it — give told_by), 'rumor' (heard a crowd/market "
    "rumor), 'inferred', 'role' (knows by its job), or 'none' (no knowledge). Also give the value it "
    "believes (from the candidate list; may be a distorted rumor value). Return JSON for ALL characters.")
_ONESHOT_DEV = (
    "Read the NARRATIVE about an event, then answer for ONE target character — using ONLY the narrative: "
    "does it know the answer; HOW (observed / told / rumor / inferred / role / none); told_by (name or "
    "null); and the value it believes (from candidates). Return JSON.")


def _graph_schema(cands):
    item = {"type": "object", "additionalProperties": False,
            "properties": {"name": {"type": "string"}, "knows": {"type": "boolean"},
                           "how": {"type": "string", "enum": list(HOW2ST)},
                           "told_by": {"type": ["string", "null"]},
                           "value": {"type": ["string", "null"], "enum": [*cands, None]}},
            "required": ["name", "knows", "how", "told_by", "value"]}
    return {"type": "object", "additionalProperties": False,
            "properties": {"agents": {"type": "array", "items": item}}, "required": ["agents"]}


def _oneshot_schema(cands):
    return {"type": "object", "additionalProperties": False,
            "properties": {"knows": {"type": "boolean"}, "how": {"type": "string", "enum": list(HOW2ST)},
                           "told_by": {"type": ["string", "null"]},
                           "value": {"type": ["string", "null"], "enum": [*cands, None]}},
            "required": ["knows", "how", "told_by", "value"]}


def _mkclaim(prop_id, mem, st, agent, val):
    rel = 0.85 if st in STRONG else 0.4
    return Claim(proposition_id=prop_id, proposition=mem, value=val, source_type=st, source_agent=agent,
                 time_obtained=1, reliability=rel, verification_status="verified" if rel >= 0.7 else "unverified")


def name2id(world):
    m = {}
    for n in world.npcs:
        for k in (n.npc_id, n.name, n.role):
            if k:
                m[str(k).strip().lower()] = n.npc_id
    return m


def narrative_graph_states(world, narrative, prop_id, client):
    """One joint call over the narrative -> EpistemicState for every npc."""
    cands = candidates_for(world, prop_id)
    fact = world.facts[prop_id]
    payload = {"question": fact.proposition, "characters": [n.name for n in world.npcs],
               "candidate_values": cands, "narrative": narrative}
    states = {n.npc_id: EpistemicState(npc_id=n.npc_id) for n in world.npcs}
    try:
        out = client.complete_json([{"role": "developer", "content": _GRAPH_DEV},
                                    {"role": "user", "content": json.dumps(payload, ensure_ascii=True)}],
                                   response_schema=_graph_schema(cands))
    except Exception:
        return states
    nm = name2id(world)
    for a in out.get("agents", []):
        nid = nm.get(str(a.get("name", "")).strip().lower())
        if not nid or not a.get("knows"):
            continue
        st = HOW2ST.get(a.get("how"))
        if st is None or not a.get("value"):
            continue
        agent = nm.get(str(a.get("told_by") or "").strip().lower()) or a.get("told_by")
        states[nid].add_claim(_mkclaim(prop_id, narrative[:120], st, agent, a["value"]))
    return states


def narrative_oneshot_state(world, npc_id, narrative, prop_id, client):
    cands = candidates_for(world, prop_id)
    fact = world.facts[prop_id]
    es = EpistemicState(npc_id=npc_id)
    payload = {"question": fact.proposition, "target_character": world.npc(npc_id).name,
               "candidate_values": cands, "narrative": narrative}
    try:
        a = client.complete_json([{"role": "developer", "content": _ONESHOT_DEV},
                                  {"role": "user", "content": json.dumps(payload, ensure_ascii=True)}],
                                 response_schema=_oneshot_schema(cands))
    except Exception:
        return es
    if a.get("knows") and a.get("value"):
        st = HOW2ST.get(a.get("how"))
        if st:
            es.add_claim(_mkclaim(prop_id, narrative[:120], st, a.get("told_by"), a["value"]))
    return es


def gold_structural_state(world, npc_id, prop_id, client):
    es = EpistemicState(npc_id=npc_id)
    mems = [c.proposition for c in world.epistemic_states[npc_id].claims_for(prop_id)]
    if not mems:
        return es
    st, src, _ = _structural_type(world, npc_id, prop_id)
    st = st or "role_knowledge"
    val = llm_value(client, world.facts[prop_id].proposition, mems, candidates_for(world, prop_id))
    if val:
        es.add_claim(_mkclaim(prop_id, mems[0], st, src, val))
    return es


def main():
    raws = [json.loads(l) for l in open(PATH) if l.strip()]
    if MAXW:
        raws = raws[:MAXW]
    client = OpenAICompatibleChatClient(model=MODEL, base_url=BASE, api_key=os.environ.get("OPENAI_API_KEY"),
                                        temperature=0.0, use_structured_output=True)
    ctrl = StructuredSourceCalibratedController(client=client, name="structured")
    methods = ["narrative_oneshot", "narrative_graph", "gold_structural"]
    agg = {m: defaultdict(int) for m in methods}
    rows = {m: defaultdict(list) for m in methods}
    for i, raw in enumerate(raws):
        w, msg = validate_and_build(raw, f"nar_{i:03d}")
        if w is None:
            continue
        narrative = raw.get("narrative", "")
        queries = build_queries(w)
        gcache = {}
        for q in queries:
            gold = w.epistemic_states[q.target_npc_id].best_claim(q.proposition_id)
            for m in methods:
                if m == "narrative_oneshot":
                    es = narrative_oneshot_state(w, q.target_npc_id, narrative, q.proposition_id, client)
                elif m == "narrative_graph":
                    if q.proposition_id not in gcache:
                        gcache[q.proposition_id] = narrative_graph_states(w, narrative, q.proposition_id, client)
                    es = gcache[q.proposition_id].get(q.target_npc_id, EpistemicState(npc_id=q.target_npc_id))
                else:
                    es = gold_structural_state(w, q.target_npc_id, q.proposition_id, client)
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

    print(f"\n=== extract from NARRATIVE prose ({MODEL}, {len(raws)} worlds) ===")
    print(f"  {'method':18s}{'presence':>10}{'source_type':>13}{'value':>8} | {'EC':>6}{'OLR↓':>6}{'SAA':>6}")
    out = {}
    for m in methods:
        a = agg[m]
        out[m] = {"presence": round(a['pres']/max(1, a['tot']), 3), "src": round(a['src']/max(1, a['both']), 3),
                  "val": round(a['val']/max(1, a['both']), 3), "EC": round(mean(m, 'epistemic_consistency'), 3),
                  "OLR": round(mean(m, 'omniscience_leakage'), 3), "SAA": round(mean(m, 'source_attribution_accuracy'), 3)}
        print(f"  {m:18s}{a['pres']/max(1,a['tot']):>10.2f}{a['src']/max(1,a['both']):>13.2f}{a['val']/max(1,a['both']):>8.2f} | "
              f"{mean(m,'epistemic_consistency'):>6.2f}{mean(m,'omniscience_leakage'):>6.2f}{mean(m,'source_attribution_accuracy'):>6.2f}")
    json.dump(out, open("outputs/graph_extract3_results.json", "w"), indent=2)
    print("\nwrote outputs/graph_extract3_results.json")
    print("Reading: does narrative_graph (reconstruct relational structure from text) beat narrative_oneshot,")
    print("approaching the gold_structural upper bound? That validates 'recover the graph from narrative'.")


if __name__ == "__main__":
    main()
