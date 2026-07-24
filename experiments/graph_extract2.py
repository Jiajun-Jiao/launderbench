"""Step 2 — extract provenance from PROSE (the genuinely novel/hard step per the novelty audit).
Step-1 (`graph_extract.py`) decided source_type from a GIVEN information-flow graph (observers/edges).
Here source_type must come from the NPC's memory-text PROSE (which implies, but never names, the type —
i.e. linguistic evidentiality). We compare four extractors on the same data:
  one-shot   : §9 joint LLM guess (events+channels+memory+candidates -> everything)
  structural : step-1 (graph decides type; LLM only value)         [uses given structure]
  prose      : LLM evidentiality classifier on memory text only -> type; LLM value   [pure prose]
  combined   : structure where it fires (observer/edge), prose otherwise; LLM value  [best of both]
Question: how much of step-1's gain survives when type is read from text, not handed over?
Local qwen by default; key from env if an OpenAI extractor is used."""
import dataclasses
import json
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, "/path/to/launderbench")
from query_generator import generate_dataset
from epistemic_state import Claim, EpistemicState, SourceType
from llm_controller import OpenAICompatibleChatClient, StructuredSourceCalibratedController
from evaluator import _answer_matches, evaluate_response
from provenance_experiment import candidates_for, extract_state
from graph_extract import STRONG, llm_value

SOURCE_TYPES = [s.value for s in SourceType if s.value != "forbidden_global_state"]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "7,11").split(",")]
WORLDS = int(os.environ.get("WORLDS", "5"))
QPW = int(os.environ.get("QPW", "4"))
MODEL = os.environ.get("MODEL", "qwen2.5:32b-instruct")
BASE = os.environ.get("BASE", "http://localhost:11434/v1")

_EVID_DEV = (
    "Read ONE NPC's own memory note(s). Classify HOW this NPC knows the matter, choosing the single "
    "best source_type from: direct_observation (it personally saw/witnessed it), testimony (a specific "
    "named person told it), hearsay (a rumor / crowd / 'people are saying'), environmental_evidence "
    "(physical clues), inference (deduced by reasoning), role_knowledge (knows by virtue of its job/role), "
    "public_record, episodic_memory. The note IMPLIES this but will NOT name the type. Return JSON.")


def _evid_schema():
    return {"type": "object", "additionalProperties": False,
            "properties": {"reasoning": {"type": "string"},
                           "source_type": {"type": "string", "enum": SOURCE_TYPES}},
            "required": ["reasoning", "source_type"]}


def prose_type(client, mems):
    try:
        p = client.complete_json(
            [{"role": "developer", "content": _EVID_DEV},
             {"role": "user", "content": json.dumps({"memory_notes": mems}, ensure_ascii=True)}],
            response_schema=_evid_schema())
        return p.get("source_type")
    except Exception:
        return None


def _structural_type(world, npc_id, prop_id):
    observers = set()
    for e in world.events:
        observers |= set(e.observers)
    incoming = [g.get("source") for g in world.source_graph
                if g.get("target") == npc_id and g.get("proposition_id") in (None, prop_id)]
    if npc_id in observers:
        return "direct_observation", None, True
    if incoming:
        src = incoming[0]
        return ("testimony" if (src and src not in ("?", "crowd", "")) else "hearsay"), src, True
    return None, None, False


def build(world, npc_id, prop_id, client, mode):
    es = EpistemicState(npc_id=npc_id)
    mems = [c.proposition for c in world.epistemic_states[npc_id].claims_for(prop_id)]
    if not mems:
        return es
    st_struct, src, confident = _structural_type(world, npc_id, prop_id)
    agent = None
    if mode == "structural":
        st, agent = (st_struct or "role_knowledge"), src
    elif mode == "prose":
        st = prose_type(client, mems) or "role_knowledge"
    else:  # combined
        if confident:
            st, agent = st_struct, src
        else:
            st = prose_type(client, mems) or "role_knowledge"
    val = llm_value(client, world.facts[prop_id].proposition, mems, candidates_for(world, prop_id))
    if val:
        rel = 0.85 if st in STRONG else 0.4
        es.add_claim(Claim(proposition_id=prop_id, proposition=mems[0], value=val, source_type=st,
                           source_agent=agent, time_obtained=1, reliability=rel,
                           verification_status="verified" if rel >= 0.7 else "unverified"))
    return es


def main():
    client = OpenAICompatibleChatClient(model=MODEL, base_url=BASE, api_key=os.environ.get("OPENAI_API_KEY"),
                                        temperature=0.0, use_structured_output=True)
    ctrl = StructuredSourceCalibratedController(client=client, name="structured")
    methods = ["oneshot", "structural", "prose", "combined"]
    agg = {m: defaultdict(int) for m in methods}
    rows = {m: defaultdict(list) for m in methods}
    for seed in SEEDS:
        worlds, queries = generate_dataset(WORLDS, QPW, seed)
        by = {w.world_id: w for w in worlds}
        for q in queries:
            gw = by[q.world_id]
            gold = gw.epistemic_states[q.target_npc_id].best_claim(q.proposition_id)
            for m in methods:
                es = (extract_state(gw, q.target_npc_id, q.proposition_id, client)[0]
                      if m == "oneshot" else build(gw, q.target_npc_id, q.proposition_id, client, m))
                eb = es.best_claim(q.proposition_id)
                a = agg[m]
                a["tot"] += 1
                gh, eh = gold is not None, eb is not None
                a["pres"] += int(gh == eh)
                if gh and eh:
                    a["both"] += 1
                    a["src"] += int(eb.source_type == gold.source_type)
                    a["val"] += int(_answer_matches(gold.value, eb.value))
                ew = dataclasses.replace(gw, epistemic_states={**gw.epistemic_states, q.target_npc_id: es})
                for k, v in evaluate_response(gw, q, ctrl.respond(ew, q)).metrics.items():
                    if v is not None:
                        rows[m][k].append(v)
        print(f"seed {seed} done", flush=True)

    def mean(m, k):
        v = rows[m].get(k, [])
        return statistics.mean(v) if v else float("nan")

    print(f"\n=== provenance extraction: 4 extractors ({MODEL}) ===")
    print(f"  {'method':12s}{'presence':>10}{'source_type':>13}{'value':>8} | {'EC':>6}{'OLR↓':>6}{'SAA':>6}")
    out = {}
    for m in methods:
        a = agg[m]
        out[m] = {"presence": round(a['pres']/a['tot'], 3), "src": round(a['src']/max(1, a['both']), 3),
                  "val": round(a['val']/max(1, a['both']), 3), "EC": round(mean(m, 'epistemic_consistency'), 3),
                  "OLR": round(mean(m, 'omniscience_leakage'), 3), "SAA": round(mean(m, 'source_attribution_accuracy'), 3)}
        print(f"  {m:12s}{a['pres']/a['tot']:>10.2f}{a['src']/max(1,a['both']):>13.2f}{a['val']/max(1,a['both']):>8.2f} | "
              f"{mean(m,'epistemic_consistency'):>6.2f}{mean(m,'omniscience_leakage'):>6.2f}{mean(m,'source_attribution_accuracy'):>6.2f}")
    json.dump(out, open("outputs/graph_extract2_results.json", "w"), indent=2)
    print("\nwrote outputs/graph_extract2_results.json")
    print("Reading: does 'prose' (type from text) recover step-1's gain? does 'combined' match/beat 'structural'?")


if __name__ == "__main__":
    main()
