"""Push extraction precision: self-consistency on the narrative flow-graph extraction.
Sample the flow-graph extractor K times (temperature>0) and MAJORITY-VOTE each relation
(observer / telling-edge / rumor-hearer) before propagating. Compares K=1 (current narrative_struct)
vs K=5 (self-consistency) vs gold_structural (ceiling) on the narrative worlds. Values come from the
joint call (already strong). Local qwen by default."""
import dataclasses
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/path/to/launderbench")
from eco_worlds import build_queries, validate_and_build
from epistemic_state import EpistemicState
from llm_controller import OpenAICompatibleChatClient, StructuredSourceCalibratedController
from evaluator import _answer_matches, evaluate_response
from graph_extract3 import _mkclaim, gold_structural_state, name2id, narrative_graph_states
from graph_extract4 import extract_flow_graph

PATH = os.environ.get("NARR_PATH", "outputs/eco_narrative.jsonl")
MODEL = os.environ.get("MODEL", "qwen2.5:32b-instruct")
BASE = os.environ.get("BASE", "http://localhost:11434/v1")
MAXW = int(os.environ.get("MAXW", "0"))
K = int(os.environ.get("K", "5"))


def vote_flow(narrative, names, hot, k):
    obs, tell, rum = Counter(), Counter(), Counter()
    for _ in range(k):
        f = extract_flow_graph(narrative, names, hot)
        for x in f.get("observers", []):
            obs[str(x).strip().lower()] += 1
        for t in f.get("telling", []):
            tell[(str(t.get("teller", "")).strip().lower(), str(t.get("hearer", "")).strip().lower())] += 1
        for x in f.get("rumor_hearers", []):
            rum[str(x).strip().lower()] += 1
    thr = (k // 2) + 1
    return ({o for o, c in obs.items() if c >= thr},
            {(a, b) for (a, b), c in tell.items() if c >= thr},
            {r for r, c in rum.items() if c >= thr})


def struct_from_flow(world, narrative, prop_id, obs_n, tell_n, rum_n, cold):
    nm = name2id(world)
    obs = {nm.get(o) for o in obs_n}
    told_by = {nm.get(b): (nm.get(a) or a) for (a, b) in tell_n if nm.get(b)}
    rumor = {nm.get(r) for r in rum_n}
    joint = narrative_graph_states(world, narrative, prop_id, cold)  # values
    states = {}
    for n in world.npcs:
        es = EpistemicState(npc_id=n.npc_id)
        jc = joint.get(n.npc_id).best_claim(prop_id) if joint.get(n.npc_id) else None
        if jc and jc.value:
            if n.npc_id in obs:
                st, ag = "direct_observation", None
            elif n.npc_id in told_by:
                st, ag = "testimony", told_by[n.npc_id]
            elif n.npc_id in rumor:
                st, ag = "hearsay", None
            else:
                st, ag = "role_knowledge", None
            es.add_claim(_mkclaim(prop_id, narrative[:120], st, ag, jc.value))
        states[n.npc_id] = es
    return states


def main():
    raws = [json.loads(l) for l in open(PATH) if l.strip()]
    if MAXW:
        raws = raws[:MAXW]
    cold = OpenAICompatibleChatClient(model=MODEL, base_url=BASE, api_key=os.environ.get("OPENAI_API_KEY"),
                                      temperature=0.0, use_structured_output=True)
    hot = OpenAICompatibleChatClient(model=MODEL, base_url=BASE, api_key=os.environ.get("OPENAI_API_KEY"),
                                     temperature=0.7, use_structured_output=True)
    ctrl = StructuredSourceCalibratedController(client=cold, name="structured")
    methods = ["struct_k1", f"struct_k{K}", "gold_structural"]
    agg = {m: defaultdict(int) for m in methods}
    rows = {m: defaultdict(list) for m in methods}
    for i, raw in enumerate(raws):
        w, _ = validate_and_build(raw, f"nar_{i:03d}")
        if w is None:
            continue
        narrative = raw.get("narrative", "")
        names = [n.name for n in w.npcs]
        cache = {}
        for q in build_queries(w):
            gold = w.epistemic_states[q.target_npc_id].best_claim(q.proposition_id)
            for m in methods:
                if m == "gold_structural":
                    es = gold_structural_state(w, q.target_npc_id, q.proposition_id, cold)
                else:
                    key = (m, q.proposition_id)
                    if key not in cache:
                        k = 1 if m == "struct_k1" else K
                        client = cold if k == 1 else hot
                        o, t, r = vote_flow(narrative, names, client, k)
                        cache[key] = struct_from_flow(w, narrative, q.proposition_id, o, t, r, cold)
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
                for kk, v in evaluate_response(w, q, ctrl.respond(ew, q)).metrics.items():
                    if v is not None:
                        rows[m][kk].append(v)
        print(f"world {i} done", flush=True)

    def mean(m, kk):
        v = rows[m].get(kk, [])
        return statistics.mean(v) if v else float("nan")

    print(f"\n=== self-consistency on flow-graph ({MODEL}, {len(raws)} worlds, K={K}) ===")
    print(f"  {'method':16s}{'presence':>10}{'source_type':>13}{'value':>8} | {'EC':>6}{'OLR↓':>6}{'SAA':>6}")
    out = {}
    for m in methods:
        a = agg[m]
        out[m] = {"presence": round(a['pres']/max(1, a['tot']), 3), "src": round(a['src']/max(1, a['both']), 3),
                  "val": round(a['val']/max(1, a['both']), 3), "EC": round(mean(m, 'epistemic_consistency'), 3),
                  "OLR": round(mean(m, 'omniscience_leakage'), 3), "SAA": round(mean(m, 'source_attribution_accuracy'), 3)}
        print(f"  {m:16s}{a['pres']/max(1,a['tot']):>10.2f}{a['src']/max(1,a['both']):>13.2f}{a['val']/max(1,a['both']):>8.2f} | "
              f"{mean(m,'epistemic_consistency'):>6.2f}{mean(m,'omniscience_leakage'):>6.2f}{mean(m,'source_attribution_accuracy'):>6.2f}")
    json.dump(out, open("outputs/graph_extract5_results.json", "w"), indent=2)
    print("\nwrote outputs/graph_extract5_results.json")


if __name__ == "__main__":
    main()
