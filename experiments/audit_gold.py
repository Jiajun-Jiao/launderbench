"""Per-NPC audit: run the holistic extractor on vhard worlds and print gold vs predicted source_type,
so we can adjudicate whether the .00 hearsay collapse is the MODEL failing the clean true chain (real)
or the GOLD being debatable on the false/inference chains (artifact). Env: MODEL(claude-opus-4-8) MAXW(2)
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eco_worlds import validate_and_build
from graph_extract3 import narrative_graph_states
from failure_breakdown import gold_depths, client_for

MODEL = os.environ.get("MODEL", "claude-opus-4-8")
PATH = os.path.join(_ROOT, "outputs", "eco_narrative_vhard.jsonl")
MAXW = int(os.environ.get("MAXW", "2"))


def main():
    raws = [json.loads(l) for l in open(PATH) if l.strip()][:MAXW]
    client = client_for(MODEL)
    print(f"gold audit: model={MODEL}", flush=True)
    for i, raw in enumerate(raws):
        w, _ = validate_and_build(raw, f"au_{i:03d}")
        if w is None:
            print(f"world {i}: build failed"); continue
        prop = w.primary_fact().proposition_id
        depths = gold_depths(raw)
        ext = narrative_graph_states(w, raw.get("narrative", ""), prop, client)
        nm = {n.npc_id: n.name for n in w.npcs}
        print(f"\n===== WORLD {i} (true={raw.get('true_value')} false={raw.get('false_value')}) =====")
        print(f"  {'npc':10s}{'depth':>6}  {'GOLD':22s}{'PRED':22s} match")
        n_match = n_tot = 0
        for n in w.npcs:
            gold = w.epistemic_states[n.npc_id].best_claim(prop)
            if not gold:
                continue
            eb = ext.get(n.npc_id).best_claim(prop) if ext.get(n.npc_id) else None
            pred = eb.source_type if eb else "(absent)"
            d = depths.get(n.npc_id, 0)
            ok = eb is not None and eb.source_type == gold.source_type
            n_tot += 1; n_match += int(ok)
            flag = "" if ok else "  <-- MISMATCH"
            print(f"  {nm.get(n.npc_id, n.npc_id):10s}{d:>6}  {gold.source_type:22s}{pred:22s}{'Y' if ok else 'N'}{flag}")
        print(f"  world {i} source_type match: {n_match}/{n_tot}")
    print("\nADJUDICATE: for each MISMATCH, is the GOLD defensible (model wrong=real) or is the model's")
    print("reading defensible (gold wrong=artifact)? Focus on the CLEAN true chain vs the inference/role/false chain.")


if __name__ == "__main__":
    main()
