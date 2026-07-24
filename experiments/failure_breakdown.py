"""WHERE does the best extractor fail? Runs the strongest holistic extractor (narrative_graph =
joint read of the whole narrative) and breaks down accuracy by (a) gold HOP-DEPTH from the eyewitness
and (b) gold SOURCE TYPE. Shows the cliff: provenance accuracy collapses as the chain deepens.
Env: MODEL(gpt-4.1), NARR_PATH(vhard), MAXW(8).
"""
import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eco_worlds import validate_and_build
from graph_extract3 import narrative_graph_states
from llm_controller import AnthropicChatClient, OpenAICompatibleChatClient
from evaluator import _answer_matches

MODEL = os.environ.get("MODEL", "gpt-4.1")
PATH = os.environ.get("NARR_PATH", os.path.join(_ROOT, "outputs", "eco_narrative_vhard.jsonl"))
MAXW = int(os.environ.get("MAXW", "8"))


def client_for(model):
    if model.startswith("claude-"):
        return AnthropicChatClient(model=model, api_key=os.environ.get("ANTHROPIC_API_KEY", ""), max_tokens=2048)
    base = "https://api.openai.com/v1" if model.startswith(("gpt-", "o1", "o3", "o4")) else os.environ.get("OLLAMA_BASE", "http://localhost:11434/v1")
    return OpenAICompatibleChatClient(model=model, base_url=base, api_key=os.environ.get("OPENAI_API_KEY"),
                                      temperature=0.0, use_structured_output=True)


def gold_depths(raw):
    npcs, claims = raw.get("npcs", []), raw.get("claims", [])
    ref = {}
    for n in npcs:
        for k in (n.get("id"), n.get("name"), n.get("role")):
            if k:
                ref[str(k).strip().lower()] = n["id"]
    teller = {}
    for c in claims:
        nid = ref.get(str(c.get("npc", "")).strip().lower())
        if not nid:
            continue
        sa = c.get("source_agent")
        teller[nid] = ref.get(str(sa).strip().lower()) if sa else None

    def depth(nid, seen):
        if nid in seen:
            return 0
        seen = seen | {nid}
        t = teller.get(nid)
        if not t or t not in teller:
            return 1 if t else 0
        return 1 + depth(t, seen)

    return {nid: depth(nid, set()) for nid in teller}


def main():
    raws = [json.loads(l) for l in open(PATH) if l.strip()][:MAXW]
    client = client_for(MODEL)
    by_depth = defaultdict(lambda: defaultdict(int))
    by_type = defaultdict(lambda: defaultdict(int))
    for i, raw in enumerate(raws):
        w, _ = validate_and_build(raw, f"fb_{i:03d}")
        if w is None:
            continue
        prop = w.primary_fact().proposition_id
        depths = gold_depths(raw)
        ext = narrative_graph_states(w, raw.get("narrative", ""), prop, client)
        for n in w.npcs:
            gold = w.epistemic_states[n.npc_id].best_claim(prop)
            if not gold:
                continue
            d = depths.get(n.npc_id, 0)
            dk = "5+" if d >= 5 else str(d)
            eb = ext.get(n.npc_id).best_claim(prop) if ext.get(n.npc_id) else None
            for bucket, key in ((by_depth, dk), (by_type, gold.source_type)):
                bucket[key]["n"] += 1
                bucket[key]["pres"] += int(eb is not None)
                if eb is not None:
                    bucket[key]["src"] += int(eb.source_type == gold.source_type)
                    bucket[key]["val"] += int(_answer_matches(gold.value, eb.value))
        print(f"world {i} done", flush=True)

    def show(title, bucket, order):
        print(f"\n=== {title} ({MODEL}, holistic extractor) ===")
        print(f"  {'bucket':14s}{'n':>5}{'presence':>10}{'source_type':>13}{'value':>8}")
        for k in order:
            if k not in bucket:
                continue
            b = bucket[k]
            n = b["n"]
            print(f"  {k:14s}{n:>5}{b['pres']/max(1,n):>10.2f}"
                  f"{b['src']/max(1,b['pres']):>13.2f}{b['val']/max(1,b['pres']):>8.2f}")

    show("accuracy by GOLD HOP-DEPTH (0=eyewitness)", by_depth, ["0", "1", "2", "3", "4", "5+"])
    show("accuracy by GOLD SOURCE TYPE", by_type,
         ["direct_observation", "testimony", "hearsay", "inference", "environmental_evidence",
          "role_knowledge", "public_record", "episodic_memory"])
    print("\nThe collapse of source_type accuracy with hop-depth = the unsolved problem to optimize.")


if __name__ == "__main__":
    main()
