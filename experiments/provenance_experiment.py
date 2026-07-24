"""Experiment 1: close the provenance-extraction loop.
An LLM extractor reconstructs each NPC's provenance-typed epistemic state from RAW inputs
(events + observers + info channels + the NPC's raw memory texts, with source_type/value
labels HIDDEN). We then (a) score extraction accuracy vs gold, and (b) run the structured
controller on the EXTRACTED state, judged against ground truth, vs the same controller on the
gold state. The gap = cost of imperfect provenance extraction.  Runs free on local Ollama."""
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

SOURCE_TYPES = [s.value for s in SourceType]
STRONG = {"direct_observation", "role_knowledge", "public_record", "episodic_memory", "testimony"}
SEEDS = [int(x) for x in os.environ.get("SEEDS", "7").split(",")]
WORLDS = int(os.environ.get("WORLDS", "4"))
QPW = int(os.environ.get("QPW", "3"))
EXTRACTOR_MODEL = os.environ.get("EXTRACTOR_MODEL", os.environ.get("MODEL", "qwen2.5:32b-instruct"))
EXTRACTOR_BASE = os.environ.get("EXTRACTOR_BASE", "http://localhost:11434/v1")
CTRL_MODEL = os.environ.get("CTRL_MODEL", "qwen2.5:32b-instruct")  # fixed across the curve
CTRL_BASE = os.environ.get("CTRL_BASE", "http://localhost:11434/v1")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

def make_schema(candidates):
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reasoning": {"type": "string"},
            "has_knowledge": {"type": "boolean"},
            "value": {"type": ["string", "null"], "enum": [*candidates, None]},
            "source_type": {"type": "string", "enum": [*SOURCE_TYPES, "unknown"]},
            "source_agent": {"type": ["string", "null"]},
        },
        "required": ["reasoning", "has_knowledge", "value", "source_type", "source_agent"],
    }


def candidates_for(world, prop_id):
    vals = set()
    for st in world.epistemic_states.values():
        for c in st.claims_for(prop_id):
            if c.value is not None:
                vals.add(c.value)
    fact = world.facts[prop_id]
    for v in (fact.true_value, fact.false_value):
        if v:
            vals.add(v)
    return sorted(vals)


_DEV = (
    "You reconstruct ONE NPC's epistemic state from raw world facts. You are given the events "
    "(each with who observed it), the information channels (who relayed info to whom), and the "
    "NPC's own raw memory texts. First think in `reasoning`. Then output: "
    "has_knowledge (does this NPC have any basis to answer the proposition); "
    "value (MUST be EXACTLY one of the listed possible_values, copied verbatim — this is what THIS "
    "NPC believes the answer is; null if has_knowledge is false); "
    "source_type (HOW this NPC knows, from the allowed list: direct_observation if it personally "
    "observed the event, testimony if a specific named person told it, hearsay if it heard a "
    "rumor/crowd, environmental_evidence or inference if deduced from physical clues, role_knowledge "
    "if its role makes it know, else unknown); source_agent (who/what it got it from). "
    "Capture what THIS NPC actually believes from ITS OWN memory and observations — even if that "
    "contradicts what other events reveal. Do NOT substitute the globally-true answer for the NPC's "
    "actual belief (an NPC who only heard a false rumor believes the rumor). If memories conflict, "
    "pick the single most authoritative source. Return exactly one JSON object, no Markdown."
)


def build_extract_messages(world, npc_id, prop_id, candidates):
    npc = world.npc(npc_id)
    fact = world.facts[prop_id]
    gold = world.epistemic_states[npc_id]
    # raw memory: the NL text the NPC holds — NO source_type label, NO value label
    memories = [
        {"memory_text": c.proposition, "received_from": c.source_agent, "time": c.time_obtained}
        for c in gold.claims_for(prop_id)
    ]
    payload = {
        "target_npc": {"id": npc.npc_id, "role": npc.role, "location": npc.location},
        "proposition": fact.proposition,
        "possible_values": candidates,
        "events": [
            {"description": e.description, "location": e.location,
             "participants": e.participants, "observers": e.observers}
            for e in world.events
        ],
        "information_channels": [
            {"from": g["source"], "to": g["target"]} for g in world.source_graph
        ],
        "npc_raw_memories": memories,
        "allowed_source_types": SOURCE_TYPES,
    }
    return [
        {"role": "developer", "content": _DEV},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=True, indent=2)},
    ]


def extract_state(world, npc_id, prop_id, client):
    cands = candidates_for(world, prop_id)
    parsed = client.complete_json(
        build_extract_messages(world, npc_id, prop_id, cands), response_schema=make_schema(cands)
    )
    es = EpistemicState(npc_id=npc_id)
    st, val = parsed.get("source_type"), parsed.get("value")
    if parsed.get("has_knowledge") and st in SOURCE_TYPES and val:
        rel = 0.85 if st in STRONG else 0.4
        es.add_claim(Claim(
            proposition_id=prop_id, proposition=world.facts[prop_id].proposition, value=val,
            source_type=st, source_agent=parsed.get("source_agent"), time_obtained=1,
            reliability=rel, verification_status="verified" if rel >= 0.7 else "unverified",
        ))
    return es, parsed


def main():
    extractor = OpenAICompatibleChatClient(model=EXTRACTOR_MODEL, base_url=EXTRACTOR_BASE,
                                           api_key=OPENAI_KEY, temperature=0.0, use_structured_output=True)
    ctrl_client = OpenAICompatibleChatClient(model=CTRL_MODEL, base_url=CTRL_BASE,
                                             api_key=OPENAI_KEY, temperature=0.0, use_structured_output=True)
    ctrl = StructuredSourceCalibratedController(client=ctrl_client, name="structured")
    pres_ok = both = src_ok = val_ok = total = 0
    rows = defaultdict(list)
    examples = []
    for seed in SEEDS:
        worlds, queries = generate_dataset(WORLDS, QPW, seed)
        by = {w.world_id: w for w in worlds}
        for q in queries:
            gw = by[q.world_id]
            es, parsed = extract_state(gw, q.target_npc_id, q.proposition_id, extractor)
            gold_best = gw.epistemic_states[q.target_npc_id].best_claim(q.proposition_id)
            ext_best = es.best_claim(q.proposition_id)
            total += 1
            gh, eh = gold_best is not None, ext_best is not None
            pres_ok += int(gh == eh)
            if gh and eh:
                both += 1
                src_ok += int(ext_best.source_type == gold_best.source_type)
                val_ok += int(_answer_matches(gold_best.value, ext_best.value))
            if len(examples) < 8:
                examples.append((q.scenario_type, q.target_npc_id,
                                 (gold_best.source_type, gold_best.value) if gh else None,
                                 (ext_best.source_type, ext_best.value) if eh else None))
            ew = dataclasses.replace(gw, epistemic_states={**gw.epistemic_states, q.target_npc_id: es})
            for label, world in (("gold", gw), ("extracted", ew)):
                rec = evaluate_response(gw, q, ctrl.respond(world, q))  # always judged vs gold
                for m, v in rec.metrics.items():
                    if v is not None:
                        rows[(label, m)].append(v)
        print(f"seed {seed} done", flush=True)

    print(f"\n=== EXTRACTION ACCURACY (extractor={EXTRACTOR_MODEL}, ctrl={CTRL_MODEL}, n={total}) ===")
    print(f"  presence (has/no-knowledge correct) = {pres_ok/total:.2f}")
    print(f"  source_type correct (both have claim, n={both}) = {src_ok/max(1,both):.2f}")
    print(f"  value correct (both have claim) = {val_ok/max(1,both):.2f}")

    def mean(label, m):
        v = rows.get((label, m), [])
        return statistics.mean(v) if v else float("nan")

    print("\n=== DOWNSTREAM: structured on GOLD state vs EXTRACTED state (judged vs ground truth) ===")
    print(f"  {'metric':30s}{'gold':>8}{'extracted':>11}{'Δ':>8}")
    for m in ["epistemic_consistency", "warranted_answer_accuracy", "omniscience_leakage",
              "source_attribution_accuracy", "false_witness"]:
        g, e = mean("gold", m), mean("extracted", m)
        print(f"  {m:30s}{g:>8.2f}{e:>11.2f}{e-g:>+8.2f}")

    print("\nexamples (scenario | npc | gold(src,val) | extracted(src,val)):")
    for sc, npc, gold, ext in examples:
        print(f"  {sc:22s} {npc:10s} {str(gold):28s} -> {ext}")
    cost = (extractor.input_tokens / 1e6 * 0.40 + extractor.output_tokens / 1e6 * 1.60) if "openai" in EXTRACTOR_BASE else 0.0
    print("CURVE_JSON " + json.dumps({
        "extractor": EXTRACTOR_MODEL, "ctrl": CTRL_MODEL, "n": total,
        "src_acc": round(src_ok / max(1, both), 3), "presence": round(pres_ok / total, 3),
        "val_acc": round(val_ok / max(1, both), 3),
        "EC_gold": round(mean("gold", "epistemic_consistency"), 3),
        "EC_ext": round(mean("extracted", "epistemic_consistency"), 3),
        "OLR_ext": round(mean("extracted", "omniscience_leakage"), 3),
        "SAA_ext": round(mean("extracted", "source_attribution_accuracy"), 3),
        "FWR_ext": round(mean("extracted", "false_witness"), 3),
        "extractor_openai_cost": round(cost, 3),
    }))


if __name__ == "__main__":
    main()
