"""HARD narrative worlds — to test the hypothesis that our decoupling method's marginal value was
hidden by SHALLOW scenarios (depth-1/2 info flow that strong models read holistically). These worlds
have DEEP multi-hop telling chains (>=3 hops), a CONFLICTING wrong-value rumor chain, and 8-12 agents,
so holistic reading should break and structural reconstruction should matter. Author = gpt-4.1 (strong
+ strict-schema-enforced; mini/Claude are weaker/looser for complex consistent worlds).

Reuses the same gold schema + validator as the shallow set, so the sweep is directly comparable.
Reports complexity stats (chain depth / agents / distinct believed values) for the hard set vs the
shallow set, to CONFIRM the worlds are actually harder. Env: N(18), MODEL(gpt-4.1).
"""
import collections
import copy
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eco_worlds import AUTHOR_SCHEMA, DOMAINS, build_queries, validate_and_build  # noqa: F401
from llm_controller import OpenAICompatibleChatClient

N = int(os.environ.get("N", "18"))
MODEL = os.environ.get("MODEL", "gpt-4.1")
BASE = os.environ.get("BASE", "https://api.openai.com/v1")
PATH = os.environ.get("HARD_PATH", os.path.join(_ROOT, "outputs", "eco_narrative_hard.jsonl"))
SHALLOW = os.path.join(_ROOT, "outputs", "eco_narrative.jsonl")
MIN_HOPS = int(os.environ.get("MIN_HOPS", "3"))
NPC_MIN = int(os.environ.get("NPC_MIN", "8"))
NPC_MAX = int(os.environ.get("NPC_MAX", "12"))
N_CONF = int(os.environ.get("N_CONF", "1"))

NARR_SCHEMA = copy.deepcopy(AUTHOR_SCHEMA)
NARR_SCHEMA["properties"]["narrative"] = {"type": "string"}
NARR_SCHEMA["required"] = [*AUTHOR_SCHEMA["required"], "narrative"]

_HARD_DEV = (
    "You are a scenario author for an epistemic-consistency benchmark for game NPCs. Invent a COMPLEX, "
    "SELF-CONSISTENT 'who-did-what' mystery whose answer is a SHORT ATOM.\n"
    "- The hidden fact is answerable by ONE short atom: a NAME, yes/no, or a code/number. true_value = "
    "that atom; false_value = a plausible WRONG atom of the same kind. NEVER use a sentence as a value.\n"
    "- 8 to 12 NPCs, with distinct roles and locations.\n"
    "- ONE concrete event; in `observers` list EXACTLY the 1-2 NPCs who personally witnessed it.\n"
    "- Build a DEEP information chain about the fact, AT LEAST 3 HOPS long: the eyewitness -> "
    "direct_observation (value = true_value); someone the eyewitness told DIRECTLY -> testimony "
    "(source_agent = the eyewitness, value = true_value); then people told by a NON-witness (2+ hops "
    "from the event) -> hearsay (source_agent = the person who actually told them). The value MAY "
    "distort as it travels down the chain. Make the chain FULLY TRACEABLE: each non-eyewitness's "
    "source_agent must NAME the specific person who told THEM, and that teller must ALSO appear as a "
    "claim (except the original eyewitness) — so the whole multi-hop chain is followable hop-by-hop.\n"
    "- ALSO build a SEPARATE, CONFLICTING rumor chain that spreads the WRONG atom (false_value) to 2+ "
    "other NPCs -> hearsay carrying the false_value. So different NPCs end up believing DIFFERENT atoms.\n"
    "- Optionally: someone who saw physical clues -> environmental_evidence or inference; an official "
    "whose job it is -> role_knowledge.\n"
    "- Leave 1-2 NPCs with NO claim (omit them from `claims`). Use AT LEAST 3 distinct source_type "
    "values across NPCs.\n"
    "- Each claim: `value` is the SHORT ATOM that NPC believes; `source_agent` is the NAME of whoever "
    "told them (or null for the eyewitness / for role_knowledge); `memory_text` is the natural sentence "
    "the NPC remembers (it must imply the value but must NOT name the source_type); reference the NPC by "
    "its `id` in `npc`, and people by NAME in source_agent/memory_text.\n"
    "- ALSO write `narrative`: a vivid 6-12 sentence prose paragraph that recounts the event and makes "
    "EXPLICIT, IN ORDER, the FULL information chain BY NAME — who witnessed it, who told whom along the "
    "whole multi-hop chain, AND the conflicting rumor and who it reached. Do NOT use the words "
    "direct_observation/testimony/hearsay/inference. The narrative MUST match the npcs/events/claims "
    "exactly (same who-knows-what-and-how).\n"
    "Vary the setting each time. Return exactly one JSON object."
)


def author(client, idx):
    override = (
        f"\n\nMANDATORY OVERRIDES (override any softer numbers above): use {NPC_MIN}-{NPC_MAX} NPCs; the "
        f"main eyewitness->...->hearer telling chain must be AT LEAST {MIN_HOPS} HOPS deep (the furthest "
        f"informed NPC is {MIN_HOPS}+ retellings from the event, each hop a distinct named person who is "
        f"ALSO a claim); include {N_CONF} SEPARATE conflicting rumor chain(s), each spreading a WRONG "
        f"atom to 2+ NPCs. Keep it self-consistent and fully traceable through source_agent links.")
    dev = _HARD_DEV + override + f"\n\nSetting hint: {DOMAINS[idx % len(DOMAINS)]} (#{idx})."
    return client.complete_json([{"role": "developer", "content": dev},
                                 {"role": "user", "content": "Author one COMPLEX scenario (with narrative) now as JSON."}],
                                response_schema=NARR_SCHEMA)


def complexity(raw):
    """max telling-chain depth, #agents, #claims, #distinct believed values (conflict indicator)."""
    npcs, claims = raw.get("npcs", []), raw.get("claims", [])
    ref = {}
    for n in npcs:
        for k in (n.get("id"), n.get("name"), n.get("role")):
            if k:
                ref[str(k).strip().lower()] = n["id"]
    teller = {}
    vals = []
    for c in claims:
        nid = ref.get(str(c.get("npc", "")).strip().lower())
        if not nid:
            continue
        sa = c.get("source_agent")
        teller[nid] = ref.get(str(sa).strip().lower()) if sa else None
        if c.get("value"):
            vals.append(c["value"])

    def depth(nid, seen):
        if nid in seen:
            return 0
        seen = seen | {nid}
        t = teller.get(nid)
        if not t or t not in teller:
            return 1 if t else 0  # told by a non-claimant (eyewitness) = 1; observer = 0
        return 1 + depth(t, seen)

    depths = [depth(n, set()) for n in teller]
    return {"npcs": len(npcs), "claims": len(claims), "max_depth": max(depths) if depths else 0,
            "distinct_values": len(set(v.strip().lower() for v in vals))}


def stats(path, label):
    if not os.path.exists(path):
        print(f"  {label}: (missing {path})")
        return
    raws = [json.loads(l) for l in open(path) if l.strip()]
    cs = [complexity(r) for r in raws]
    if not cs:
        return
    avg = lambda k: round(sum(c[k] for c in cs) / len(cs), 2)  # noqa: E731
    print(f"  {label} (n={len(raws)}): avg_npcs={avg('npcs')} avg_claims={avg('claims')} "
          f"avg_max_depth={avg('max_depth')} avg_distinct_values={avg('distinct_values')} "
          f"| deep(>=3hop)={sum(1 for c in cs if c['max_depth']>=3)} conflict(>=3vals)={sum(1 for c in cs if c['distinct_values']>=3)}")


def main():
    existing = [json.loads(l) for l in open(PATH)] if os.path.exists(PATH) else []
    client = OpenAICompatibleChatClient(model=MODEL, base_url=BASE, api_key=os.environ.get("OPENAI_API_KEY"),
                                        temperature=0.6, use_structured_output=True)
    fh = open(PATH, "a")
    valid, attempts, idx = len(existing), 0, len(existing)
    rej = collections.Counter()
    while valid < N and attempts < (N - len(existing)) * 4 + 12:
        attempts += 1
        try:
            raw = author(client, idx)
            idx += 1
            w, msg = validate_and_build(raw, f"hard_{valid:03d}")
            if w is None:
                rej[msg.split(":")[0]] += 1
                continue
            if not raw.get("narrative") or len(raw["narrative"].split()) < 40:
                rej["narrative too short"] += 1
                continue
            fh.write(json.dumps(raw, ensure_ascii=False) + "\n")
            fh.flush()
            valid += 1
            if valid % 5 == 0:
                print(f"  authored {valid}/{N}", flush=True)
        except Exception as exc:  # noqa: BLE001
            rej[f"err:{str(exc)[:40]}"] += 1
    fh.close()
    ci, co = (2.0, 8.0) if MODEL == "gpt-4.1" else (0.40, 1.60)
    cost = client.input_tokens / 1e6 * ci + client.output_tokens / 1e6 * co
    print(f"\nauthored {valid} HARD worlds in {attempts} attempts ({MODEL}, cost≈${cost:.3f}); rejects={dict(rej)}")
    print("\n=== complexity: hard vs shallow (confirm the hard set is actually deeper) ===")
    stats(PATH, "HARD ")
    stats(SHALLOW, "shallow")
    raws = [json.loads(l) for l in open(PATH) if l.strip()]
    if raws:
        w0 = raws[0]
        print(f"\nsample HARD [{w0['scenario_label']}] true={w0['true_value']!r} false={w0['false_value']!r}")
        print("claims:", [(c["npc"], c["source_type"], c["value"], "<-"+str(c.get("source_agent"))) for c in w0["claims"]])
        print("narrative:", w0["narrative"][:600])


if __name__ == "__main__":
    main()
