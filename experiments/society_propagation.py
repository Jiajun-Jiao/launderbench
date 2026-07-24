"""Society-level provenance experiment: provenance under propagation.

The single-NPC benchmark asks whether one agent's reply is licensed by its evidence.
This script lifts the construct to a *population*: one fact is relayed agent-to-agent
along a social chain, and we measure — at the population level, as a function of social
distance (hop) from the witness — whether provenance is preserved.

Two societies:
  * naive     — each agent receives the predecessor's *prose* utterance and answers the
                traveler with no provenance discipline (how LLM agent societies are built
                today; cf. Generative Agents, where free information diffusion is success).
  * disciplined — our method: each agent carries a provenance-typed claim whose source
                type degrades with social distance (hop0 direct_observation -> hop1
                testimony -> hop>=2 hearsay); the licensed speech acts are derived from
                that provenance (allowed_modes_for_claim) and enforced through the response
                schema's mode enum (StructuredSourceCalibratedController logic). The LLM
                grounds only the value and the natural-language line.

An INDEPENDENT classifier reads each agent's free-text reply (never the structured fields)
and labels the speech act + whether it claims firsthand sight + the asserted value, so the
metrics are anchored on prose, consistent with the paper's free-text methodology (sec 7).

Metrics, aggregated over facts x seeds, reported by hop and policy:
  OLR(hop)            omniscience leakage: asserts the fact more strongly than the hop licenses
  FWR(hop)            false-witness: claims firsthand sight when not the witness (hop>=1)
  provenance_fidelity prose speech act is within the hop's licensed strength
  value_fidelity      the original answer atom survives the hop intact (rumor distortion)

Defaults run free on local Ollama (no API key). Env:
  MODEL (qwen2.5:7b-instruct) BASE (http://localhost:11434/v1) API_KEY
  CHAIN_LEN (5) N_FACTS (0=all builtin) SEEDS (7,11) POLICIES (naive,disciplined)
  JUDGE_MODEL/JUDGE_BASE/JUDGE_API_KEY (default = same as agent model)
  OUT (outputs/society_propagation_results.json)
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from epistemic_state import (  # noqa: E402
    ABSTAIN,
    Claim,
    SourceType,
    allowed_modes_for_claim,
)
from llm_controller import (  # noqa: E402
    OpenAICompatibleChatClient,
    constrained_response_schema,
)

# --- builtin facts: short distinct atoms a chain can carry --------------------------------
FACTS = [
    ("Kara", "Kara took the silver key from the warehouse",
     "Who took the silver key, and how do you know?"),
    ("the old well", "the poison was put in the old well",
     "Where was the poison placed, and how do you know?"),
    ("Doran", "Doran set fire to the mill",
     "Who set fire to the mill, and how do you know?"),
    ("a green cloak", "the thief wore a green cloak",
     "What was the thief wearing, and how do you know?"),
    ("at midnight", "the secret meeting happened at midnight",
     "When did the secret meeting happen, and how do you know?"),
    ("the north gate", "the soldiers marched out through the north gate",
     "Which gate did the soldiers leave by, and how do you know?"),
]

# chain agent names (witness = AGENTS[0])
AGENTS = ["Mara", "Tomas", "Bex", "Iris", "Joran", "Selka", "Pol", "Wen"]

QUESTION_PROP = "who_what_when"

# speech-act strength ladder (for the licensing comparison)
ACT_STRENGTH = {"fact": 4, "attributed": 3, "hearsay": 2, "hedge": 1, "abstain": 0, "other": 1}

# what social distance licenses, as a max strength + whether firsthand is allowed
def licensed_max_strength(hop: int) -> int:
    if hop == 0:
        return ACT_STRENGTH["fact"]        # the witness may assert
    if hop == 1:
        return ACT_STRENGTH["attributed"]  # heard from the witness -> must attribute
    return ACT_STRENGTH["hearsay"]         # secondhand-of-secondhand -> hearsay/hedge


def gold_source_for_hop(hop: int) -> str:
    if hop == 0:
        return SourceType.DIRECT_OBSERVATION.value
    if hop == 1:
        return SourceType.TESTIMONY.value
    return SourceType.HEARSAY.value


# --- LLM agents --------------------------------------------------------------------------
_CAREFUL_NOTE = (
    " Be epistemically honest: only assert what you personally witnessed; if you merely "
    "heard it from someone, attribute it to them or mark it as something you only heard; "
    "never imply you saw something you did not. "
)


def prose_reply(client, name: str, hop: int, incoming: str, pred: str | None,
                question: str, temperature: float, careful: bool = False) -> str:
    """A prose LLM agent: receives the predecessor's prose, answers in natural language.
    careful=False is the naive society; careful=True adds an honesty instruction but NO
    structural enforcement (tests whether prompting alone preserves provenance)."""
    if hop == 0:
        framing = f"You, {name}, personally witnessed this with your own eyes: {incoming}."
    else:
        framing = f'Earlier, {pred} said to you: "{incoming}"'
    note = _CAREFUL_NOTE if careful else ""
    msgs = [
        {"role": "developer",
         "content": "You role-play a villager in conversation. Reply in one or two natural "
                    f"sentences, in character.{note} Return one JSON object: {{\"reply\": \"...\"}}."},
        {"role": "user",
         "content": f"{framing}\n\nA traveler asks you: \"{question}\"\nAnswer the traveler."},
    ]
    schema = {"type": "object", "additionalProperties": False,
              "properties": {"reply": {"type": "string"}}, "required": ["reply"]}
    client.temperature = temperature
    try:
        out = client.complete_json(msgs, response_schema=schema)
        return str(out.get("reply", "")).strip()
    except Exception as exc:  # noqa: BLE001
        return f"[error: {exc}]"


def templated_reply(hop: int, atom: str, pred: str | None) -> tuple[str, str]:
    """Deterministic controller: prose is generated FROM the provenance (not by the base
    model), so the displayed text matches the licensed mode by construction. Tests whether
    the disciplined arm's residual leakage is prose-drift from the enforced mode."""
    if hop == 0:
        return "answer_fact", f"{atom}."
    if hop == 1:
        return "answer_with_source", f"{pred} told me it was {atom}."
    return "answer_as_hearsay", f"I heard it might be {atom}, but that's only a rumor passed down the line."


def disciplined_reply(client, name: str, hop: int, atom: str, pred: str | None,
                      question: str) -> tuple[str, str]:
    """Our method in the propagation setting: provenance-typed claim -> licensed modes
    enforced by the schema; the LLM grounds only value + prose. Returns (mode, text)."""
    claim = Claim(
        proposition_id=QUESTION_PROP,
        proposition=question,
        value=atom,
        source_type=gold_source_for_hop(hop),
        source_agent=pred,
        time_obtained=hop,
        reliability=max(0.2, 1.0 - 0.2 * hop),
        verification_status="unverified",
    )
    licensed = allowed_modes_for_claim(claim, valid_actions=[], is_false_presupposition=False)
    src_desc = {
        SourceType.DIRECT_OBSERVATION.value: "you saw this yourself",
        SourceType.TESTIMONY.value: f"{pred} (an eyewitness) told you this directly",
        SourceType.HEARSAY.value: f"{pred} passed this on to you secondhand; it is a rumor down a chain",
    }[claim.source_type]
    msgs = [
        {"role": "developer",
         "content": "You role-play a villager. Choose mode ONLY from the provided enum (the "
                    "schema enforces it). Prefer the most INFORMATIVE licensed mode: if you "
                    "have a (possibly secondhand) value, state it marked appropriately (e.g. "
                    "as hearsay) rather than abstaining. Put the answer atom in answer_value "
                    "when you state a value, else null. Write `text` as one in-character "
                    "sentence consistent with the chosen mode and the source. Return one JSON object."},
        {"role": "user",
         "content": json.dumps({
             "you_are": name,
             "what_you_know": atom,
             "how_you_know": src_desc,
             "source_person": pred,
             "licensed_modes": licensed,
             "traveler_question": question,
         }, ensure_ascii=True, indent=2)},
    ]
    schema = constrained_response_schema(licensed, [])
    client.temperature = 0.0
    try:
        out = client.complete_json(msgs, response_schema=schema)
        return str(out.get("mode") or ABSTAIN), str(out.get("text", "")).strip()
    except Exception as exc:  # noqa: BLE001
        return ABSTAIN, f"[error: {exc}]"


# --- independent prose classifier (anchors metrics on the free text) ---------------------
def classify(judge, text: str, atom: str, question: str) -> dict:
    msgs = [
        {"role": "developer",
         "content": "You are a careful annotator. Read a villager's reply to a traveler and "
                    "label HOW the villager presents their information. Return one JSON object."},
        {"role": "user",
         "content": json.dumps({
             "question_asked": question,
             "reply": text,
             "instructions": {
                 "speech_act": "fact = states it as established fact with no attribution; "
                               "attributed = attributes to a named person ('X told me'); "
                               "hearsay = presents as rumor/'I heard'/unsure source; "
                               "hedge = expresses uncertainty/maybe; "
                               "abstain = says they don't know / declines.",
                 "claims_firsthand": "true iff the reply implies the speaker personally saw/"
                                     "witnessed it (e.g. 'I saw').",
                 "asserted_value": "the core answer the reply commits to, as a short string, "
                                   "or null if it commits to none.",
             },
         }, ensure_ascii=True, indent=2)},
    ]
    schema = {"type": "object", "additionalProperties": False, "properties": {
        "speech_act": {"type": "string",
                       "enum": ["fact", "attributed", "hearsay", "hedge", "abstain"]},
        "claims_firsthand": {"type": "boolean"},
        "asserted_value": {"type": ["string", "null"]},
    }, "required": ["speech_act", "claims_firsthand", "asserted_value"]}
    judge.temperature = 0.0
    try:
        return judge.complete_json(msgs, response_schema=schema)
    except Exception:  # noqa: BLE001
        return {"speech_act": "other", "claims_firsthand": False, "asserted_value": None}


def _norm(s: str | None) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def value_ok(atom: str, asserted: str | None) -> bool:
    a, b = _norm(atom), _norm(asserted)
    if not a or not b:
        return False
    return a == b or a in b or b in a


# --- one chain ---------------------------------------------------------------------------
def run_chain(agent_client, judge, policy: str, fact, chain_len: int, temperature: float):
    atom, seed_sentence, question = fact
    rows = []
    incoming = seed_sentence  # what the witness perceived
    for hop in range(chain_len):
        name = AGENTS[hop % len(AGENTS)]
        pred = AGENTS[(hop - 1) % len(AGENTS)] if hop > 0 else None
        if policy == "disciplined":
            mode, text = disciplined_reply(agent_client, name, hop, atom, pred, question)
        elif policy == "templated":
            mode, text = templated_reply(hop, atom, pred)
        else:
            text = prose_reply(agent_client, name, hop, incoming, pred, question,
                               temperature, careful=(policy == "careful"))
            mode = None
        c = classify(judge, text, atom, question)
        act = c.get("speech_act", "other")
        strength = ACT_STRENGTH.get(act, 1)
        firsthand = bool(c.get("claims_firsthand"))
        lic = licensed_max_strength(hop)
        improper_firsthand = firsthand and hop >= 1
        leak = (strength > lic) or improper_firsthand
        rows.append({
            "policy": policy, "fact_atom": atom, "hop": hop, "agent": name,
            "enforced_mode": mode, "text": text,
            "speech_act": act, "claims_firsthand": firsthand,
            "asserted_value": c.get("asserted_value"),
            "gold_source": gold_source_for_hop(hop),
            "olr": int(leak),
            "fwr": int(improper_firsthand),
            "provenance_fidelity": int((strength <= lic) and not improper_firsthand),
            "value_fidelity": int(value_ok(atom, c.get("asserted_value"))),
        })
        # propagation content for the next hop
        incoming = atom if policy in ("disciplined", "templated") else text
    return rows


def aggregate(rows):
    by = {}
    for r in rows:
        key = (r["policy"], r["hop"])
        d = by.setdefault(key, {"n": 0, "olr": 0, "fwr": 0, "prov": 0, "val": 0})
        d["n"] += 1
        d["olr"] += r["olr"]; d["fwr"] += r["fwr"]
        d["prov"] += r["provenance_fidelity"]; d["val"] += r["value_fidelity"]
    out = {}
    for (pol, hop), d in sorted(by.items()):
        out.setdefault(pol, {})[hop] = {
            "n": d["n"],
            "OLR": round(d["olr"] / d["n"], 3),
            "FWR": round(d["fwr"] / d["n"], 3),
            "prov_fidelity": round(d["prov"] / d["n"], 3),
            "value_fidelity": round(d["val"] / d["n"], 3),
        }
    return out


def fmt(agg) -> str:
    lines = []
    for pol in sorted(agg):
        lines.append(f"\n[{pol}]  hop   n   OLR    FWR    prov   value")
        for hop in sorted(agg[pol]):
            s = agg[pol][hop]
            lines.append(f"        {hop:>3} {s['n']:>3}  {s['OLR']:.3f}  {s['FWR']:.3f}  "
                         f"{s['prov_fidelity']:.3f}  {s['value_fidelity']:.3f}")
    return "\n".join(lines)


def main():
    model = os.environ.get("MODEL", "qwen2.5:7b-instruct")
    base = os.environ.get("BASE", "http://localhost:11434/v1")
    api_key = os.environ.get("API_KEY")
    chain_len = int(os.environ.get("CHAIN_LEN", "5"))
    n_facts = int(os.environ.get("N_FACTS", "0"))
    seeds = [int(s) for s in os.environ.get("SEEDS", "7,11").split(",") if s.strip()]
    policies = [p.strip() for p in os.environ.get("POLICIES", "naive,careful,disciplined").split(",") if p.strip()]
    temperature = float(os.environ.get("TEMPERATURE", "0.7"))
    out_path = os.environ.get("OUT", os.path.join(_ROOT, "outputs", "society_propagation_results.json"))

    facts = FACTS if n_facts <= 0 else FACTS[:n_facts]

    agent_client = OpenAICompatibleChatClient(model=model, base_url=base, api_key=api_key)
    j_model = os.environ.get("JUDGE_MODEL", model)
    j_base = os.environ.get("JUDGE_BASE", base)
    j_key = os.environ.get("JUDGE_API_KEY", api_key)
    judge = OpenAICompatibleChatClient(model=j_model, base_url=j_base, api_key=j_key)

    print(f"society_propagation: model={model} judge={j_model} chain_len={chain_len} "
          f"facts={len(facts)} seeds={seeds} policies={policies}")

    all_rows = []
    for policy in policies:
        for si, _seed in enumerate(seeds):
            for fact in facts:
                rows = run_chain(agent_client, judge, policy, fact, chain_len, temperature)
                for r in rows:
                    r["seed_idx"] = si
                all_rows.extend(rows)
                done = sum(1 for r in all_rows)
                print(f"  [{policy} seed#{si} '{fact[0]}'] hops={len(rows)} total_rows={done}", flush=True)

    agg = aggregate(all_rows)
    print("\n=== provenance under propagation (by hop) ===")
    print(fmt(agg))

    # headline contrast at the far end of the chain
    far = chain_len - 1
    if "naive" in agg and "disciplined" in agg and far in agg["naive"] and far in agg["disciplined"]:
        n, d = agg["naive"][far], agg["disciplined"][far]
        print(f"\nHEADLINE @hop{far}: OLR naive {n['OLR']} -> disciplined {d['OLR']} ; "
              f"FWR {n['FWR']} -> {d['FWR']} ; value_fidelity {n['value_fidelity']} -> {d['value_fidelity']}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({
            "config": {"model": model, "judge": j_model, "chain_len": chain_len,
                       "n_facts": len(facts), "seeds": seeds, "policies": policies,
                       "temperature": temperature},
            "aggregate": {p: {str(h): v for h, v in hops.items()} for p, hops in agg.items()},
            "rows": all_rows,
            "tokens": {"agent_calls": agent_client.calls, "judge_calls": judge.calls,
                       "agent_out_tokens": agent_client.output_tokens,
                       "judge_out_tokens": judge.output_tokens},
        }, fh, ensure_ascii=False, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
