"""Experiment 2: ecological validity. An LLM authors diverse, self-consistent scenarios
(varied themes, NPC counts, NL event/memory/question text); we strictly VALIDATE them into
the benchmark's structured gold schema, then check the main conclusions hold on this
non-templated data. Runs free on local Ollama."""
import json
import os
import sys

sys.path.insert(0, "/path/to/launderbench")
from random import Random

from epistemic_state import (
    CORRECT_FALSE_PRESUPPOSITION,
    Claim,
    EpistemicState,
    SourceType,
    allowed_modes_for_claim,
    unique_modes,
)
from world_generator import NPC, Event, Fact, World
from query_generator import QueryItem
from baselines import default_controllers
from llm_controller import (
    OpenAICompatibleChatClient,
    StructuredSourceCalibratedController,
    make_llm_controllers,
)
from evaluator import aggregate_results, evaluate_all, format_summary_table

SRC = [s.value for s in SourceType if s.value != "forbidden_global_state"]
STRONG = {"direct_observation", "role_knowledge", "public_record", "episodic_memory", "testimony"}
DOMAINS = ["a market theft", "a workshop sabotage", "a missing courier", "a tavern poisoning rumor",
           "a guild contract forgery", "a castle vault secret", "a harbor smuggling tip",
           "a temple relic disappearance"]

AUTHOR_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "scenario_label": {"type": "string"},
        "proposition": {"type": "string"},
        "true_value": {"type": "string"},
        "false_value": {"type": ["string", "null"]},
        "is_false_presupposition": {"type": "boolean"},
        "npcs": {"type": "array",
            "items": {"type": "object", "additionalProperties": False,
            "properties": {"id": {"type": "string"}, "name": {"type": "string"},
                           "role": {"type": "string"}, "location": {"type": "string"}},
            "required": ["id", "name", "role", "location"]}},
        "events": {"type": "array", "items": {"type": "object", "additionalProperties": False,
            "properties": {"description": {"type": "string"}, "location": {"type": "string"},
                           "participants": {"type": "array", "items": {"type": "string"}},
                           "observers": {"type": "array", "items": {"type": "string"}}},
            "required": ["description", "location", "participants", "observers"]}},
        "claims": {"type": "array", "items": {"type": "object", "additionalProperties": False,
            "properties": {"npc": {"type": "string"}, "value": {"type": "string"},
                           "source_type": {"type": "string", "enum": SRC},
                           "source_agent": {"type": ["string", "null"]},
                           "memory_text": {"type": "string"}},
            "required": ["npc", "value", "source_type", "source_agent", "memory_text"]}},
    },
    "required": ["scenario_label", "proposition", "true_value", "false_value",
                 "is_false_presupposition", "npcs", "events", "claims"],
}

_DEV = (
    "You are a scenario author for an epistemic-consistency benchmark for game NPCs. Invent a "
    "SMALL, CONCRETE, SELF-CONSISTENT 'who-did-what' mystery whose answer is a SHORT ATOM.\n"
    "- The hidden fact must be answerable by ONE short atom: a person's NAME, or yes/no, or a "
    "code/number. Set true_value to that atom (e.g. 'Mara', 'yes', '4827'); false_value is a "
    "plausible WRONG atom of the same kind. NEVER use a sentence as a value.\n"
    "- EXACTLY 5 to 7 NPCs (no fewer than 5), with distinct roles and locations.\n"
    "- ONE concrete event; in `observers` list EXACTLY the NPCs who personally witnessed it.\n"
    "- Then how the fact reached the others, using AT LEAST THREE different source_type values "
    "across NPCs: the eyewitness -> direct_observation; someone a named witness told -> testimony; "
    "someone who heard a market/crowd rumor -> hearsay (give THAT one a DISTORTED/WRONG value); "
    "someone who saw physical clues -> environmental_evidence or inference; an official whose job it "
    "is -> role_knowledge. Leave 1-2 NPCs with NO claim (omit them from claims).\n"
    "- Each claim: `value` is the SHORT ATOM that NPC believes (the true atom, or the distorted atom "
    "for the rumor); `memory_text` is the natural sentence the NPC remembers (it must imply the value "
    "but must NOT name the source_type); reference the NPC by its `id`.\n"
    "Vary the setting each time. Return exactly one JSON object."
)


def author_world(client, idx):
    dev = _DEV + f"\n\nThis scenario's setting hint: {DOMAINS[idx % len(DOMAINS)]} (#{idx})."
    user = "Author one scenario now as JSON."
    return client.complete_json([{"role": "developer", "content": dev},
                                 {"role": "user", "content": user}], response_schema=AUTHOR_SCHEMA)


def validate_and_build(raw, world_id):
    try:
        npcs = [NPC(npc_id=n["id"], name=n["name"], role=n["role"], location=n["location"]) for n in raw["npcs"]]
        ids = {n.npc_id for n in npcs}
        if len(npcs) < 4 or len(ids) != len(npcs):
            return None, "npc count/uniqueness"
        ref = {}
        for n in raw["npcs"]:
            for k in (n["id"], n["name"], n["role"]):
                if k:
                    ref[str(k).strip().lower()] = n["id"]

        def resolve(x):
            return ref.get(str(x).strip().lower()) if x is not None else None

        prop_id = raw["scenario_label"] or "fact"
        observers = set()
        events = []
        for i, e in enumerate(raw["events"]):
            obs = [r for r in (resolve(o) for o in e["observers"]) if r]
            observers |= set(obs)
            events.append(Event(event_id=f"{world_id}_e{i}", time=1, location=e["location"],
                                description=e["description"],
                                participants=[r for r in (resolve(p) for p in e["participants"]) if r],
                                observers=obs, hidden_truth={prop_id: raw["true_value"]}))
        claims = raw["claims"]
        if not claims:
            return None, "no claims"
        if len(claims) >= len(npcs):
            return None, "no NPC left without knowledge"

        def atom_ok(v):
            return isinstance(v, str) and 0 < len(v) <= 24 and len(v.split()) <= 3
        if not atom_ok(raw["true_value"]):
            return None, "non-atomic true_value"
        if not all(atom_ok(c.get("value")) for c in claims):
            return None, "non-atomic claim value"
        for c in claims:
            nid = resolve(c["npc"])
            if not nid:
                return None, f"claim for unknown npc {c['npc']}"
            c["npc"] = nid
            # repair: direct_observation is only legitimate for an actual observer; else downgrade
            if c["source_type"] == "direct_observation" and nid not in observers:
                c["source_type"] = "inference"
        if len({c["source_type"] for c in claims}) < 2:
            return None, "needs >=2 distinct source types"
        states = {n.npc_id: EpistemicState(npc_id=n.npc_id) for n in npcs}
        for c in claims:
            rel = 0.9 if c["source_type"] in STRONG else 0.35
            states[c["npc"]].add_claim(Claim(
                proposition_id=prop_id, proposition=c["memory_text"], value=c["value"],
                source_type=c["source_type"], source_agent=c.get("source_agent"), time_obtained=2,
                reliability=rel, verification_status="verified" if rel >= 0.7 else "unverified"))
        source_graph = [{"source": c.get("source_agent") or "?", "target": c["npc"],
                         "proposition_id": prop_id, "channel": c["source_type"]}
                        for c in claims if c["source_type"] != "direct_observation"]
        an_npc = npcs[0].npc_id
        fact = Fact(proposition_id=prop_id, proposition=raw["proposition"], true_value=raw["true_value"],
                    answer_text=raw["true_value"], hidden=True,
                    suggested_actions=[f"ask_npc:{an_npc}", f"inspect_location:{events[0].location if events else 'site'}",
                                       "query_public_record:registry"],
                    false_value=raw.get("false_value"),
                    metadata={"authored": True, "is_false_presupposition": raw["is_false_presupposition"]})
        world = World(world_id=world_id, scenario_type="authored", locations=sorted({n.location for n in npcs}),
                      npcs=npcs, events=events, facts={prop_id: fact}, epistemic_states=states, source_graph=source_graph)
        return world, "ok"
    except Exception as exc:  # noqa: BLE001
        return None, f"exception: {exc}"


def build_queries(world):
    fact = world.primary_fact()
    is_fp = bool(fact.metadata.get("is_false_presupposition"))
    qs = []
    for i, npc in enumerate(world.npcs):
        best = world.epistemic_states[npc.npc_id].best_claim(fact.proposition_id)
        modes = allowed_modes_for_claim(best, valid_actions=fact.suggested_actions, is_false_presupposition=is_fp)
        if is_fp and best and best.value == "no":
            modes = unique_modes([CORRECT_FALSE_PRESUPPOSITION, *modes])
        qs.append(QueryItem(
            query_id=f"{world.world_id}_q{i}", world_id=world.world_id, scenario_type="authored",
            target_npc_id=npc.npc_id, question=f"About this: {fact.proposition} What do you actually know, and how?",
            proposition_id=fact.proposition_id, true_value=fact.true_value,
            expected_local_value=best.value if best else None, acceptable_modes=modes,
            valid_actions=fact.suggested_actions, is_false_presupposition=is_fp,
            presupposed_value=fact.false_value if is_fp else None,
            metadata={"target_role": npc.role, "local_claim": best.to_dict() if best else None}))
    return qs


def main():
    n = int(os.environ.get("N", "3"))
    a_model = os.environ.get("MODEL", "gpt-4.1-mini")          # author (paid, cheap)
    a_base = os.environ.get("BASE", "https://api.openai.com/v1")
    api_key = os.environ.get("OPENAI_API_KEY")
    author = OpenAICompatibleChatClient(model=a_model, base_url=a_base, api_key=api_key,
                                        temperature=0.5, use_structured_output=True)
    worlds, queries, raws, attempts, fails = [], [], [], 0, []
    rng_idx = 0
    while len(worlds) < n and attempts < n * 3:
        attempts += 1
        try:
            raw = author_world(author, rng_idx); rng_idx += 1
            w, msg = validate_and_build(raw, f"eco_{len(worlds):03d}")
            if w is None:
                fails.append(msg); continue
            worlds.append(w); queries.extend(build_queries(w)); raws.append(raw)
        except Exception as exc:  # noqa: BLE001
            fails.append(f"author err: {str(exc)[:80]}")
    print(f"authored {len(worlds)} valid worlds in {attempts} attempts ({a_model}); rejects={fails[:6]}")
    if not worlds:
        return
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/eco_worlds.json", "w") as fh:
        json.dump(raws, fh, ensure_ascii=False, indent=1)
    print(f"saved {len(raws)} authored worlds -> outputs/eco_worlds.json")
    for w in worlds:
        f = w.primary_fact()
        srcs = sorted({(w.epistemic_states[npc.npc_id].best_claim(f.proposition_id) or _N).source_type
                       for npc in w.npcs} - {None})
        print(f"  [{w.world_id}] {f.proposition[:60]!r} true={f.true_value!r} npcs={len(w.npcs)} src={srcs}")

    print(f"\n=== deterministic baselines on {len(worlds)} AUTHORED worlds, {len(queries)} queries ===")
    print(format_summary_table(aggregate_results(evaluate_all(worlds, queries, default_controllers()))))
    print("\n(LLM controllers: run eco_run_llm.py on outputs/eco_worlds.json, local & free)")


class _N:
    source_type = None


if __name__ == "__main__":
    main()
