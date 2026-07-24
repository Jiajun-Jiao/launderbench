"""Synthetic text-world generation for the E-NPC benchmark MVP."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any

from epistemic_state import (
    Claim,
    EpistemicState,
    SourceType,
    VerificationStatus,
)


NAMES = [
    "Ada",
    "Bram",
    "Cato",
    "Dina",
    "Eli",
    "Faye",
    "Galen",
    "Hana",
    "Iris",
    "Jules",
    "Kara",
    "Lio",
    "Mara",
    "Niko",
    "Orin",
    "Pia",
    "Quinn",
    "Rhea",
    "Soren",
    "Tala",
    "Uma",
    "Vera",
    "Wren",
    "Yara",
]

SCENARIO_TYPES = [
    "theft_mystery",
    "rumor_propagation",
    "private_role_knowledge",
    "false_presupposition",
]


@dataclass(frozen=True)
class NPC:
    npc_id: str
    name: str
    role: str
    location: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "npc_id": self.npc_id,
            "name": self.name,
            "role": self.role,
            "location": self.location,
        }


@dataclass(frozen=True)
class Fact:
    proposition_id: str
    proposition: str
    true_value: str
    answer_text: str
    hidden: bool
    suggested_actions: list[str]
    false_value: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposition_id": self.proposition_id,
            "proposition": self.proposition,
            "true_value": self.true_value,
            "answer_text": self.answer_text,
            "hidden": self.hidden,
            "suggested_actions": self.suggested_actions,
            "false_value": self.false_value,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class Event:
    event_id: str
    time: int
    location: str
    description: str
    participants: list[str]
    observers: list[str]
    hidden_truth: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "time": self.time,
            "location": self.location,
            "description": self.description,
            "participants": self.participants,
            "observers": self.observers,
            "hidden_truth": self.hidden_truth,
        }


@dataclass
class World:
    world_id: str
    scenario_type: str
    locations: list[str]
    npcs: list[NPC]
    events: list[Event]
    facts: dict[str, Fact]
    epistemic_states: dict[str, EpistemicState]
    source_graph: list[dict[str, str]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def npc(self, npc_id: str) -> NPC:
        for npc in self.npcs:
            if npc.npc_id == npc_id:
                return npc
        raise KeyError(npc_id)

    def primary_fact(self) -> Fact:
        return next(iter(self.facts.values()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_id": self.world_id,
            "scenario_type": self.scenario_type,
            "locations": self.locations,
            "npcs": [npc.to_dict() for npc in self.npcs],
            "events": [event.to_dict() for event in self.events],
            "facts": {key: fact.to_dict() for key, fact in self.facts.items()},
            "epistemic_states": {
                npc_id: state.to_dict()
                for npc_id, state in self.epistemic_states.items()
            },
            "source_graph": self.source_graph,
            "metadata": self.metadata,
        }


def generate_worlds(
    num_worlds: int = 50,
    seed: int = 7,
    scenario_types: list[str] | None = None,
) -> list[World]:
    rng = Random(seed)
    types = scenario_types or SCENARIO_TYPES
    builders = {
        "theft_mystery": _generate_theft_world,
        "rumor_propagation": _generate_rumor_world,
        "private_role_knowledge": _generate_private_role_world,
        "false_presupposition": _generate_false_presupposition_world,
        "conflicting_evidence": _generate_conflicting_world,
        "multi_hop_rumor": _generate_multihop_rumor_world,
    }
    worlds: list[World] = []
    for index in range(num_worlds):
        scenario_type = types[index % len(types)]
        world_id = f"world_{index:04d}"
        try:
            builder = builders[scenario_type]
        except KeyError:
            raise ValueError(f"unknown scenario type: {scenario_type}")
        worlds.append(builder(world_id, rng))
    return worlds


def _generate_theft_world(world_id: str, rng: Random) -> World:
    names = rng.sample(NAMES, 5)
    roles = ["thief", "guard", "innkeeper", "blacksmith", "clerk"]
    locations = ["warehouse", "guard_post", "inn", "forge", "market"]
    npcs = _make_npcs(names, roles, locations)
    thief, guard, innkeeper, blacksmith, clerk = npcs
    prop_id = "theft_culprit"
    fact = Fact(
        proposition_id=prop_id,
        proposition=f"{thief.name} stole the silver key from the warehouse.",
        true_value=thief.name,
        answer_text=thief.name,
        hidden=True,
        suggested_actions=[
            f"ask_npc:{guard.npc_id}",
            "inspect_location:warehouse",
            "query_public_record:watch_log",
        ],
        false_value=rng.choice([npc.name for npc in npcs if npc.npc_id != thief.npc_id]),
        metadata={"object": "silver key"},
    )

    states = _empty_states(npcs)
    _add_claim(
        states[thief.npc_id],
        prop_id,
        fact.proposition,
        thief.name,
        SourceType.DIRECT_OBSERVATION,
        thief.npc_id,
        1,
        0.99,
    )
    _add_claim(
        states[guard.npc_id],
        prop_id,
        fact.proposition,
        thief.name,
        SourceType.DIRECT_OBSERVATION,
        guard.npc_id,
        2,
        0.9,
        {"observation": "saw the thief leaving the warehouse with the key"},
    )
    _add_claim(
        states[innkeeper.npc_id],
        prop_id,
        fact.proposition,
        thief.name,
        SourceType.TESTIMONY,
        guard.npc_id,
        3,
        0.75,
    )
    _add_claim(
        states[clerk.npc_id],
        prop_id,
        f"Footprints matching {thief.name}'s boots were near the warehouse lock.",
        thief.name,
        SourceType.ENVIRONMENTAL_EVIDENCE,
        "warehouse_lock",
        4,
        0.55,
    )

    source_graph = [
        _edge(guard.npc_id, innkeeper.npc_id, prop_id, SourceType.TESTIMONY.value),
        _edge("warehouse_lock", clerk.npc_id, prop_id, SourceType.ENVIRONMENTAL_EVIDENCE.value),
    ]
    event = Event(
        event_id=f"{world_id}_event_0",
        time=1,
        location="warehouse",
        description=f"{thief.name} stole the silver key at night.",
        participants=[thief.npc_id],
        observers=[thief.npc_id, guard.npc_id],
        hidden_truth={prop_id: thief.name},
    )
    return World(
        world_id=world_id,
        scenario_type="theft_mystery",
        locations=locations,
        npcs=npcs,
        events=[event],
        facts={prop_id: fact},
        epistemic_states=states,
        source_graph=source_graph,
    )


def _generate_rumor_world(world_id: str, rng: Random) -> World:
    names = rng.sample(NAMES, 5)
    roles = ["healer", "gossip", "guard", "farmer", "mayor"]
    locations = ["well", "clinic", "square", "farm", "council_hall"]
    npcs = _make_npcs(names, roles, locations)
    healer, gossip, guard, farmer, mayor = npcs
    prop_id = "mayor_poisoned_well"
    fact = Fact(
        proposition_id=prop_id,
        proposition=f"{mayor.name} did not poison the village well.",
        true_value="no",
        answer_text="no",
        hidden=True,
        suggested_actions=[
            f"ask_npc:{healer.npc_id}",
            "inspect_location:well",
            "query_public_record:clinic_notes",
        ],
        false_value="yes",
        metadata={"rumor": f"{mayor.name} poisoned the well"},
    )

    states = _empty_states(npcs)
    _add_claim(
        states[healer.npc_id],
        prop_id,
        fact.proposition,
        "no",
        SourceType.DIRECT_OBSERVATION,
        healer.npc_id,
        1,
        0.95,
        {"observation": "found a cracked dye bottle, not poison"},
    )
    _add_claim(
        states[guard.npc_id],
        prop_id,
        fact.proposition,
        "no",
        SourceType.TESTIMONY,
        healer.npc_id,
        2,
        0.8,
    )
    for target in [gossip, farmer]:
        _add_claim(
            states[target.npc_id],
            prop_id,
            f"People in the square say {mayor.name} poisoned the well.",
            "yes",
            SourceType.HEARSAY,
            "square_crowd",
            3,
            0.35,
            {"rumor": True},
        )
    _add_claim(
        states[mayor.npc_id],
        prop_id,
        fact.proposition,
        "no",
        SourceType.ROLE_KNOWLEDGE,
        mayor.npc_id,
        1,
        0.9,
    )

    source_graph = [
        _edge(healer.npc_id, guard.npc_id, prop_id, SourceType.TESTIMONY.value),
        _edge("square_crowd", gossip.npc_id, prop_id, SourceType.HEARSAY.value),
        _edge("square_crowd", farmer.npc_id, prop_id, SourceType.HEARSAY.value),
    ]
    event = Event(
        event_id=f"{world_id}_event_0",
        time=1,
        location="well",
        description="The well water changed color because a dye bottle cracked nearby.",
        participants=[],
        observers=[healer.npc_id],
        hidden_truth={prop_id: "no"},
    )
    return World(
        world_id=world_id,
        scenario_type="rumor_propagation",
        locations=locations,
        npcs=npcs,
        events=[event],
        facts={prop_id: fact},
        epistemic_states=states,
        source_graph=source_graph,
    )


def _generate_private_role_world(world_id: str, rng: Random) -> World:
    names = rng.sample(NAMES, 5)
    roles = ["archivist", "captain", "merchant", "cook", "apprentice"]
    locations = ["archive", "gatehouse", "market", "kitchen", "courtyard"]
    npcs = _make_npcs(names, roles, locations)
    archivist, captain, merchant, cook, apprentice = npcs
    code = str(rng.randint(2000, 8999))
    prop_id = "vault_code"
    fact = Fact(
        proposition_id=prop_id,
        proposition=f"The archive vault code is {code}.",
        true_value=code,
        answer_text=code,
        hidden=True,
        suggested_actions=[
            f"ask_npc:{archivist.npc_id}",
            "query_public_record:sealed_registry",
        ],
        false_value=str(rng.randint(2000, 8999)),
        metadata={"object": "archive vault"},
    )

    states = _empty_states(npcs)
    _add_claim(
        states[archivist.npc_id],
        prop_id,
        fact.proposition,
        code,
        SourceType.ROLE_KNOWLEDGE,
        archivist.npc_id,
        1,
        0.98,
    )
    _add_claim(
        states[captain.npc_id],
        prop_id,
        fact.proposition,
        code,
        SourceType.TESTIMONY,
        archivist.npc_id,
        2,
        0.8,
    )
    _add_claim(
        states[apprentice.npc_id],
        prop_id,
        f"The apprentice heard that the archive code may be {fact.false_value}.",
        fact.false_value or "unknown",
        SourceType.HEARSAY,
        "market_rumor",
        3,
        0.25,
        {"rumor": True},
    )

    source_graph = [
        _edge(archivist.npc_id, captain.npc_id, prop_id, SourceType.TESTIMONY.value),
        _edge("market_rumor", apprentice.npc_id, prop_id, SourceType.HEARSAY.value),
    ]
    event = Event(
        event_id=f"{world_id}_event_0",
        time=1,
        location="archive",
        description="The archivist sealed a new vault code in the registry.",
        participants=[archivist.npc_id],
        observers=[archivist.npc_id],
        hidden_truth={prop_id: code},
    )
    return World(
        world_id=world_id,
        scenario_type="private_role_knowledge",
        locations=locations,
        npcs=npcs,
        events=[event],
        facts={prop_id: fact},
        epistemic_states=states,
        source_graph=source_graph,
    )


def _generate_false_presupposition_world(world_id: str, rng: Random) -> World:
    names = rng.sample(NAMES, 5)
    roles = ["suspect", "witness", "gossip", "guard", "scholar"]
    locations = ["chapel", "market", "square", "barracks", "library"]
    npcs = _make_npcs(names, roles, locations)
    suspect, witness, gossip, guard, scholar = npcs
    prop_id = "suspect_hid_relic"
    fact = Fact(
        proposition_id=prop_id,
        proposition=f"{suspect.name} did not hide the relic in the chapel.",
        true_value="no",
        answer_text="no",
        hidden=True,
        suggested_actions=[
            f"ask_npc:{witness.npc_id}",
            "inspect_location:chapel",
            "query_public_record:market_log",
        ],
        false_value="yes",
        metadata={
            "presupposed_claim": f"{suspect.name} hid the relic in the chapel",
            "object": "relic",
        },
    )

    states = _empty_states(npcs)
    _add_claim(
        states[suspect.npc_id],
        prop_id,
        fact.proposition,
        "no",
        SourceType.DIRECT_OBSERVATION,
        suspect.npc_id,
        1,
        0.99,
        {"alibi": "was repairing a cart"},
    )
    _add_claim(
        states[witness.npc_id],
        prop_id,
        fact.proposition,
        "no",
        SourceType.DIRECT_OBSERVATION,
        witness.npc_id,
        1,
        0.92,
        {"observation": f"saw {suspect.name} in the market"},
    )
    _add_claim(
        states[guard.npc_id],
        prop_id,
        fact.proposition,
        "no",
        SourceType.TESTIMONY,
        witness.npc_id,
        2,
        0.75,
    )
    _add_claim(
        states[gossip.npc_id],
        prop_id,
        f"People say {suspect.name} hid the relic in the chapel.",
        "yes",
        SourceType.HEARSAY,
        "square_crowd",
        3,
        0.3,
        {"rumor": True},
    )

    source_graph = [
        _edge(witness.npc_id, guard.npc_id, prop_id, SourceType.TESTIMONY.value),
        _edge("square_crowd", gossip.npc_id, prop_id, SourceType.HEARSAY.value),
    ]
    event = Event(
        event_id=f"{world_id}_event_0",
        time=1,
        location="market",
        description=f"{witness.name} saw {suspect.name} in the market while the chapel was empty.",
        participants=[suspect.npc_id, witness.npc_id],
        observers=[suspect.npc_id, witness.npc_id],
        hidden_truth={prop_id: "no"},
    )
    return World(
        world_id=world_id,
        scenario_type="false_presupposition",
        locations=locations,
        npcs=npcs,
        events=[event],
        facts={prop_id: fact},
        epistemic_states=states,
        source_graph=source_graph,
    )


def _generate_conflicting_world(world_id: str, rng: Random) -> World:
    """One proposition, but several NPCs hold BOTH a strong source (true) and a conflicting
    rumor (false). best_claim should resolve to the strong source; the LLM sees both."""
    names = rng.sample(NAMES, 5)
    roles = ["witness", "detective", "clerk", "guard", "bystander"]
    locations = ["alley", "office", "archive", "gate", "plaza"]
    npcs = _make_npcs(names, roles, locations)
    witness, detective, clerk, guard, bystander = npcs
    true_name = rng.choice(names)
    false_name = rng.choice([name for name in names if name != true_name])
    prop_id = "window_breaker"
    fact = Fact(
        proposition_id=prop_id,
        proposition=f"{true_name} broke the window.",
        true_value=true_name,
        answer_text=true_name,
        hidden=True,
        suggested_actions=[
            f"ask_npc:{witness.npc_id}",
            "inspect_location:alley",
            "query_public_record:incident_log",
        ],
        false_value=false_name,
        metadata={"object": "window", "conflict": True},
    )
    rumor = f"Some people say {false_name} broke the window."
    states = _empty_states(npcs)
    _add_claim(states[witness.npc_id], prop_id, fact.proposition, true_name,
               SourceType.DIRECT_OBSERVATION, witness.npc_id, 1, 0.95,
               {"observation": f"saw {true_name} do it"})
    # detective and guard each hold a conflicting pair (strong-true + rumor-false)
    _add_claim(states[detective.npc_id], prop_id, fact.proposition, true_name,
               SourceType.DIRECT_OBSERVATION, detective.npc_id, 1, 0.92)
    _add_claim(states[detective.npc_id], prop_id, rumor, false_name,
               SourceType.HEARSAY, "street_rumor", 3, 0.30, {"rumor": True})
    _add_claim(states[guard.npc_id], prop_id, fact.proposition, true_name,
               SourceType.TESTIMONY, witness.npc_id, 2, 0.80)
    _add_claim(states[guard.npc_id], prop_id, rumor, false_name,
               SourceType.HEARSAY, "street_rumor", 3, 0.30, {"rumor": True})
    _add_claim(states[clerk.npc_id], prop_id, rumor, false_name,
               SourceType.HEARSAY, "street_rumor", 3, 0.30, {"rumor": True})
    source_graph = [
        _edge(witness.npc_id, guard.npc_id, prop_id, SourceType.TESTIMONY.value),
        _edge("street_rumor", detective.npc_id, prop_id, SourceType.HEARSAY.value),
        _edge("street_rumor", guard.npc_id, prop_id, SourceType.HEARSAY.value),
        _edge("street_rumor", clerk.npc_id, prop_id, SourceType.HEARSAY.value),
    ]
    event = Event(
        event_id=f"{world_id}_event_0",
        time=1,
        location="alley",
        description=f"{true_name} broke the window; a false rumor blamed {false_name}.",
        participants=[],
        observers=[witness.npc_id, detective.npc_id],
        hidden_truth={prop_id: true_name},
    )
    return World(
        world_id=world_id,
        scenario_type="conflicting_evidence",
        locations=locations,
        npcs=npcs,
        events=[event],
        facts={prop_id: fact},
        epistemic_states=states,
        source_graph=source_graph,
    )


def _generate_multihop_rumor_world(world_id: str, rng: Random) -> World:
    """Truth originates at hop 0 (direct), is relayed as testimony at hop 1, then mutates into
    the opposite value as hearsay at hops 2-3. Each NPC should report from its own evidence."""
    names = rng.sample(NAMES, 5)
    roles = ["engineer", "relay_one", "relay_two", "relay_three", "outsider"]
    locations = ["bridge", "mill", "tavern", "farm", "road"]
    npcs = _make_npcs(names, roles, locations)
    engineer, relay_one, relay_two, relay_three, outsider = npcs
    prop_id = "bridge_safe"
    fact = Fact(
        proposition_id=prop_id,
        proposition="The bridge is safe to cross.",
        true_value="yes",
        answer_text="yes",
        hidden=True,
        suggested_actions=[
            f"ask_npc:{engineer.npc_id}",
            "inspect_location:bridge",
            "query_public_record:inspection_log",
        ],
        false_value="no",
        metadata={"object": "bridge", "multi_hop": True},
    )
    distorted = "People are saying the bridge is not safe."
    states = _empty_states(npcs)
    _add_claim(states[engineer.npc_id], prop_id, fact.proposition, "yes",
               SourceType.DIRECT_OBSERVATION, engineer.npc_id, 1, 0.95,
               {"observation": "inspected the bridge", "hop": 0})
    _add_claim(states[relay_one.npc_id], prop_id, fact.proposition, "yes",
               SourceType.TESTIMONY, engineer.npc_id, 2, 0.75, {"hop": 1})
    _add_claim(states[relay_two.npc_id], prop_id, distorted, "no",
               SourceType.HEARSAY, relay_one.npc_id, 3, 0.40,
               {"rumor": True, "hop": 2, "distorted": True})
    _add_claim(states[relay_three.npc_id], prop_id, distorted, "no",
               SourceType.HEARSAY, relay_two.npc_id, 4, 0.25,
               {"rumor": True, "hop": 3, "distorted": True})
    source_graph = [
        _edge(engineer.npc_id, relay_one.npc_id, prop_id, SourceType.TESTIMONY.value),
        _edge(relay_one.npc_id, relay_two.npc_id, prop_id, SourceType.HEARSAY.value),
        _edge(relay_two.npc_id, relay_three.npc_id, prop_id, SourceType.HEARSAY.value),
    ]
    event = Event(
        event_id=f"{world_id}_event_0",
        time=1,
        location="bridge",
        description="The engineer found the bridge safe; a collapse rumor spread and flipped to 'not safe'.",
        participants=[engineer.npc_id],
        observers=[engineer.npc_id],
        hidden_truth={prop_id: "yes"},
    )
    return World(
        world_id=world_id,
        scenario_type="multi_hop_rumor",
        locations=locations,
        npcs=npcs,
        events=[event],
        facts={prop_id: fact},
        epistemic_states=states,
        source_graph=source_graph,
    )


def _make_npcs(names: list[str], roles: list[str], locations: list[str]) -> list[NPC]:
    npcs: list[NPC] = []
    for name, role, location in zip(names, roles, locations, strict=True):
        npcs.append(
            NPC(
                npc_id=role,
                name=name,
                role=role,
                location=location,
            )
        )
    return npcs


def _empty_states(npcs: list[NPC]) -> dict[str, EpistemicState]:
    return {npc.npc_id: EpistemicState(npc_id=npc.npc_id) for npc in npcs}


def _add_claim(
    state: EpistemicState,
    proposition_id: str,
    proposition: str,
    value: str,
    source_type: SourceType,
    source_agent: str | None,
    time_obtained: int,
    reliability: float,
    metadata: dict[str, Any] | None = None,
) -> None:
    state.add_claim(
        Claim(
            proposition_id=proposition_id,
            proposition=proposition,
            value=value,
            source_type=source_type.value,
            source_agent=source_agent,
            time_obtained=time_obtained,
            reliability=reliability,
            verification_status=VerificationStatus.VERIFIED.value
            if reliability >= 0.7
            else VerificationStatus.UNVERIFIED.value,
            metadata=metadata or {},
        )
    )


def _edge(source: str, target: str, proposition_id: str, channel: str) -> dict[str, str]:
    return {
        "source": source,
        "target": target,
        "proposition_id": proposition_id,
        "channel": channel,
    }
