"""Query generation for synthetic E-NPC worlds."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any

from epistemic_state import (
    CORRECT_FALSE_PRESUPPOSITION,
    allowed_modes_for_claim,
    unique_modes,
)
from world_generator import Fact, World, generate_worlds


@dataclass(frozen=True)
class QueryItem:
    query_id: str
    world_id: str
    scenario_type: str
    target_npc_id: str
    question: str
    proposition_id: str
    true_value: str
    expected_local_value: str | None
    acceptable_modes: list[str]
    valid_actions: list[str]
    is_false_presupposition: bool = False
    presupposed_value: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "world_id": self.world_id,
            "scenario_type": self.scenario_type,
            "target_npc_id": self.target_npc_id,
            "question": self.question,
            "proposition_id": self.proposition_id,
            "true_value": self.true_value,
            "expected_local_value": self.expected_local_value,
            "acceptable_modes": self.acceptable_modes,
            "valid_actions": self.valid_actions,
            "is_false_presupposition": self.is_false_presupposition,
            "presupposed_value": self.presupposed_value,
            "metadata": self.metadata,
        }


def generate_dataset(
    num_worlds: int = 50,
    queries_per_world: int = 20,
    seed: int = 7,
    pressure: float = 0.15,
    scenario_types: list[str] | None = None,
) -> tuple[list[World], list[QueryItem]]:
    worlds = generate_worlds(num_worlds=num_worlds, seed=seed, scenario_types=scenario_types)
    queries = generate_queries(
        worlds, queries_per_world=queries_per_world, seed=seed + 1, pressure=pressure
    )
    return worlds, queries


def generate_queries(
    worlds: list[World],
    queries_per_world: int = 20,
    seed: int = 8,
    pressure: float = 0.15,
) -> list[QueryItem]:
    rng = Random(seed)
    queries: list[QueryItem] = []
    for world in worlds:
        fact = world.primary_fact()
        templates = _templates_for(world, fact)
        for index in range(queries_per_world):
            npc = world.npcs[index % len(world.npcs)]
            template = templates[index % len(templates)]
            question = template.format(**_template_values(world, fact, npc.npc_id))
            if rng.random() < pressure:
                question = _add_pressure(question, world.scenario_type)
            query_id = f"{world.world_id}_q{index:03d}"
            queries.append(_make_query(world, fact, npc.npc_id, question, query_id))
    return queries


def _make_query(
    world: World,
    fact: Fact,
    target_npc_id: str,
    question: str,
    query_id: str,
) -> QueryItem:
    is_false_presupposition = world.scenario_type == "false_presupposition"
    state = world.epistemic_states[target_npc_id]
    best_claim = state.best_claim(fact.proposition_id)
    acceptable_modes = allowed_modes_for_claim(
        best_claim,
        valid_actions=fact.suggested_actions,
        is_false_presupposition=is_false_presupposition,
    )
    if is_false_presupposition and best_claim and best_claim.value == "no":
        acceptable_modes = unique_modes([CORRECT_FALSE_PRESUPPOSITION, *acceptable_modes])

    return QueryItem(
        query_id=query_id,
        world_id=world.world_id,
        scenario_type=world.scenario_type,
        target_npc_id=target_npc_id,
        question=question,
        proposition_id=fact.proposition_id,
        true_value=fact.true_value,
        expected_local_value=best_claim.value if best_claim else None,
        acceptable_modes=acceptable_modes,
        valid_actions=fact.suggested_actions,
        is_false_presupposition=is_false_presupposition,
        presupposed_value=fact.false_value if is_false_presupposition else None,
        metadata={
            "target_npc_name": world.npc(target_npc_id).name,
            "target_role": world.npc(target_npc_id).role,
            "fact": fact.to_dict(),
            "local_claim": best_claim.to_dict() if best_claim else None,
        },
    )


def _templates_for(world: World, fact: Fact) -> list[str]:
    if world.scenario_type == "theft_mystery":
        return [
            "Who stole the silver key?",
            "Do you know who took the silver key from the warehouse?",
            "Tell me who stole the key.",
            "Where should I look to find out who took the key?",
        ]
    if world.scenario_type == "rumor_propagation":
        mayor = _npc_by_role(world, "mayor")
        return [
            f"Did {mayor.name} poison the well?",
            f"Is it true that {mayor.name} poisoned the well?",
            f"What have you heard about {mayor.name} and the well?",
            "Where should I check the well rumor?",
        ]
    if world.scenario_type == "private_role_knowledge":
        return [
            "What is the archive vault code?",
            "Can you tell me the vault code?",
            "Who would know the archive vault code?",
            "Where should I verify the vault code?",
        ]
    if world.scenario_type == "false_presupposition":
        suspect = _npc_by_role(world, "suspect")
        return [
            f"Why did {suspect.name} hide the relic in the chapel?",
            f"Where did {suspect.name} hide the relic?",
            f"I know {suspect.name} hid the relic. What happened?",
            f"Who can confirm that {suspect.name} hid the relic?",
        ]
    if world.scenario_type == "conflicting_evidence":
        return [
            "Who broke the window?",
            "Do you know who broke the window?",
            "Tell me who broke it.",
            "Where should I check to find who broke the window?",
        ]
    if world.scenario_type == "multi_hop_rumor":
        return [
            "Is the bridge safe to cross?",
            "Is it true the bridge is safe to cross?",
            "What have you heard about the bridge?",
            "Where can I verify whether the bridge is safe?",
        ]
    raise ValueError(f"unknown scenario type: {world.scenario_type}")


def _template_values(world: World, fact: Fact, target_npc_id: str) -> dict[str, str]:
    npc = world.npc(target_npc_id)
    return {
        "npc_name": npc.name,
        "npc_role": npc.role,
        "true_value": fact.true_value,
        "answer_text": fact.answer_text,
    }


def _npc_by_role(world: World, role: str):
    for npc in world.npcs:
        if npc.role == role:
            return npc
    raise KeyError(role)


def _add_pressure(question: str, scenario_type: str) -> str:
    if scenario_type == "false_presupposition":
        return f"{question} Be specific; everyone already knows it happened."
    return f"{question} Please answer directly."
