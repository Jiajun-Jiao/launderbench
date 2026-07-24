"""Reference NPC controllers for the E-NPC benchmark MVP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from epistemic_state import (
    ABSTAIN,
    ANSWER_FACT,
    ANSWER_WITH_SOURCE,
    ASK_CLARIFICATION,
    CORRECT_FALSE_PRESUPPOSITION,
    HEDGE,
    SourceType,
    action_mode,
    preferred_mode_for_claim,
)
from query_generator import QueryItem
from world_generator import World


@dataclass(frozen=True)
class NPCResponse:
    controller: str
    query_id: str
    target_npc_id: str
    mode: str
    answer_value: str | None
    text: str
    source_type: str | None = None
    source_agent: str | None = None
    claimed_source_type: str | None = None
    action: str | None = None
    used_global_state: bool = False
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "controller": self.controller,
            "query_id": self.query_id,
            "target_npc_id": self.target_npc_id,
            "mode": self.mode,
            "answer_value": self.answer_value,
            "text": self.text,
            "source_type": self.source_type,
            "source_agent": self.source_agent,
            "claimed_source_type": self.claimed_source_type,
            "action": self.action,
            "used_global_state": self.used_global_state,
            "metadata": self.metadata or {},
        }


class NPCController(Protocol):
    name: str

    def respond(self, world: World, query: QueryItem) -> NPCResponse:
        raise NotImplementedError


class GlobalStateController:
    name = "global_state"

    def respond(self, world: World, query: QueryItem) -> NPCResponse:
        fact = world.facts[query.proposition_id]
        return NPCResponse(
            controller=self.name,
            query_id=query.query_id,
            target_npc_id=query.target_npc_id,
            mode=ANSWER_FACT,
            answer_value=fact.true_value,
            text=f"The answer is {fact.answer_text}.",
            used_global_state=True,
        )


class LocalTranscriptController:
    name = "local_transcript"

    def respond(self, world: World, query: QueryItem) -> NPCResponse:
        state = world.epistemic_states[query.target_npc_id]
        claim = state.best_claim(query.proposition_id)
        if not claim:
            return NPCResponse(
                controller=self.name,
                query_id=query.query_id,
                target_npc_id=query.target_npc_id,
                mode=ABSTAIN,
                answer_value=None,
                text="I do not know from anything I have seen or heard.",
            )

        mode = preferred_mode_for_claim(claim, query.is_false_presupposition)
        return NPCResponse(
            controller=self.name,
            query_id=query.query_id,
            target_npc_id=query.target_npc_id,
            mode=mode,
            answer_value=claim.value,
            text=_text_for_claim(world, claim, mode),
            source_type=claim.source_type,
            source_agent=claim.source_agent,
            claimed_source_type=claim.source_type,
        )


class RAGNPCController:
    """Naive shared-memory RAG that ignores character-local access boundaries."""

    name = "rag_npc"

    def respond(self, world: World, query: QueryItem) -> NPCResponse:
        claim = _best_claim_anywhere(world, query.proposition_id)
        if not claim:
            return NPCResponse(
                controller=self.name,
                query_id=query.query_id,
                target_npc_id=query.target_npc_id,
                mode=ABSTAIN,
                answer_value=None,
                text="I cannot retrieve anything about that.",
            )
        return NPCResponse(
            controller=self.name,
            query_id=query.query_id,
            target_npc_id=query.target_npc_id,
            mode=ANSWER_FACT,
            answer_value=claim.value,
            text=f"I know it was {claim.value}.",
            source_type=claim.source_type,
            source_agent=claim.source_agent,
            claimed_source_type=SourceType.DIRECT_OBSERVATION.value,
            used_global_state=False,
            metadata={"retrieved_from_shared_memory": True},
        )


class PromptOnlyController:
    """Rule prompt simulation without structured provenance enforcement."""

    name = "prompt_only"

    def respond(self, world: World, query: QueryItem) -> NPCResponse:
        state = world.epistemic_states[query.target_npc_id]
        claim = state.best_claim(query.proposition_id)

        if claim:
            if claim.source_type in {
                SourceType.DIRECT_OBSERVATION.value,
                SourceType.ROLE_KNOWLEDGE.value,
                SourceType.PUBLIC_RECORD.value,
            }:
                mode = ANSWER_FACT
                claimed_source_type = claim.source_type
            elif claim.source_type == SourceType.TESTIMONY.value:
                mode = ANSWER_WITH_SOURCE
                claimed_source_type = claim.source_type
            else:
                mode = ANSWER_FACT
                claimed_source_type = SourceType.DIRECT_OBSERVATION.value
            return NPCResponse(
                controller=self.name,
                query_id=query.query_id,
                target_npc_id=query.target_npc_id,
                mode=mode,
                answer_value=claim.value,
                text=_text_for_claim(world, claim, mode),
                source_type=claim.source_type,
                source_agent=claim.source_agent,
                claimed_source_type=claimed_source_type,
            )

        if query.is_false_presupposition and query.presupposed_value is not None:
            return NPCResponse(
                controller=self.name,
                query_id=query.query_id,
                target_npc_id=query.target_npc_id,
                mode=ANSWER_FACT,
                answer_value=query.presupposed_value,
                text="If everyone says it happened, that must be what happened.",
                source_type=SourceType.PLAYER_CLAIM.value,
                source_agent="player",
                claimed_source_type=SourceType.PLAYER_CLAIM.value,
            )

        return NPCResponse(
            controller=self.name,
            query_id=query.query_id,
            target_npc_id=query.target_npc_id,
            mode=ABSTAIN,
            answer_value=None,
            text="I should not claim that without evidence.",
        )


class SourceCalibratedController:
    name = "source_calibrated"

    def respond(self, world: World, query: QueryItem) -> NPCResponse:
        state = world.epistemic_states[query.target_npc_id]
        claim = state.best_claim(query.proposition_id)
        if not claim:
            action = _preferred_action(query)
            return NPCResponse(
                controller=self.name,
                query_id=query.query_id,
                target_npc_id=query.target_npc_id,
                mode=action_mode(action) if action else ASK_CLARIFICATION,
                answer_value=None,
                text=_text_for_action(world, query, action),
                action=action,
            )

        mode = preferred_mode_for_claim(claim, query.is_false_presupposition)
        if mode == CORRECT_FALSE_PRESUPPOSITION:
            text = "I do not know that this happened; my information points the other way."
        elif mode == HEDGE:
            text = _text_for_claim(world, claim, mode)
        else:
            text = _text_for_claim(world, claim, mode)

        return NPCResponse(
            controller=self.name,
            query_id=query.query_id,
            target_npc_id=query.target_npc_id,
            mode=mode,
            answer_value=claim.value,
            text=text,
            source_type=claim.source_type,
            source_agent=claim.source_agent,
            claimed_source_type=claim.source_type,
        )


def default_controllers() -> list[NPCController]:
    return [
        GlobalStateController(),
        LocalTranscriptController(),
        RAGNPCController(),
        PromptOnlyController(),
        SourceCalibratedController(),
    ]


def _best_claim_anywhere(world: World, proposition_id: str):
    claims = []
    for state in world.epistemic_states.values():
        claim = state.best_claim(proposition_id)
        if claim:
            claims.append(claim)
    if not claims:
        return None
    return max(claims, key=lambda claim: (claim.reliability, claim.time_obtained))


def _preferred_action(query: QueryItem) -> str | None:
    if not query.valid_actions:
        return None
    for prefix in ("ask_npc:", "inspect_location:", "query_public_record:"):
        for action in query.valid_actions:
            if action.startswith(prefix):
                return action
    return query.valid_actions[0]


def _text_for_claim(world: World, claim, mode: str) -> str:
    if mode == ANSWER_FACT:
        return f"{claim.value}."
    source = _source_name(world, claim.source_agent)
    if mode == ANSWER_WITH_SOURCE:
        return f"{source} told me it was {claim.value}."
    if mode == CORRECT_FALSE_PRESUPPOSITION:
        return f"I have evidence that the answer is {claim.value}, not the premise in the question."
    if mode == HEDGE:
        return f"I am not certain, but the claim I have is {claim.value}."
    return f"I heard from {source} that it may be {claim.value}."


def _text_for_action(world: World, query: QueryItem, action: str | None) -> str:
    if not action:
        return "I do not have enough evidence; ask me a more specific question."
    mode = action_mode(action)
    value = action.split(":", 1)[1] if ":" in action else action
    if mode == "ask_npc":
        return f"I do not know. You should ask {_source_name(world, value)}."
    if mode == "inspect_location":
        return f"I do not know. You should inspect the {value}."
    if mode == "query_public_record":
        return f"I do not know. You should check {value}."
    return "I do not know, but there is an in-world way to verify it."


def _source_name(world: World, source_agent: str | None) -> str:
    if source_agent is None:
        return "an unknown source"
    try:
        return world.npc(source_agent).name
    except KeyError:
        return source_agent.replace("_", " ")
