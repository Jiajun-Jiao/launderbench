"""Core epistemic state primitives for the E-NPC benchmark MVP."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SourceType(str, Enum):
    DIRECT_OBSERVATION = "direct_observation"
    ENVIRONMENTAL_EVIDENCE = "environmental_evidence"
    TESTIMONY = "testimony"
    HEARSAY = "hearsay"
    PUBLIC_RECORD = "public_record"
    ROLE_KNOWLEDGE = "role_knowledge"
    EPISODIC_MEMORY = "episodic_memory"
    INFERENCE = "inference"
    PLAYER_CLAIM = "player_claim"
    FORBIDDEN_GLOBAL_STATE = "forbidden_global_state"


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    DISPUTED = "disputed"
    STALE = "stale"
    FORBIDDEN = "forbidden"


ANSWER_FACT = "answer_fact"
ANSWER_WITH_SOURCE = "answer_with_source"
ANSWER_AS_HEARSAY = "answer_as_hearsay"
ANSWER_AS_INFERENCE = "answer_as_inference"
CORRECT_FALSE_PRESUPPOSITION = "correct_false_presupposition"
HEDGE = "hedge"
ABSTAIN = "abstain"
ASK_CLARIFICATION = "ask_clarification"
SUGGEST_SOURCE = "suggest_source"
ASK_NPC = "ask_npc"
INSPECT_LOCATION = "inspect_location"
INSPECT_OBJECT = "inspect_object"
QUERY_PUBLIC_RECORD = "query_public_record"
DECLINE_META = "decline_meta"

ASSERTIVE_MODES = {
    ANSWER_FACT,
    ANSWER_WITH_SOURCE,
    ANSWER_AS_HEARSAY,
    ANSWER_AS_INFERENCE,
    CORRECT_FALSE_PRESUPPOSITION,
}

FACTUAL_ASSERTION_MODES = {
    ANSWER_FACT,
    ANSWER_WITH_SOURCE,
    ANSWER_AS_INFERENCE,
    CORRECT_FALSE_PRESUPPOSITION,
}

INFORMATION_SEEKING_MODES = {
    ASK_CLARIFICATION,
    SUGGEST_SOURCE,
    ASK_NPC,
    INSPECT_LOCATION,
    INSPECT_OBJECT,
    QUERY_PUBLIC_RECORD,
}

STRONG_FACT_SOURCES = {
    SourceType.DIRECT_OBSERVATION.value,
    SourceType.PUBLIC_RECORD.value,
    SourceType.ROLE_KNOWLEDGE.value,
    SourceType.EPISODIC_MEMORY.value,
}

ATTRIBUTED_FACT_SOURCES = {
    SourceType.TESTIMONY.value,
}

WEAK_SOURCES = {
    SourceType.HEARSAY.value,
    SourceType.PLAYER_CLAIM.value,
}

INFERENCE_SOURCES = {
    SourceType.ENVIRONMENTAL_EVIDENCE.value,
    SourceType.INFERENCE.value,
}

SOURCE_PRIORITY = {
    SourceType.DIRECT_OBSERVATION.value: 100,
    SourceType.PUBLIC_RECORD.value: 95,
    SourceType.ROLE_KNOWLEDGE.value: 90,
    SourceType.EPISODIC_MEMORY.value: 80,
    SourceType.TESTIMONY.value: 65,
    SourceType.ENVIRONMENTAL_EVIDENCE.value: 55,
    SourceType.INFERENCE.value: 50,
    SourceType.HEARSAY.value: 30,
    SourceType.PLAYER_CLAIM.value: 10,
    SourceType.FORBIDDEN_GLOBAL_STATE.value: -100,
}


@dataclass(frozen=True)
class Claim:
    proposition_id: str
    proposition: str
    value: str
    source_type: str
    source_agent: str | None
    time_obtained: int
    reliability: float
    verification_status: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposition_id": self.proposition_id,
            "proposition": self.proposition,
            "value": self.value,
            "source_type": self.source_type,
            "source_agent": self.source_agent,
            "time_obtained": self.time_obtained,
            "reliability": self.reliability,
            "verification_status": self.verification_status,
            "metadata": self.metadata,
        }


@dataclass
class EpistemicState:
    npc_id: str
    claims: list[Claim] = field(default_factory=list)

    def add_claim(self, claim: Claim) -> None:
        if claim.proposition_id == "":
            raise ValueError("claim.proposition_id must not be empty")
        self.claims.append(claim)

    def claims_for(self, proposition_id: str) -> list[Claim]:
        return [claim for claim in self.claims if claim.proposition_id == proposition_id]

    def best_claim(self, proposition_id: str) -> Claim | None:
        claims = self.claims_for(proposition_id)
        if not claims:
            return None
        return max(
            claims,
            key=lambda claim: (
                claim.verification_status == VerificationStatus.VERIFIED.value,
                SOURCE_PRIORITY.get(claim.source_type, 0),
                claim.reliability,
                claim.time_obtained,
            ),
        )

    def has_direct_claim(self, proposition_id: str) -> bool:
        return any(
            claim.source_type == SourceType.DIRECT_OBSERVATION.value
            for claim in self.claims_for(proposition_id)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "npc_id": self.npc_id,
            "claims": [claim.to_dict() for claim in self.claims],
        }


def action_mode(action: str) -> str:
    """Return the response mode encoded by an action string such as ask_npc:guard."""
    return action.split(":", 1)[0]


def allowed_modes_for_claim(
    claim: Claim | None,
    valid_actions: list[str] | None = None,
    is_false_presupposition: bool = False,
) -> list[str]:
    action_modes = [action_mode(action) for action in valid_actions or []]

    if claim is None:
        return unique_modes([ABSTAIN, SUGGEST_SOURCE, *action_modes, ASK_CLARIFICATION])

    source_type = claim.source_type
    modes: list[str]
    if source_type in STRONG_FACT_SOURCES:
        modes = [ANSWER_FACT, ANSWER_WITH_SOURCE]
    elif source_type in ATTRIBUTED_FACT_SOURCES:
        modes = [ANSWER_WITH_SOURCE, HEDGE]
    elif source_type == SourceType.HEARSAY.value:
        modes = [ANSWER_AS_HEARSAY, HEDGE, SUGGEST_SOURCE, *action_modes]
    elif source_type in INFERENCE_SOURCES:
        modes = [ANSWER_AS_INFERENCE, HEDGE, SUGGEST_SOURCE, *action_modes]
    elif source_type == SourceType.PLAYER_CLAIM.value:
        modes = [HEDGE, ASK_CLARIFICATION, SUGGEST_SOURCE, *action_modes, ABSTAIN]
    elif source_type == SourceType.FORBIDDEN_GLOBAL_STATE.value:
        modes = [ABSTAIN, DECLINE_META]
    else:
        modes = [HEDGE, SUGGEST_SOURCE, *action_modes]

    if is_false_presupposition and claim.value == "no":
        modes.insert(0, CORRECT_FALSE_PRESUPPOSITION)

    return unique_modes(modes)


def preferred_mode_for_claim(
    claim: Claim,
    is_false_presupposition: bool = False,
) -> str:
    if is_false_presupposition and claim.value == "no":
        return CORRECT_FALSE_PRESUPPOSITION
    if claim.source_type in STRONG_FACT_SOURCES:
        return ANSWER_FACT
    if claim.source_type in ATTRIBUTED_FACT_SOURCES:
        return ANSWER_WITH_SOURCE
    if claim.source_type == SourceType.HEARSAY.value:
        return ANSWER_AS_HEARSAY
    if claim.source_type in INFERENCE_SOURCES:
        return ANSWER_AS_INFERENCE
    if claim.source_type == SourceType.PLAYER_CLAIM.value:
        return HEDGE
    return ABSTAIN


def source_supports_mode(claim: Claim | None, mode: str) -> bool:
    if claim is None:
        return mode in {ABSTAIN, ASK_CLARIFICATION, SUGGEST_SOURCE} | INFORMATION_SEEKING_MODES

    source_type = claim.source_type
    if mode == ANSWER_FACT:
        return source_type in STRONG_FACT_SOURCES
    if mode == ANSWER_WITH_SOURCE:
        return source_type in STRONG_FACT_SOURCES | ATTRIBUTED_FACT_SOURCES
    if mode == ANSWER_AS_HEARSAY:
        return source_type == SourceType.HEARSAY.value
    if mode == ANSWER_AS_INFERENCE:
        return source_type in INFERENCE_SOURCES
    if mode == CORRECT_FALSE_PRESUPPOSITION:
        return claim.value == "no" and source_type in STRONG_FACT_SOURCES | ATTRIBUTED_FACT_SOURCES
    if mode == HEDGE:
        return source_type not in {SourceType.FORBIDDEN_GLOBAL_STATE.value}
    if mode in {ABSTAIN, ASK_CLARIFICATION, SUGGEST_SOURCE} | INFORMATION_SEEKING_MODES:
        return True
    return False


def unique_modes(modes: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for mode in modes:
        if mode and mode not in seen:
            result.append(mode)
            seen.add(mode)
    return result
