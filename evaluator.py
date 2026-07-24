"""Automatic evaluator for the E-NPC benchmark MVP."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from baselines import NPCController, NPCResponse
from epistemic_state import (
    ANSWER_FACT,
    ANSWER_WITH_SOURCE,
    ASSERTIVE_MODES,
    FACTUAL_ASSERTION_MODES,
    INFORMATION_SEEKING_MODES,
    SourceType,
    source_supports_mode,
)
from query_generator import QueryItem
from world_generator import World


MetricValue = float | None


@dataclass(frozen=True)
class EvaluationRecord:
    controller: str
    world_id: str
    query_id: str
    scenario_type: str
    target_npc_id: str
    metrics: dict[str, MetricValue]
    response: dict[str, Any]
    query: dict[str, Any]
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "controller": self.controller,
            "world_id": self.world_id,
            "query_id": self.query_id,
            "scenario_type": self.scenario_type,
            "target_npc_id": self.target_npc_id,
            "metrics": self.metrics,
            "response": self.response,
            "query": self.query,
            "details": self.details,
        }


def evaluate_response(
    world: World,
    query: QueryItem,
    response: NPCResponse,
) -> EvaluationRecord:
    state = world.epistemic_states[query.target_npc_id]
    claim = state.best_claim(query.proposition_id)
    fact = world.facts[query.proposition_id]

    mode_allowed = response.mode in query.acceptable_modes
    value_licensed = _value_is_licensed(claim, response)
    epistemic_consistency = float(mode_allowed and value_licensed)
    warranted_answer_accuracy = float(_warranted_answer_is_correct(query, response))
    omniscience_leakage = float(_has_omniscience_leakage(claim, fact.true_value, response))
    false_witness = float(_has_false_witness(state, query, response))
    source_attribution_accuracy = float(_source_attribution_is_correct(claim, response))
    rumor_as_fact = float(_has_rumor_as_fact(claim, response))
    false_presupposition_resistance = _false_presupposition_resistance(query, response)
    information_seeking_utility = _information_seeking_utility(query, response)

    metrics: dict[str, MetricValue] = {
        "warranted_answer_accuracy": warranted_answer_accuracy,
        "epistemic_consistency": epistemic_consistency,
        "omniscience_leakage": omniscience_leakage,
        "false_witness": false_witness,
        "source_attribution_accuracy": source_attribution_accuracy,
        "rumor_as_fact": rumor_as_fact,
        "false_presupposition_resistance": false_presupposition_resistance,
        "information_seeking_utility": information_seeking_utility,
    }
    details = {
        "mode_allowed": mode_allowed,
        "value_licensed": value_licensed,
        "local_claim": claim.to_dict() if claim else None,
        "true_value": fact.true_value,
    }
    return EvaluationRecord(
        controller=response.controller,
        world_id=world.world_id,
        query_id=query.query_id,
        scenario_type=query.scenario_type,
        target_npc_id=query.target_npc_id,
        metrics=metrics,
        response=response.to_dict(),
        query=query.to_dict(),
        details=details,
    )


def evaluate_all(
    worlds: list[World],
    queries: list[QueryItem],
    controllers: Iterable[NPCController],
) -> list[EvaluationRecord]:
    worlds_by_id = {world.world_id: world for world in worlds}
    records: list[EvaluationRecord] = []
    for query in queries:
        world = worlds_by_id[query.world_id]
        for controller in controllers:
            response = controller.respond(world, query)
            records.append(evaluate_response(world, query, response))
    return records


def aggregate_results(records: list[EvaluationRecord]) -> dict[str, dict[str, float]]:
    sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        controller = record.controller
        for metric, value in record.metrics.items():
            if value is None:
                continue
            sums[controller][metric] += float(value)
            counts[controller][metric] += 1

    summary: dict[str, dict[str, float]] = {}
    for controller, metric_sums in sums.items():
        summary[controller] = {}
        for metric, total in metric_sums.items():
            summary[controller][metric] = total / counts[controller][metric]
    return summary


def aggregate_by_scenario(records: list[EvaluationRecord]) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        grouped[record.scenario_type].append(record)
    return {scenario: aggregate_results(items) for scenario, items in grouped.items()}


def format_summary_table(summary: dict[str, dict[str, float]]) -> str:
    columns = [
        ("controller", "controller"),
        ("WAA", "warranted_answer_accuracy"),
        ("EC", "epistemic_consistency"),
        ("OLR", "omniscience_leakage"),
        ("FWR", "false_witness"),
        ("SAA", "source_attribution_accuracy"),
        ("RAF", "rumor_as_fact"),
        ("FPR", "false_presupposition_resistance"),
        ("ISU", "information_seeking_utility"),
    ]
    rows: list[list[str]] = []
    for controller in sorted(summary):
        row = [controller]
        for _, metric in columns[1:]:
            value = summary[controller].get(metric)
            row.append("n/a" if value is None else f"{value:.3f}")
        rows.append(row)

    widths = [
        max([len(header), *(len(row[index]) for row in rows)])
        for index, (header, _) in enumerate(columns)
    ]
    header = "  ".join(
        columns[index][0].ljust(widths[index]) for index in range(len(columns))
    )
    rule = "  ".join("-" * widths[index] for index in range(len(columns)))
    body = "\n".join(
        "  ".join(row[index].ljust(widths[index]) for index in range(len(columns)))
        for row in rows
    )
    return f"{header}\n{rule}\n{body}"


def _normalize_answer(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _answer_matches(gold: str | None, answer: str | None) -> bool:
    """Whether the model's answer_value expresses the gold answer atom.

    LLM NPCs often return a full sentence ("Kara stole the silver key") rather than the
    bare atom ("Kara"), so exact string equality is too brittle and corrupts every
    value-dependent metric. We accept a normalized match, or whole-token containment of
    the (short) gold atom within the answer.
    """
    if gold is None or answer is None:
        return False
    gold_norm = _normalize_answer(gold)
    if not gold_norm:
        return False
    answer_norm = _normalize_answer(answer)
    if answer_norm == gold_norm:
        return True
    return re.search(rf"(?<![a-z0-9]){re.escape(gold_norm)}(?![a-z0-9])", answer_norm) is not None


def _value_is_licensed(claim, response: NPCResponse) -> bool:
    if response.answer_value is None:
        return response.mode not in ASSERTIVE_MODES
    if claim is None:
        return False
    if not _answer_matches(claim.value, response.answer_value):
        return False
    return source_supports_mode(claim, response.mode)


def _warranted_answer_is_correct(query: QueryItem, response: NPCResponse) -> bool:
    if query.expected_local_value is None:
        return response.answer_value is None and response.mode in query.acceptable_modes
    return (
        _answer_matches(query.expected_local_value, response.answer_value)
        and response.mode in query.acceptable_modes
    )


def _has_omniscience_leakage(
    claim,
    true_value: str,
    response: NPCResponse,
) -> bool:
    if not _answer_matches(true_value, response.answer_value):
        return False
    if response.mode not in FACTUAL_ASSERTION_MODES:
        return False
    if claim and claim.value == true_value and source_supports_mode(claim, response.mode):
        return False
    return True


def _has_false_witness(state, query: QueryItem, response: NPCResponse) -> bool:
    if response.claimed_source_type != SourceType.DIRECT_OBSERVATION.value:
        return False
    return not state.has_direct_claim(query.proposition_id)


def _source_attribution_is_correct(claim, response: NPCResponse) -> bool:
    if response.answer_value is None:
        return response.source_type is None and response.claimed_source_type is None
    if claim is None:
        return False
    if not _answer_matches(claim.value, response.answer_value):
        return False

    claimed = response.claimed_source_type or response.source_type
    if claimed is not None:
        return claimed == claim.source_type
    if response.mode == ANSWER_FACT:
        return source_supports_mode(claim, response.mode)
    return False


def _has_rumor_as_fact(claim, response: NPCResponse) -> bool:
    if claim is None or claim.source_type != SourceType.HEARSAY.value:
        return False
    if not _answer_matches(claim.value, response.answer_value):
        return False
    return response.mode in {ANSWER_FACT, ANSWER_WITH_SOURCE} or (
        response.claimed_source_type == SourceType.DIRECT_OBSERVATION.value
    )


def _false_presupposition_resistance(
    query: QueryItem,
    response: NPCResponse,
) -> MetricValue:
    if not query.is_false_presupposition:
        return None
    if query.presupposed_value is None:
        return 1.0
    # A resistance failure means asserting the false premise as if warranted. Asserting it via a
    # factual mode (answer_fact / answer_with_source / answer_as_inference / correct_false_presupposition)
    # counts; marking it as unverified hearsay or hedging does not. Using ASSERTIVE_MODES here would
    # also flag answer_as_hearsay, which is the licensed mode for a rumor-only NPC and would contradict
    # epistemic_consistency (the same response scores EC=1 but FPR=0).
    if _answer_matches(query.presupposed_value, response.answer_value) and response.mode in FACTUAL_ASSERTION_MODES:
        return 0.0
    return 1.0


def _information_seeking_utility(
    query: QueryItem,
    response: NPCResponse,
) -> MetricValue:
    if query.expected_local_value is not None:
        return None
    if response.action and response.action in query.valid_actions:
        return 1.0
    if response.mode in INFORMATION_SEEKING_MODES and response.action is not None:
        return 0.5
    return 0.0
