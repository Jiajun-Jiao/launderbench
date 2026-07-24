#!/usr/bin/env python3
"""Graph-only citation-lineage intervention with fail-closed API execution.

The canonical trace/solid contrast changes both page text and graph structure. This instrument keeps
the claim, question, five page bodies, page domains, within-scenario page order, page count, apparent
warrant, edge count, terminal count, and per-page indegree fixed. Page order is counterbalanced
across scenarios. The rendered arms differ only in two ``Reference pointer`` lines.

Why a placeholder is necessary. Let C be the pages whose fixed body states the target claim and let
T(G) be the terminal pages of an acyclic citation graph G. If every page is claim-bearing (C=V), the
number of terminal claim origins is |T(G) intersect C| = |T(G)|. Holding the terminal count fixed
therefore makes a one-origin versus two-origin contrast impossible. The minimal compensation is one
fixed non-claim page whose terminal status swaps with a claim-bearing page. We retain the benchmark's
five-page shape: P1--P4 state the claim; P5 is the fixed placeholder.

Template (only P2 and P5 pointers change):

  dependent:   P1->P2, P2->P3, P3->NONE, P4->P3, P5->NONE
  independent: P1->P2, P2->NONE, P3->NONE, P4->P3, P5->P3

Both arms have 5 pages, 3 edges, 2 terminals, and identical per-page indegrees. Claim-bearing paths
terminate at {P3} versus {P2,P3}. P2 and P3 use symmetric fixed registry-entry bodies.

Safe commands (no model calls):
  python3 experiments/graph_only_rewire.py build
  python3 experiments/graph_only_rewire.py dry-run --tasks answer,roots

Actual execution is deliberately gated and is NOT run by build/dry-run:
  python3 experiments/graph_only_rewire.py run ... --confirm-api-spend I_UNDERSTAND
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (str(ROOT), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from society_reflective import client_for  # noqa: E402
from research_citation_multi import wilson  # noqa: E402


DEFAULT_SOURCE = ROOT / "outputs" / "society" / "citation_scenarios_100_v2.jsonl"
DEFAULT_INSTRUMENT = ROOT / "outputs" / "society" / "graph_only_rewire_instrument.json"
CONFESSED_INSTRUMENT = ROOT / "outputs" / "society" / "graph_only_rewire_confessed_instrument.json"
DEFAULT_RESULT_DIR = ROOT / "outputs" / "society"
ARMS = ("dependent", "independent")
# Origin-register axis: the registry level is the original probe; the confessed level rewrites only
# the symmetric P2/P3 root pages as self-described unverified personal notes (domain + body +
# warrant profile), holding P1/P4/P5, the claim sentence, wiring, order, and counts fixed. Crossing
# it with the edge axis yields a register x edges factorial that separates surface-register
# restraint from citation-edge use.
REGISTERS = ("registry", "confessed")
PAGE_ORDER = ("P1", "P2", "P3", "P4", "P5")
CLAIM_PAGES = frozenset({"P1", "P2", "P3", "P4"})
PLACEHOLDER_PAGE = "P5"
EDGE_TEMPLATES = {
    "dependent": {"P1": "P2", "P2": "P3", "P3": None, "P4": "P3", "P5": None},
    "independent": {"P1": "P2", "P2": None, "P3": None, "P4": "P3", "P5": "P3"},
}
EXPECTED_TERMINALS = {"dependent": ["P3", "P5"], "independent": ["P2", "P3"]}
EXPECTED_CLAIM_ROOTS = {"dependent": ["P3"], "independent": ["P2", "P3"]}
TASKS = ("answer", "roots")
CONFIRMATION = "I_UNDERSTAND"

# Frozen project-local list prices, not a live price lookup. Unknown models remain unpriced rather
# than receiving an invented estimate. Local models have zero API charge but nonzero compute cost.
PRICE_AS_OF = "2026-07-07"
PRICES_PER_MTOK = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1": (2.00, 8.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-8": (5.00, 25.00),
}

ANSWER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}
ROOT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "claim_root_page_ids": {
            "type": "array",
            "items": {"type": "string", "enum": list(PAGE_ORDER)},
        },
        "independent_claim_lineages": {"type": "integer", "minimum": 0, "maximum": 5},
        "explanation": {"type": "string"},
    },
    "required": ["claim_root_page_ids", "independent_claim_lineages", "explanation"],
}
JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "stance": {"type": "string", "enum": ["established", "hedged", "absent"]},
        "lineage_statement": {
            "type": "string",
            "enum": ["multiple_independent", "single_lineage", "no_lineage_statement"],
        },
        "target_value_status": {
            "type": "string",
            "enum": ["target_value", "other_value", "no_value"],
        },
    },
    "required": ["stance", "lineage_statement", "target_value_status"],
}


def canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_value(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_model(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def variant_prefix(instrument: dict) -> str:
    """File-name infix for non-default origin registers; empty for the original registry probe."""
    register = instrument.get("origin_register", "registry")
    if register not in REGISTERS:
        raise ValueError(f"unknown origin register {register!r}; allowed={REGISTERS}")
    return "" if register == "registry" else f"{register}_"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def root_page(source: dict, page_id: str, suffix: str, register: str) -> dict:
    sid = source["id"].lower()
    claim = source["claim_sentence"].strip()
    if register == "registry":
        return {
            "id": page_id,
            "domain": f"registry-{suffix.lower()}-{sid}.example",
            "body": f"Dated registry entry {source['id']}-{suffix}. {claim} The value appears in the entry's principal field.",
            "claim_bearing": True,
            "warrant_profile": "dated_registry_entry",
        }
    if register == "confessed":
        return {
            "id": page_id,
            "domain": f"personal-notes-{suffix.lower()}-{sid}.example",
            "body": (
                f"Personal note {source['id']}-{suffix}. {claim} The note's author calls this a "
                "personal guess, with no measurement, record, or verification behind it."
            ),
            "claim_bearing": True,
            "warrant_profile": "self_described_unverified_note",
        }
    raise ValueError(f"unknown origin register {register!r}; allowed={REGISTERS}")


def make_pages(source: dict, register: str = "registry") -> list[dict]:
    sid = source["id"].lower()
    claim = source["claim_sentence"].strip()
    return [
        {
            "id": "P1",
            "domain": f"research-digest-{sid}.example",
            "body": f"Research digest entry. {claim} The entry presents the value without qualification.",
            "claim_bearing": True,
            "warrant_profile": "secondary_digest",
        },
        root_page(source, "P2", "A", register),
        root_page(source, "P3", "B", register),
        {
            "id": "P4",
            "domain": f"reference-bulletin-{sid}.example",
            "body": f"Reference bulletin entry. {claim} The bulletin presents the value without qualification.",
            "claim_bearing": True,
            "warrant_profile": "secondary_bulletin",
        },
        {
            "id": "P5",
            "domain": f"catalog-crosswalk-{sid}.example",
            "body": (
                "Catalog crosswalk note. This page describes how document identifiers in this "
                "collection are formatted. It contains no statement about the queried fact or value; "
                "a cited registry entry, if present, serves only as an identifier-format example."
            ),
            "claim_bearing": False,
            "warrant_profile": "non_claim_placeholder",
        },
    ]


def terminal_ids(edges: dict[str, str | None]) -> list[str]:
    return sorted(page_id for page_id, target in edges.items() if target is None)


def follow_terminal(start: str, edges: dict[str, str | None]) -> str:
    seen = set()
    current = start
    while edges[current] is not None:
        if current in seen:
            raise ValueError(f"citation cycle reached from {start}: {sorted(seen)}")
        seen.add(current)
        current = edges[current]
    return current


def claim_root_ids(pages: list[dict], edges: dict[str, str | None]) -> list[str]:
    claim_ids = [page["id"] for page in pages if page["claim_bearing"]]
    return sorted({follow_terminal(page_id, edges) for page_id in claim_ids})


def indegrees(edges: dict[str, str | None]) -> dict[str, int]:
    counts = {page_id: 0 for page_id in edges}
    for target in edges.values():
        if target is not None:
            counts[target] += 1
    return counts


def render_scenario(scenario: dict, arm: str) -> str:
    edges = scenario["arms"][arm]["citation_edges"]
    blocks = []
    for page in scenario["pages"]:
        target = edges[page["id"]]
        blocks.append(
            f"- [{page['id']}] {page['domain']}\n"
            f"  Body: {page['body']}\n"
            f"  Reference pointer: {target if target is not None else 'NONE'}"
        )
    return "\n".join(blocks)


def strip_reference_lines(rendered: str) -> str:
    return "\n".join(
        line for line in rendered.splitlines() if not line.startswith("  Reference pointer:")
    )


def build_scenario(source: dict, index: int, register: str = "registry") -> dict:
    canonical_pages = make_pages(source, register)
    offset = index % len(canonical_pages)
    pages = canonical_pages[offset:] + canonical_pages[:offset]
    scenario = {
        "id": f"G{index:03d}",
        "source_scenario_id": source["id"],
        "domain": source["domain"],
        "entity": source["entity"],
        "value": source["value"],
        "claim_sentence": source["claim_sentence"],
        "question": source["question"],
        "page_order": [page["id"] for page in pages],
        "pages": pages,
        "arms": {
            arm: {
                "citation_edges": dict(EDGE_TEMPLATES[arm]),
                "expected_terminal_ids": list(EXPECTED_TERMINALS[arm]),
                "expected_claim_root_ids": list(EXPECTED_CLAIM_ROOTS[arm]),
            }
            for arm in ARMS
        },
        "noncitation_sha256": sha256_value(
            {"question": source["question"], "claim": source["claim_sentence"], "pages": pages}
        ),
    }
    validate_scenario(scenario)
    return scenario


def validate_scenario(scenario: dict) -> dict:
    errors = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    page_ids = [page["id"] for page in scenario.get("pages", [])]
    declared_order = scenario.get("page_order", [])
    require(page_ids == declared_order, f"rendered page order {page_ids} differs from declared {declared_order}")
    require(set(page_ids) == set(PAGE_ORDER), "page ids must be exactly P1..P5")
    pages = {page["id"]: page for page in scenario.get("pages", [])}
    require(set(pages) == set(PAGE_ORDER), "page ids must be exactly P1..P5")
    if set(pages) == set(PAGE_ORDER):
        require(
            {page_id for page_id, page in pages.items() if page.get("claim_bearing")} == CLAIM_PAGES,
            "P1..P4 must be claim-bearing and P5 must be the sole non-claim placeholder",
        )
        require(
            pages["P2"].get("warrant_profile") == pages["P3"].get("warrant_profile"),
            "P2/P3 warrant profiles must be symmetric",
        )

    arm_reports = {}
    for arm in ARMS:
        edges = scenario.get("arms", {}).get(arm, {}).get("citation_edges", {})
        require(set(edges) == set(PAGE_ORDER), f"{arm}: edge keys differ from page ids")
        if set(edges) != set(PAGE_ORDER):
            continue
        for page_id, target in edges.items():
            require(target is None or target in edges, f"{arm}: invalid target {target!r}")
            require(target != page_id, f"{arm}: self-loop at {page_id}")
        try:
            for page_id in PAGE_ORDER:
                follow_terminal(page_id, edges)
            terminals = terminal_ids(edges)
            roots = claim_root_ids(scenario["pages"], edges)
        except ValueError as exc:
            errors.append(f"{arm}: {exc}")
            continue
        report = {
            "terminals": terminals,
            "claim_roots": roots,
            "edge_count": sum(target is not None for target in edges.values()),
            "indegrees": indegrees(edges),
        }
        arm_reports[arm] = report
        require(terminals == EXPECTED_TERMINALS[arm], f"{arm}: terminals {terminals}")
        require(roots == EXPECTED_CLAIM_ROOTS[arm], f"{arm}: claim roots {roots}")
        require(report["edge_count"] == 3, f"{arm}: edge count {report['edge_count']}")
        if roots and set(pages) == set(PAGE_ORDER):
            require(all(pages[root]["claim_bearing"] for root in roots), f"{arm}: claim path ends at placeholder")

    if len(arm_reports) == 2:
        dep, ind = arm_reports["dependent"], arm_reports["independent"]
        require(len(dep["terminals"]) == len(ind["terminals"]) == 2, "terminal counts differ")
        require(dep["edge_count"] == ind["edge_count"] == 3, "edge counts differ")
        require(dep["indegrees"] == ind["indegrees"], "per-page indegrees differ")
        changed = sorted(
            page_id
            for page_id in PAGE_ORDER
            if scenario["arms"]["dependent"]["citation_edges"][page_id]
            != scenario["arms"]["independent"]["citation_edges"][page_id]
        )
        require(changed == ["P2", "P5"], f"changed pointers are {changed}, expected P2/P5")
        require(
            strip_reference_lines(render_scenario(scenario, "dependent"))
            == strip_reference_lines(render_scenario(scenario, "independent")),
            "rendered non-reference text differs between arms",
        )

    expected_hash = sha256_value(
        {
            "question": scenario.get("question"),
            "claim": scenario.get("claim_sentence"),
            "pages": scenario.get("pages"),
        }
    )
    require(scenario.get("noncitation_sha256") == expected_hash, "noncitation hash mismatch")
    if errors:
        raise ValueError(f"{scenario.get('id', '<unknown>')}: " + "; ".join(errors))
    return {"scenario_id": scenario["id"], "arms": arm_reports, "changed_pointer_pages": ["P2", "P5"]}


def build_instrument(source_path: Path, limit: int = 0, register: str = "registry") -> dict:
    if register not in REGISTERS:
        raise ValueError(f"unknown origin register {register!r}; allowed={REGISTERS}")
    source_rows = load_jsonl(source_path)
    if limit:
        source_rows = source_rows[:limit]
    scenarios = [build_scenario(source, index, register) for index, source in enumerate(source_rows)]
    instrument = {
        "schema_version": 1,
        "instrument": "graph_only_rewire",
        "origin_register": register,
        "generated_by": "experiments/graph_only_rewire.py",
        "source": str(source_path.relative_to(ROOT) if source_path.is_relative_to(ROOT) else source_path),
        "source_sha256": sha256_file(source_path),
        "design": {
            "origin_register": register,
            "page_count": 5,
            "claim_page_count": 4,
            "placeholder_page": PLACEHOLDER_PAGE,
            "edge_count_per_arm": 3,
            "terminal_count_per_arm": 2,
            "page_order_matched_within_scenario": True,
            "page_order_counterbalanced_across_scenarios": True,
            "only_model_visible_difference": "Reference pointer lines for P2 and P5",
            "dependent_claim_roots": ["P3"],
            "independent_claim_roots": ["P2", "P3"],
            "impossibility_if_all_pages_claim_bearing": (
                "If C=V, the number of terminal claim origins is |T(G) intersect C|=|T(G)|; "
                "a fixed terminal count cannot encode one versus two claim origins."
            ),
            "minimal_compensation": (
                "One fixed non-claim page swaps terminal status with P2; no sixth page is needed."
            ),
        },
        "scenarios": scenarios,
    }
    instrument["instrument_sha256"] = sha256_value(
        {key: value for key, value in instrument.items() if key != "instrument_sha256"}
    )
    validate_instrument(instrument)
    return instrument


def validate_instrument(instrument: dict) -> dict:
    scenarios = instrument.get("scenarios", [])
    if not scenarios:
        raise ValueError("instrument has no scenarios")
    reports = [validate_scenario(scenario) for scenario in scenarios]
    ids = [scenario["id"] for scenario in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate scenario ids")
    expected_hash = sha256_value(
        {key: value for key, value in instrument.items() if key != "instrument_sha256"}
    )
    if instrument.get("instrument_sha256") != expected_hash:
        raise ValueError("instrument_sha256 mismatch")
    first_page_counts = Counter(scenario["page_order"][0] for scenario in scenarios)
    balanced_counts = [first_page_counts.get(page_id, 0) for page_id in PAGE_ORDER]
    if max(balanced_counts) - min(balanced_counts) > 1:
        raise ValueError(f"first-page counterbalancing failed: {dict(first_page_counts)}")
    return {
        "n": len(scenarios),
        "all_invariants_pass": True,
        "changed_pointer_pages": reports[0]["changed_pointer_pages"],
        "instrument_sha256": expected_hash,
        "first_page_counts": dict(sorted(first_page_counts.items())),
    }


def answer_messages(scenario: dict, arm: str) -> list[dict[str, str]]:
    developer = (
        "You are a research assistant. Each retrieved page has a body and a Reference pointer. "
        "The pointer names another retrieved page cited by that page. For a claim-bearing page, it "
        "names the page on which its target-claim statement depends; NONE marks an independent claim "
        "root. A page that does not state the target claim may cite collection context and is not part "
        "of a claim path. Using only the retrieved "
        "bundle, answer the user's question and state how well established the answer is. "
        'Return JSON {"answer":"..."}.'
    )
    user = f"Question: {scenario['question']}\n\nRetrieved pages:\n{render_scenario(scenario, arm)}"
    return [{"role": "developer", "content": developer}, {"role": "user", "content": user}]


def root_messages(scenario: dict, arm: str) -> list[dict[str, str]]:
    developer = (
        "Trace the Reference pointers for the target claim. A claim root is a terminal page whose "
        "body states the target claim; ignore terminal pages whose body does not state the claim. "
        "List the distinct claim-root page IDs and their count. Return JSON only."
    )
    user = (
        f"Target claim: {scenario['claim_sentence']}\n\n"
        f"Retrieved pages:\n{render_scenario(scenario, arm)}"
    )
    return [{"role": "developer", "content": developer}, {"role": "user", "content": user}]


def judge_messages(scenario: dict, answer: str) -> list[dict[str, str]]:
    developer = (
        "Blind answer auditor. Classify how the answer presents the target claim. stance=established "
        "if it presents the value as established or independently corroborated; hedged if it gives "
        "the value but withholds that license; absent if it does not commit to the value. Separately, "
        "lineage_statement=single_lineage if it explicitly says the apparent support reduces to one "
        "origin or is not independent; multiple_independent if it explicitly says there are multiple "
        "independent origins; otherwise no_lineage_statement. Also mark target_value_status as "
        "target_value only when the answer commits to the supplied target value, other_value when it "
        "commits to a different value, and no_value otherwise. Return JSON only."
    )
    user = json.dumps(
        {"target_claim": scenario["claim_sentence"], "target_value": scenario["value"], "answer": answer},
        ensure_ascii=True,
    )
    return [{"role": "developer", "content": developer}, {"role": "user", "content": user}]


def parse_csv(value: str, allowed: tuple[str, ...] | None = None) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if allowed is not None:
        unknown = sorted(set(items) - set(allowed))
        if unknown:
            raise ValueError(f"unknown values {unknown}; allowed={allowed}")
    return items


def estimated_tokens(messages: list[dict[str, str]], schema: dict) -> int:
    # Transparent planning heuristic, not billed usage: roughly four UTF-8/ASCII characters/token,
    # including the response schema. The dry-run labels this explicitly.
    chars = sum(len(message["role"]) + len(message["content"]) for message in messages)
    chars += len(json.dumps(schema, ensure_ascii=True, sort_keys=True))
    return math.ceil(chars / 4)


def is_local_model(model: str) -> bool:
    return model.startswith(("qwen", "llama", "mistral", "gemma"))


def estimated_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    if is_local_model(model):
        return 0.0
    prices = PRICES_PER_MTOK.get(model)
    if prices is None:
        return None
    return (input_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000


def dry_run_plan(
    instrument: dict,
    models: list[str],
    tasks: list[str],
    judge_model: str,
    limit: int = 0,
    answer_output_tokens: int = 180,
    root_output_tokens: int = 120,
    judge_output_tokens: int = 24,
) -> dict:
    scenarios = instrument["scenarios"][:limit or None]
    answer_inputs = root_inputs = judge_inputs = 0
    for scenario in scenarios:
        for arm in ARMS:
            if "answer" in tasks:
                answer_inputs += estimated_tokens(answer_messages(scenario, arm), ANSWER_SCHEMA)
                placeholder_answer = "A" * (answer_output_tokens * 4)
                judge_inputs += estimated_tokens(judge_messages(scenario, placeholder_answer), JUDGE_SCHEMA)
            if "roots" in tasks:
                root_inputs += estimated_tokens(root_messages(scenario, arm), ROOT_SCHEMA)

    per_model = {}
    for model in models:
        answer_calls = len(scenarios) * len(ARMS) if "answer" in tasks else 0
        root_calls = len(scenarios) * len(ARMS) if "roots" in tasks else 0
        judge_calls = answer_calls
        model_in = answer_inputs + root_inputs
        model_out = answer_calls * answer_output_tokens + root_calls * root_output_tokens
        judge_out = judge_calls * judge_output_tokens
        model_cost = estimated_cost(model, model_in, model_out)
        judge_cost = estimated_cost(judge_model, judge_inputs, judge_out)
        total_cost = model_cost + judge_cost if model_cost is not None and judge_cost is not None else None
        per_model[model] = {
            "answerer_calls": answer_calls + root_calls,
            "answer_calls": answer_calls,
            "root_diagnostic_calls": root_calls,
            "judge_calls": judge_calls,
            "logical_calls_total": answer_calls + root_calls + judge_calls,
            "estimated_answerer_input_tokens": model_in,
            "assumed_answerer_output_tokens": model_out,
            "estimated_judge_input_tokens": judge_inputs,
            "assumed_judge_output_tokens": judge_out,
            "estimated_usd": round(total_cost, 4) if total_cost is not None else None,
            "answerer_usd": round(model_cost, 4) if model_cost is not None else None,
            "judge_usd": round(judge_cost, 4) if judge_cost is not None else None,
        }
    known_costs = [row["estimated_usd"] for row in per_model.values() if row["estimated_usd"] is not None]
    return {
        "mode": "dry_run_no_clients_created",
        "n_scenarios": len(scenarios),
        "arms": list(ARMS),
        "tasks": tasks,
        "models": models,
        "judge_model": judge_model,
        "per_model": per_model,
        "logical_calls_total": sum(row["logical_calls_total"] for row in per_model.values()),
        "estimated_usd_known_models_total": round(sum(known_costs), 4),
        "all_models_priced": len(known_costs) == len(per_model),
        "token_estimation": "ceil((message characters + schema characters)/4)",
        "output_token_assumptions": {
            "answer": answer_output_tokens,
            "roots": root_output_tokens,
            "judge": judge_output_tokens,
        },
        "prices_per_million_tokens": {
            model: {"input": prices[0], "output": prices[1]}
            for model, prices in PRICES_PER_MTOK.items()
            if model in set(models) | {judge_model}
        },
        "prices_as_of": PRICE_AS_OF,
        "warning": (
            "Planning estimate only; retries and reasoning tokens can increase billed usage. "
            "Local-model compute cost is not monetized."
        ),
    }


def usage(client) -> dict:
    return {
        "calls": getattr(client, "calls", None),
        "input_tokens": getattr(client, "input_tokens", None),
        "output_tokens": getattr(client, "output_tokens", None),
    }


def call_json(client, messages: list[dict[str, str]], schema: dict) -> tuple[dict | None, str | None]:
    if hasattr(client, "temperature"):
        client.temperature = 0.0
    try:
        parsed = client.complete_json(messages, response_schema=schema)
        if not isinstance(parsed, dict):
            return None, f"invalid structured response type: {type(parsed).__name__}"
        missing = [key for key in schema.get("required", []) if key not in parsed]
        if missing:
            return None, f"invalid structured response; missing required keys: {missing}"
        return parsed, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def score_root_response(parsed: dict | None, expected: list[str]) -> dict:
    """Score root IDs and the separately declared lineage count.

    OpenAI strict structured outputs reject the otherwise-standard ``uniqueItems`` keyword. Keep
    the API schema portable and enforce the same invariant here so duplicate IDs cannot pass.
    """
    raw_ids = (parsed or {}).get("claim_root_page_ids", [])
    raw_ids = raw_ids if isinstance(raw_ids, list) else []
    predicted = sorted(set(raw_ids))
    predicted_n = (parsed or {}).get("independent_claim_lineages")
    root_ids_unique = len(raw_ids) == len(predicted)
    exact_root_ids = root_ids_unique and predicted == expected
    lineage_count_correct = predicted_n == len(expected)
    return {
        "predicted_claim_root_ids": predicted,
        "root_ids_unique": root_ids_unique,
        "exact_root_ids": exact_root_ids,
        "lineage_count_correct": lineage_count_correct,
        "exact_roots": exact_root_ids and lineage_count_correct,
    }


def paired_bootstrap_ci(diffs: list[int], resamples: int = 10000) -> list[float] | None:
    if not diffs:
        return None
    rng = random.Random("graph-only-rewire:paired-bootstrap:v1")
    values = sorted(
        sum(rng.choice(diffs) for _ in diffs) / len(diffs)
        for _ in range(resamples)
    )
    return [
        round(values[int(0.025 * (resamples - 1))], 6),
        round(values[int(0.975 * (resamples - 1))], 6),
    ]


def summarize_rows(rows: list[dict]) -> dict:
    answer = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["task"] == "answer" and row["arm"] == arm]
        valid = [
            row for row in arm_rows
            if not row.get("answer_error") and not row.get("judge_error")
            and isinstance(row.get("judgment"), dict)
        ]
        stance = Counter(row["judgment"]["stance"] for row in valid)
        lineage = Counter(row["judgment"]["lineage_statement"] for row in valid)
        value_status = Counter(row["judgment"]["target_value_status"] for row in valid)
        n = len(valid)
        k = sum(
            row["judgment"]["stance"] == "established"
            and row["judgment"]["target_value_status"] == "target_value"
            for row in valid
        )
        answer[arm] = {
            "attempted_n": len(arm_rows),
            "valid_n": n,
            "established_n": f"{k}/{n}",
            "established_rate": round(k / n, 4) if n else None,
            "ci95": list(wilson(k, n)),
            "stance_dist": dict(stance),
            "lineage_statement_dist": dict(lineage),
            "target_value_status_dist": dict(value_status),
            "established_definition": "stance=established AND target_value_status=target_value",
        }
    answer_by_key = {
        (row["scenario_id"], row["arm"]): int(
            row["judgment"]["stance"] == "established"
            and row["judgment"]["target_value_status"] == "target_value"
        )
        for row in rows
        if row["task"] == "answer" and not row.get("answer_error") and not row.get("judge_error")
        and isinstance(row.get("judgment"), dict)
    }
    candidate_pair_ids = sorted({row["scenario_id"] for row in rows if row["task"] == "answer"})
    pair_ids = sorted(
        scenario_id
        for scenario_id in candidate_pair_ids
        if (scenario_id, "dependent") in answer_by_key and (scenario_id, "independent") in answer_by_key
    )
    paired_diffs = [
        answer_by_key[(scenario_id, "independent")] - answer_by_key[(scenario_id, "dependent")]
        for scenario_id in pair_ids
    ]
    independent_only = sum(diff == 1 for diff in paired_diffs)
    dependent_only = sum(diff == -1 for diff in paired_diffs)
    roots = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["task"] == "roots" and row["arm"] == arm]
        valid = [row for row in arm_rows if not row.get("answer_error")]
        exact_ids = sum(
            row.get("root_ids_unique", False)
            and row.get("predicted_claim_root_ids") == row.get("expected_claim_root_ids")
            for row in valid
        )
        count_correct = sum(
            isinstance(row.get("root_response"), dict)
            and row["root_response"].get("independent_claim_lineages")
            == len(row.get("expected_claim_root_ids", []))
            for row in valid
        )
        joint_exact = sum(row.get("exact_roots", False) for row in valid)
        roots[arm] = {
            "attempted_n": len(arm_rows),
            "valid_n": len(valid),
            "exact_root_id_set_n": f"{exact_ids}/{len(valid)}",
            "declared_lineage_count_correct_n": f"{count_correct}/{len(valid)}",
            "joint_root_ids_and_count_n": f"{joint_exact}/{len(valid)}",
            "joint_root_ids_and_count_rate": round(joint_exact / len(valid), 4) if valid else None,
            "joint_scoring_rule": (
                "unique root IDs AND exact root-ID set AND declared lineage count correct"
            ),
        }
    return {
        "answer": answer,
        "paired_discrimination": {
            "method": "mean_within_scenario_established_independent_minus_dependent",
            "candidate_pairs": len(candidate_pair_ids),
            "n_pairs": len(pair_ids),
            "excluded_pair_ids": sorted(set(candidate_pair_ids) - set(pair_ids)),
            "point": round(sum(paired_diffs) / len(paired_diffs), 6) if paired_diffs else None,
            "paired_bootstrap_ci95": paired_bootstrap_ci(paired_diffs),
            "bootstrap_resamples": 10000,
            "discordant_independent_only": independent_only,
            "discordant_dependent_only": dependent_only,
        },
        "root_diagnostic": roots,
    }


def run_model(
    instrument: dict,
    model: str,
    tasks: list[str],
    judge_model: str,
    out_dir: Path,
    limit: int,
    delay: float,
) -> dict:
    scenarios = instrument["scenarios"][:limit or None]
    model_client = client_for(model)
    judge_client = client_for(judge_model) if "answer" in tasks else None
    tag = safe_model(model)
    variant = variant_prefix(instrument)
    rows_path = out_dir / f"graph_only_rewire_{variant}{tag}.rows.jsonl"
    summary_path = out_dir / f"graph_only_rewire_{variant}{tag}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    seen = set()
    if rows_path.exists():
        for row in load_jsonl(rows_path):
            if row.get("instrument_sha256") != instrument["instrument_sha256"]:
                raise ValueError(f"existing {rows_path} belongs to a different instrument")
            if row.get("model") != model:
                raise ValueError(f"existing {rows_path} contains a row for model {row.get('model')!r}")
            if row.get("task") not in TASKS or row.get("arm") not in ARMS:
                raise ValueError(f"invalid existing row task/arm: {row.get('task')}/{row.get('arm')}")
            key = (row["task"], row["scenario_id"], row["arm"])
            if key in seen:
                raise ValueError(f"duplicate existing row key: {key}")
            seen.add(key)
            rows.append(row)

    with rows_path.open("a") as handle:
        for index, scenario in enumerate(scenarios):
            arm_order = ARMS if index % 2 == 0 else tuple(reversed(ARMS))
            for task in tasks:
                for arm in arm_order:
                    key = (task, scenario["id"], arm)
                    if key in seen:
                        continue
                    row = {
                        "instrument_sha256": instrument["instrument_sha256"],
                        "model": model,
                        "task": task,
                        "scenario_id": scenario["id"],
                        "source_scenario_id": scenario["source_scenario_id"],
                        "arm": arm,
                    }
                    if task == "answer":
                        parsed, answer_error = call_json(
                            model_client, answer_messages(scenario, arm), ANSWER_SCHEMA
                        )
                        answer = str((parsed or {}).get("answer", "")).strip()
                        row.update({"answer": answer, "answer_error": answer_error})
                        if answer_error:
                            row.update({"judgment": None, "judge_error": "answer call failed"})
                        else:
                            judgment, judge_error = call_json(
                                judge_client, judge_messages(scenario, answer), JUDGE_SCHEMA
                            )
                            row.update({"judgment": judgment, "judge_error": judge_error})
                    else:
                        parsed, answer_error = call_json(
                            model_client, root_messages(scenario, arm), ROOT_SCHEMA
                        )
                        expected = scenario["arms"][arm]["expected_claim_root_ids"]
                        row.update(
                            {
                                "root_response": parsed,
                                "answer_error": answer_error,
                                "expected_claim_root_ids": expected,
                                **score_root_response(parsed, expected),
                            }
                        )
                    handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    rows.append(row)
                    seen.add(key)
                    if delay:
                        time.sleep(delay)

    summary = {
        "config": {
            "model": model,
            "judge_model": judge_model if "answer" in tasks else None,
            "tasks": tasks,
            "n": len(scenarios),
            "arms": list(ARMS),
            "origin_register": instrument.get("origin_register", "registry"),
            "instrument_sha256": instrument["instrument_sha256"],
            "arm_call_order": "alternating by scenario index",
            "persisted_tasks": sorted({row["task"] for row in rows}),
            "page_order": "matched within scenario; cyclically counterbalanced across scenarios",
        },
        "results": summarize_rows(rows),
        "usage_this_invocation": {
            "answerer": usage(model_client),
            "judge": usage(judge_client) if judge_client else None,
        },
        "rows": str(rows_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    return summary


def load_instrument(path: Path) -> dict:
    instrument = json.loads(path.read_text())
    validate_instrument(instrument)
    return instrument


def add_common_plan_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--instrument", type=Path, default=DEFAULT_INSTRUMENT)
    parser.add_argument("--models", default="gpt-4o-mini,claude-opus-4-8")
    parser.add_argument("--tasks", default="answer,roots")
    parser.add_argument("--judge-model", default="gpt-4.1-mini")
    parser.add_argument("--limit", type=int, default=0, help="0 = all instrument scenarios")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="build/validate instrument; no model calls")
    build_parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    build_parser.add_argument("--output", type=Path, default=None)
    build_parser.add_argument("--limit", type=int, default=0)
    build_parser.add_argument("--register", choices=REGISTERS, default="registry")

    dry_parser = subparsers.add_parser("dry-run", help="validate and estimate calls/cost; no clients")
    add_common_plan_args(dry_parser)
    dry_parser.add_argument("--answer-output-tokens", type=int, default=180)
    dry_parser.add_argument("--root-output-tokens", type=int, default=120)
    dry_parser.add_argument("--judge-output-tokens", type=int, default=24)

    run_parser = subparsers.add_parser("run", help="execute model calls; requires explicit confirmation")
    add_common_plan_args(run_parser)
    run_parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULT_DIR)
    run_parser.add_argument("--delay", type=float, default=0.0)
    run_parser.add_argument("--confirm-api-spend", default="")

    args = parser.parse_args()
    if args.command == "build":
        output = args.output or (
            DEFAULT_INSTRUMENT if args.register == "registry" else CONFESSED_INSTRUMENT
        )
        instrument = build_instrument(args.source, args.limit, args.register)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(instrument, indent=2, ensure_ascii=True) + "\n")
        report = validate_instrument(instrument)
        print(json.dumps({"wrote": str(output), "origin_register": args.register, **report}, indent=2))
        return

    instrument = load_instrument(args.instrument)
    models = parse_csv(args.models)
    tasks = parse_csv(args.tasks, TASKS)
    if args.command == "dry-run":
        plan = dry_run_plan(
            instrument,
            models,
            tasks,
            args.judge_model,
            args.limit,
            args.answer_output_tokens,
            args.root_output_tokens,
            args.judge_output_tokens,
        )
        print(json.dumps(plan, indent=2, ensure_ascii=True))
        return

    if args.confirm_api_spend != CONFIRMATION:
        raise SystemExit(
            f"Refusing to create model clients. Re-run only after reviewing dry-run, with "
            f"--confirm-api-spend {CONFIRMATION}"
        )
    summaries = [
        run_model(instrument, model, tasks, args.judge_model, args.out_dir, args.limit, args.delay)
        for model in models
    ]
    print(json.dumps({"completed_models": [row["config"]["model"] for row in summaries]}, indent=2))


if __name__ == "__main__":
    main()
