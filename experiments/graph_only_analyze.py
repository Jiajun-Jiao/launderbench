#!/usr/bin/env python3
"""Offline integrity/statistics report for the completed graph-only experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

try:
    from graph_only_rewire import ARMS, DEFAULT_INSTRUMENT, PRICES_PER_MTOK, load_instrument
except ModuleNotFoundError:  # pragma: no cover
    from .graph_only_rewire import ARMS, DEFAULT_INSTRUMENT, PRICES_PER_MTOK, load_instrument


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "society"
MODELS = ("gpt-4o-mini", "claude-opus-4-8")
RESAMPLES = 10000
SEED = 20260710


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile_ci(values: list[float]) -> list[float]:
    values = sorted(values)
    return [
        round(values[int(0.025 * (len(values) - 1))], 6),
        round(values[int(0.975 * (len(values) - 1))], 6),
    ]


def domain_cluster_ci(diffs_by_domain: dict[str, list[int]], seed: str) -> list[float]:
    domains = sorted(diffs_by_domain)
    rng = random.Random(seed)
    values = []
    for _ in range(RESAMPLES):
        picked = [rng.choice(domains) for _ in domains]
        diffs = [diff for domain in picked for diff in diffs_by_domain[domain]]
        values.append(sum(diffs) / len(diffs))
    return percentile_ci(values)


def actual_cost(usage: dict, model: str) -> float | None:
    """Frozen-price cost; None for models outside the frozen price table (never invented)."""
    prices = PRICES_PER_MTOK.get(model)
    if prices is None:
        return None
    return (usage["input_tokens"] * prices[0] + usage["output_tokens"] * prices[1]) / 1_000_000


def binary_original(row: dict) -> int:
    return int(
        row["judgment"]["stance"] == "established"
        and row["judgment"]["target_value_status"] == "target_value"
    )


def binary_production(row: dict) -> int:
    return int(row["production_presented"] == "established_fact")


def paired_breakdown(
    scenarios: dict[str, dict],
    source_scenarios: dict[str, dict],
    values: dict[tuple[str, str], int],
) -> dict:
    by_position = defaultdict(list)
    by_topology = defaultdict(list)
    by_domain = defaultdict(list)
    for scenario_id, scenario in scenarios.items():
        diff = values[(scenario_id, "independent")] - values[(scenario_id, "dependent")]
        by_position[scenario["page_order"][0]].append(diff)
        source = source_scenarios[scenario["source_scenario_id"]]
        by_topology[source["topology"]].append(diff)
        by_domain[scenario["domain"]].append(diff)

    def summarize(groups: dict[str, list[int]]) -> dict:
        return {
            key: {"n": len(diffs), "point": round(sum(diffs) / len(diffs), 6)}
            for key, diffs in sorted(groups.items())
        }

    domain_points = [sum(diffs) / len(diffs) for diffs in by_domain.values()]
    return {
        "by_first_page": summarize(by_position),
        "by_source_topology": summarize(by_topology),
        "by_domain": summarize(by_domain),
        "domain_point_range": [round(min(domain_points), 6), round(max(domain_points), 6)],
        "domains_positive": sum(point > 0 for point in domain_points),
        "domains_total": len(domain_points),
        "domain_cluster_bootstrap_ci95": domain_cluster_ci(
            by_domain, f"{SEED}:domain-cluster:{sorted(values.items())}"
        ),
        "domain_cluster_bootstrap_resamples": RESAMPLES,
    }


def analyze_model(
    model: str,
    output_dir: Path,
    instrument: dict,
    source_scenarios: dict[str, dict],
) -> dict:
    raw_rows_path = output_dir / f"graph_only_rewire_{model}.rows.jsonl"
    raw_summary_path = output_dir / f"graph_only_rewire_{model}.json"
    production_rows_path = output_dir / f"graph_only_rejudge_production_{model}.rows.jsonl"
    production_summary_path = output_dir / f"graph_only_rejudge_production_{model}.json"
    raw_rows = read_jsonl(raw_rows_path)
    raw_summary = json.loads(raw_summary_path.read_text())
    production_rows = read_jsonl(production_rows_path)
    production_summary = json.loads(production_summary_path.read_text())

    raw_keys = [(row["task"], row["scenario_id"], row["arm"]) for row in raw_rows]
    production_keys = [(row["scenario_id"], row["arm"]) for row in production_rows]
    answer_rows = {
        (row["scenario_id"], row["arm"]): row
        for row in raw_rows
        if row["task"] == "answer"
    }
    root_rows = {
        (row["scenario_id"], row["arm"]): row
        for row in raw_rows
        if row["task"] == "roots"
    }
    production_by_key = {
        (row["scenario_id"], row["arm"]): row for row in production_rows
    }
    expected_answer_keys = {
        (scenario["id"], arm) for scenario in instrument["scenarios"] for arm in ARMS
    }
    if set(answer_rows) != expected_answer_keys or set(root_rows) != expected_answer_keys:
        raise ValueError(f"{model}: incomplete answer/root key coverage")
    if set(production_by_key) != expected_answer_keys:
        raise ValueError(f"{model}: incomplete production-rejudge key coverage")
    if len(raw_keys) != len(set(raw_keys)) or len(production_keys) != len(set(production_keys)):
        raise ValueError(f"{model}: duplicate result keys")
    if any(row.get("answer_error") or row.get("judge_error") for row in raw_rows):
        raise ValueError(f"{model}: raw result error rows present")
    if any(row.get("judge_error") for row in production_rows):
        raise ValueError(f"{model}: production-rejudge error rows present")
    if any(row["instrument_sha256"] != instrument["instrument_sha256"] for row in raw_rows):
        raise ValueError(f"{model}: raw instrument hash mismatch")
    if any(row["instrument_sha256"] != instrument["instrument_sha256"] for row in production_rows):
        raise ValueError(f"{model}: production instrument hash mismatch")

    original_values = {key: binary_original(row) for key, row in answer_rows.items()}
    production_values = {key: binary_production(row) for key, row in production_by_key.items()}
    agreement = Counter(
        (original_values[key], production_values[key]) for key in sorted(expected_answer_keys)
    )
    root_detail = {}
    for arm in ARMS:
        rows = [root_rows[(scenario["id"], arm)] for scenario in instrument["scenarios"]]
        expected_n = 1 if arm == "dependent" else 2
        root_detail[arm] = {
            "attempted_n": len(rows),
            "exact_root_id_set_n": sum(
                row["root_ids_unique"]
                and row["predicted_claim_root_ids"] == row["expected_claim_root_ids"]
                for row in rows
            ),
            "declared_lineage_count_correct_n": sum(
                row["root_response"]["independent_claim_lineages"] == expected_n for row in rows
            ),
            "joint_root_ids_and_count_n": sum(row["exact_roots"] for row in rows),
            "joint_scoring_rule": (
                "unique root IDs AND exact root-ID set AND declared lineage count correct"
            ),
            "contains_all_gold_roots_n": sum(
                set(row["expected_claim_root_ids"]).issubset(row["predicted_claim_root_ids"])
                for row in rows
            ),
            "duplicate_id_rows": sum(not row["root_ids_unique"] for row in rows),
            "predicted_set_size_dist": dict(
                Counter(str(len(row["predicted_claim_root_ids"])) for row in rows)
            ),
        }

    raw_usage = raw_summary["usage_this_invocation"]
    answerer_cost = actual_cost(raw_usage["answerer"], model)
    raw_judge_cost = actual_cost(raw_usage["judge"], "gpt-4.1-mini")
    raw_cost = None if answerer_cost is None else answerer_cost + raw_judge_cost
    production_cost = production_summary["actual_cost_usd_frozen_prices"]
    scenarios = {scenario["id"]: scenario for scenario in instrument["scenarios"]}
    return {
        "integrity": {
            "raw_rows": len(raw_rows),
            "raw_unique_keys": len(set(raw_keys)),
            "production_rejudge_rows": len(production_rows),
            "production_unique_keys": len(set(production_keys)),
            "error_rows": 0,
            "instrument_sha256": instrument["instrument_sha256"],
            "file_sha256": {
                str(raw_rows_path): sha256(raw_rows_path),
                str(raw_summary_path): sha256(raw_summary_path),
                str(production_rows_path): sha256(production_rows_path),
                str(production_summary_path): sha256(production_summary_path),
            },
        },
        "factorized_graph_judge": raw_summary["results"],
        "production_judge": production_summary["results"],
        "binary_cross_judge": {
            "agreement_n": sum(n for (a, b), n in agreement.items() if a == b),
            "total_n": sum(agreement.values()),
            "agreement_rate": round(
                sum(n for (a, b), n in agreement.items() if a == b) / sum(agreement.values()), 6
            ),
            "confusion_original_production": {
                f"{original}->{production}": n
                for (original, production), n in sorted(agreement.items())
            },
        },
        "root_diagnostic_detail": root_detail,
        "factorized_breakdown": paired_breakdown(
            scenarios, source_scenarios, original_values
        ),
        "production_breakdown": paired_breakdown(
            scenarios, source_scenarios, production_values
        ),
        "usage": {
            "main_run": raw_usage,
            "production_rejudge": production_summary["usage_this_invocation"],
        },
        "measured_cost_usd_frozen_prices": {
            "main_run": None if raw_cost is None else round(raw_cost, 6),
            "production_rejudge": production_cost,
            "combined": (
                None
                if raw_cost is None or production_cost is None
                else round(raw_cost + production_cost, 6)
            ),
            "answerer_unpriced_judge_cost_usd": (
                None if raw_cost is not None else round(raw_judge_cost, 6)
            ),
            "prices_as_of": "2026-07-07",
        },
    }


def markdown(report: dict) -> str:
    lines = [
        "# Graph-only rewiring analysis",
        "",
        f"Instrument: `{report['instrument_sha256']}`; 100 matched scenarios; 10,000 paired/domain-cluster bootstraps.",
        "",
        "| model | factorized dep/ind | factorized paired diff [95% CI] | production dep/ind | production paired diff [95% CI] | root IDs + count jointly exact dep/ind |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, data in report["models"].items():
        factor = data["factorized_graph_judge"]
        prod = data["production_judge"]
        fdep = factor["answer"]["dependent"]["established_rate"]
        find = factor["answer"]["independent"]["established_rate"]
        fp = factor["paired_discrimination"]
        pdep = prod["answer"]["dependent"]["established_rate"]
        pind = prod["answer"]["independent"]["established_rate"]
        pp = prod["paired_discrimination"]
        root = data["root_diagnostic_detail"]
        lines.append(
            f"| {model} | {fdep:.2f}/{find:.2f} | {fp['point']:+.2f} {fp['paired_bootstrap_ci95']} | "
            f"{pdep:.2f}/{pind:.2f} | {pp['point']:+.2f} {pp['paired_bootstrap_ci95']} | "
            f"{root['dependent']['joint_root_ids_and_count_n']}/100, "
            f"{root['independent']['joint_root_ids_and_count_n']}/100 |"
        )
    lines.extend(
        [
            "",
            "`factorized` is the runner's separate stance/lineage rubric. `production` is the frozen canonical-panel rubric. Both are blind to arm and expected graph.",
            "",
            "`root IDs + count jointly exact` requires unique, exact root IDs and a correct separately declared lineage count.",
            "",
            f"Recorded successful main-run + production-rejudge cost under the frozen price table: US${report['recorded_cost_usd_priced_models']:.4f} over priced models"
            + (
                f" (unpriced, usage-only: {', '.join(report['unpriced_models'])})"
                if report["unpriced_models"]
                else ""
            )
            + ". This excludes the discarded compatibility-failure partial run, whose successful-call usage was not persisted.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument", type=Path, default=DEFAULT_INSTRUMENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--source-scenarios",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "citation_scenarios_100_v2.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "graph_only_rewire_analysis.json",
    )
    parser.add_argument("--models", default=",".join(MODELS))
    args = parser.parse_args()
    instrument = load_instrument(args.instrument)
    source_scenarios = {
        scenario["id"]: scenario for scenario in read_jsonl(args.source_scenarios)
    }
    models = {
        model: analyze_model(model, args.output_dir, instrument, source_scenarios)
        for model in [m.strip() for m in args.models.split(",") if m.strip()]
    }
    report = {
        "schema_version": 1,
        "instrument_sha256": instrument["instrument_sha256"],
        "bootstrap_seed": SEED,
        "bootstrap_resamples": RESAMPLES,
        "models": models,
        "recorded_cost_usd_priced_models": round(
            sum(
                data["measured_cost_usd_frozen_prices"]["combined"]
                for data in models.values()
                if data["measured_cost_usd_frozen_prices"]["combined"] is not None
            ),
            6,
        ),
        "unpriced_models": sorted(
            model
            for model, data in models.items()
            if data["measured_cost_usd_frozen_prices"]["combined"] is None
        ),
        "cost_boundary": (
            "Recorded successful main run and production rejudge only; discarded partial-run "
            "successful calls were not persisted and invalid HTTP-400 schema requests are assumed unbilled."
        ),
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n")
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(markdown(report))
    print(f"wrote {args.output}")
    print(f"wrote {markdown_path}")


if __name__ == "__main__":
    main()
