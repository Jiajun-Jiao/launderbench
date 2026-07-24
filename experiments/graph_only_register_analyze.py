#!/usr/bin/env python3
"""Offline register x edges factorial report for the graph-only probe.

Rows: origin register (registry entries vs self-described unverified personal notes on the
symmetric P2/P3 root pages). Columns: citation edges (dependent vs independent). The registry row
is the original probe; this script joins it with the confessed row and reports, per model:

- production-judge established counts in all four cells;
- the edge effect within each register row (paired diff, bootstrap CI, exact McNemar);
- the register effect within each arm column (registry minus confessed, same statistics);
- integrity (row counts, zero errors, instrument hashes) and frozen-price cost for the new row.

No model calls; reads persisted rows only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (str(ROOT), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from graph_only_rewire import (  # noqa: E402
    ARMS,
    CONFESSED_INSTRUMENT,
    DEFAULT_INSTRUMENT,
    PRICES_PER_MTOK,
    load_instrument,
    paired_bootstrap_ci,
    safe_model,
)
from graph_only_rejudge import exact_mcnemar  # noqa: E402
from research_citation_multi import wilson  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "society"
REGISTERS = ("registry", "confessed")
MODELS = (
    "gpt-4o-mini",
    "gemini-2.5-flash",
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "gpt-4.1",
    "claude-opus-4-8",
    "o3",
    "gpt-5.5",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_stats(diffs: list[int]) -> dict:
    positive_only = sum(diff == 1 for diff in diffs)
    negative_only = sum(diff == -1 for diff in diffs)
    return {
        "n_pairs": len(diffs),
        "point": round(sum(diffs) / len(diffs), 6) if diffs else None,
        "paired_bootstrap_ci95": paired_bootstrap_ci(diffs),
        "bootstrap_resamples": 10000,
        "discordant_positive_only": positive_only,
        "discordant_negative_only": negative_only,
        "mcnemar_exact_two_sided_p": exact_mcnemar(positive_only, negative_only),
    }


def load_model_register(model: str, register: str, output_dir: Path, instrument: dict) -> dict:
    tag = safe_model(model)
    variant = "" if register == "registry" else f"{register}_"
    raw_rows_path = output_dir / f"graph_only_rewire_{variant}{tag}.rows.jsonl"
    raw_summary_path = output_dir / f"graph_only_rewire_{variant}{tag}.json"
    production_rows_path = output_dir / f"graph_only_rejudge_production_{variant}{tag}.rows.jsonl"
    production_summary_path = output_dir / f"graph_only_rejudge_production_{variant}{tag}.json"
    raw_rows = read_jsonl(raw_rows_path)
    production_rows = read_jsonl(production_rows_path)
    expected_keys = {(scenario["id"], arm) for scenario in instrument["scenarios"] for arm in ARMS}

    if any(row.get("instrument_sha256") != instrument["instrument_sha256"] for row in raw_rows):
        raise ValueError(f"{model}/{register}: raw instrument hash mismatch")
    if any(row.get("instrument_sha256") != instrument["instrument_sha256"] for row in production_rows):
        raise ValueError(f"{model}/{register}: production instrument hash mismatch")
    # Provider safety refusals are excluded pairwise and reported, never resampled or imputed.
    # Any other error type still fails the analysis.
    error_rows = [row for row in raw_rows if row.get("answer_error") or row.get("judge_error")]
    non_refusal = [
        row for row in error_rows if "refusal" not in str(row.get("answer_error", "")).lower()
    ]
    if non_refusal:
        raise ValueError(f"{model}/{register}: non-refusal error rows present")
    refused_scenarios = sorted({row["scenario_id"] for row in error_rows})
    refused_keys = {(sid, arm) for sid in refused_scenarios for arm in ARMS}
    if any(row.get("judge_error") for row in production_rows):
        raise ValueError(f"{model}/{register}: production error rows present")
    answer_rows = {
        (r["scenario_id"], r["arm"]): r
        for r in raw_rows
        if r["task"] == "answer" and not r.get("answer_error")
    }
    root_rows = {
        (r["scenario_id"], r["arm"]): r
        for r in raw_rows
        if r["task"] == "roots" and not r.get("answer_error")
    }
    production = {(r["scenario_id"], r["arm"]): r for r in production_rows}
    valid_keys = expected_keys - refused_keys
    if set(answer_rows) != valid_keys or set(root_rows) != valid_keys or set(production) != valid_keys:
        raise ValueError(f"{model}/{register}: incomplete key coverage")
    expected_keys = valid_keys

    established = {
        key: int(row["production_presented"] == "established_fact") for key, row in production.items()
    }
    factorized = {
        key: int(
            row["judgment"]["stance"] == "established"
            and row["judgment"]["target_value_status"] == "target_value"
        )
        for key, row in answer_rows.items()
    }
    agreement = sum(established[key] == factorized[key] for key in expected_keys)
    cells = {}
    for arm in ARMS:
        k = sum(established[(sid, a)] for (sid, a) in expected_keys if a == arm)
        n = sum(1 for (_, a) in expected_keys if a == arm)
        cells[arm] = {
            "established_n": f"{k}/{n}",
            "established_rate": round(k / n, 4),
            "ci95": list(wilson(k, n)),
            "production_label_dist": dict(
                Counter(production[key]["production_presented"] for key in expected_keys if key[1] == arm)
            ),
        }
    roots = {}
    for arm in ARMS:
        rows = [root_rows[key] for key in expected_keys if key[1] == arm]
        roots[arm] = {
            "exact_root_id_set_n": sum(
                row["root_ids_unique"]
                and row["predicted_claim_root_ids"] == row["expected_claim_root_ids"]
                for row in rows
            ),
            "joint_root_ids_and_count_n": sum(row["exact_roots"] for row in rows),
        }
    scenario_ids = sorted({sid for (sid, _) in expected_keys})
    edge_diffs = [
        established[(sid, "independent")] - established[(sid, "dependent")] for sid in scenario_ids
    ]
    raw_summary = json.loads(raw_summary_path.read_text())
    production_summary = json.loads(production_summary_path.read_text())
    prices = PRICES_PER_MTOK.get(model)
    answer_usage = raw_summary["usage_this_invocation"]["answerer"]
    judge_usage = raw_summary["usage_this_invocation"]["judge"]
    judge_prices = PRICES_PER_MTOK["gpt-4.1-mini"]
    main_cost = None
    if prices and answer_usage["input_tokens"] is not None:
        main_cost = (
            answer_usage["input_tokens"] * prices[0]
            + answer_usage["output_tokens"] * prices[1]
            + judge_usage["input_tokens"] * judge_prices[0]
            + judge_usage["output_tokens"] * judge_prices[1]
        ) / 1_000_000
    rejudge_cost = production_summary.get("actual_cost_usd_frozen_prices")
    return {
        "established": established,
        "report": {
            "cells_production": cells,
            "edge_effect_production": paired_stats(edge_diffs),
            "binary_cross_judge_agreement": {
                "agreement_n": agreement,
                "total_n": len(expected_keys),
                "agreement_rate": round(agreement / len(expected_keys), 6),
            },
            "root_diagnostic": roots,
            "integrity": {
                "raw_rows": len(raw_rows),
                "valid_answer_rows": len(answer_rows),
                "production_rows": len(production_rows),
                "refusal_rows": len(error_rows),
                "refused_scenario_ids": refused_scenarios,
                "refusal_policy": (
                    "provider safety refusals excluded pairwise across both arms and reported; "
                    "never resampled or imputed"
                ),
                "instrument_sha256": instrument["instrument_sha256"],
                "file_sha256": {
                    str(raw_rows_path): sha256(raw_rows_path),
                    str(production_rows_path): sha256(production_rows_path),
                },
            },
            "measured_cost_usd_frozen_prices": {
                "main_run": None if main_cost is None else round(main_cost, 6),
                "production_rejudge": rejudge_cost,
                "combined": (
                    None
                    if main_cost is None or rejudge_cost is None
                    else round(main_cost + rejudge_cost, 6)
                ),
            },
            "usage": {
                "main_run": raw_summary["usage_this_invocation"],
                "production_rejudge": production_summary["usage_this_invocation"],
            },
        },
    }


def analyze_model(model: str, output_dir: Path, instruments: dict[str, dict]) -> dict:
    loaded = {
        register: load_model_register(model, register, output_dir, instruments[register])
        for register in REGISTERS
    }
    register_effects = {}
    for arm in ARMS:
        scenario_ids = sorted(
            {sid for (sid, a) in loaded["registry"]["established"] if a == arm}
            & {sid for (sid, a) in loaded["confessed"]["established"] if a == arm}
        )
        diffs = [
            loaded["registry"]["established"][(sid, arm)]
            - loaded["confessed"]["established"][(sid, arm)]
            for sid in scenario_ids
        ]
        register_effects[arm] = paired_stats(diffs)
    return {
        "registry": loaded["registry"]["report"],
        "confessed": loaded["confessed"]["report"],
        "register_effect_production": {
            "method": "mean_within_scenario_established_registry_minus_confessed",
            **{arm: register_effects[arm] for arm in ARMS},
        },
    }


def markdown(report: dict) -> str:
    lines = [
        "# Register x edges factorial (graph-only probe)",
        "",
        "Production-judge established counts per cell; effects are within-scenario paired means.",
        "",
        "| model | registry dep/ind | confessed dep/ind | edge eff (registry) | edge eff (confessed) | register eff (dep) | register eff (ind) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, data in report["models"].items():
        reg = data["registry"]["cells_production"]
        con = data["confessed"]["cells_production"]
        ee_r = data["registry"]["edge_effect_production"]
        ee_c = data["confessed"]["edge_effect_production"]
        re_d = data["register_effect_production"]["dependent"]
        re_i = data["register_effect_production"]["independent"]
        lines.append(
            f"| {model} | {reg['dependent']['established_n']}, {reg['independent']['established_n']} "
            f"| {con['dependent']['established_n']}, {con['independent']['established_n']} "
            f"| {ee_r['point']:+.2f} | {ee_c['point']:+.2f} "
            f"| {re_d['point']:+.2f} | {re_i['point']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "Register effect = registry minus confessed within the same arm (positive = the",
            "self-confessed origin register suppresses establishment). Edge effect = independent",
            "minus dependent within the same register.",
            "",
            f"Recorded confessed-row cost under the frozen price table: "
            f"US${report['confessed_recorded_cost_usd_priced_models']:.4f} over priced models"
            + (
                f" (unpriced, usage-only: {', '.join(report['unpriced_models'])})"
                if report["unpriced_models"]
                else ""
            )
            + ".",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--registry-instrument", type=Path, default=DEFAULT_INSTRUMENT)
    parser.add_argument("--confessed-instrument", type=Path, default=CONFESSED_INSTRUMENT)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_DIR / "graph_only_register_analysis.json"
    )
    parser.add_argument("--models", default=",".join(MODELS))
    args = parser.parse_args()
    instruments = {
        "registry": load_instrument(args.registry_instrument),
        "confessed": load_instrument(args.confessed_instrument),
    }
    if instruments["confessed"].get("origin_register") != "confessed":
        raise ValueError("confessed instrument does not declare origin_register=confessed")
    models = {
        model: analyze_model(model, args.output_dir, instruments)
        for model in [m.strip() for m in args.models.split(",") if m.strip()]
    }
    priced = [
        data["confessed"]["measured_cost_usd_frozen_prices"]["combined"]
        for data in models.values()
        if data["confessed"]["measured_cost_usd_frozen_prices"]["combined"] is not None
    ]
    report = {
        "schema_version": 1,
        "design": "origin register (registry vs confessed P2/P3) x citation edges (dependent vs independent)",
        "instrument_sha256": {
            register: instruments[register]["instrument_sha256"] for register in REGISTERS
        },
        "models": models,
        "confessed_recorded_cost_usd_priced_models": round(sum(priced), 6),
        "unpriced_models": sorted(
            model
            for model, data in models.items()
            if data["confessed"]["measured_cost_usd_frozen_prices"]["combined"] is None
        ),
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n")
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(markdown(report))
    print(f"wrote {args.output}")
    print(f"wrote {markdown_path}")


if __name__ == "__main__":
    main()
