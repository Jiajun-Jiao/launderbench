"""Run the E-NPC benchmark MVP and print an aggregate metrics table."""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

from baselines import default_controllers
from evaluator import aggregate_by_scenario, aggregate_results, evaluate_all, format_summary_table
from llm_controller import OpenAICompatibleChatClient, make_llm_controllers
from query_generator import generate_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the E-NPC benchmark MVP.")
    parser.add_argument("--worlds", type=int, default=50, help="number of worlds to generate")
    parser.add_argument(
        "--queries-per-world",
        type=int,
        default=20,
        help="number of queries generated per world",
    )
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="optional path for per-response evaluation records",
    )
    parser.add_argument(
        "--samples-json",
        type=Path,
        default=None,
        help="optional path for generated worlds and queries",
    )
    parser.add_argument(
        "--by-scenario",
        action="store_true",
        help="also print metrics split by scenario type",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="optional cap on generated queries before controller evaluation",
    )
    parser.add_argument(
        "--include-llm",
        action="store_true",
        help="include OpenAI-compatible LLM controllers",
    )
    parser.add_argument(
        "--llm-only",
        action="store_true",
        help="run only LLM controllers when --include-llm is set",
    )
    parser.add_argument(
        "--llm-model",
        default=os.environ.get("OPENAI_MODEL"),
        help="LLM model name; defaults to OPENAI_MODEL",
    )
    parser.add_argument(
        "--llm-base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--llm-api-key-env",
        default="OPENAI_API_KEY",
        help="environment variable containing the LLM API key",
    )
    parser.add_argument(
        "--llm-policies",
        default="global_state,local_state,source_calibrated",
        help="comma-separated LLM policies: global_state,local_state,source_calibrated",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=60.0,
        help="LLM request timeout in seconds",
    )
    parser.add_argument(
        "--llm-retries",
        type=int,
        default=2,
        help="number of LLM request retries",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.0,
        help="LLM sampling temperature",
    )
    parser.add_argument(
        "--llm-no-structured-output",
        action="store_true",
        help="disable JSON schema response_format for OpenAI-compatible servers that do not support it",
    )
    args = parser.parse_args()

    worlds, queries = generate_dataset(
        num_worlds=args.worlds,
        queries_per_world=args.queries_per_world,
        seed=args.seed,
    )
    if args.max_queries is not None:
        queries = queries[: args.max_queries]

    controllers = []
    if not args.llm_only:
        controllers.extend(default_controllers())
    if args.include_llm:
        if not args.llm_model:
            parser.error("--include-llm requires --llm-model or OPENAI_MODEL")
        api_key = os.environ.get(args.llm_api_key_env)
        if _is_openai_url(args.llm_base_url) and not api_key:
            parser.error(f"--include-llm requires {args.llm_api_key_env} for OpenAI API")
        policies = [
            policy.strip()
            for policy in args.llm_policies.split(",")
            if policy.strip()
        ]
        client = OpenAICompatibleChatClient(
            model=args.llm_model,
            api_key=api_key,
            base_url=args.llm_base_url,
            timeout=args.llm_timeout,
            max_retries=args.llm_retries,
            temperature=args.llm_temperature,
            use_structured_output=not args.llm_no_structured_output,
        )
        controllers.extend(make_llm_controllers(client, policies=policies))
    if not controllers:
        parser.error("no controllers selected")

    records = evaluate_all(worlds, queries, controllers)
    summary = aggregate_results(records)

    print(
        f"Generated {len(worlds)} worlds, {len(queries)} queries, "
        f"{len(records)} controller responses."
    )
    print()
    print(format_summary_table(summary))

    if args.by_scenario:
        print()
        for scenario, scenario_summary in sorted(aggregate_by_scenario(records).items()):
            print(f"[{scenario}]")
            print(format_summary_table(scenario_summary))
            print()

    if args.jsonl:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=True) + "\n")

    if args.samples_json:
        args.samples_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "worlds": [world.to_dict() for world in worlds],
            "queries": [query.to_dict() for query in queries],
        }
        args.samples_json.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    return 0


def _is_openai_url(base_url: str) -> bool:
    return "api.openai.com" in base_url


if __name__ == "__main__":
    raise SystemExit(main())
