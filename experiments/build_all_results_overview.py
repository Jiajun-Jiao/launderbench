#!/usr/bin/env python3
"""Build a navigation-level summary across current LaunderBench artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "society"
HUMAN_DIR = ROOT / "outputs" / "human_eval"

EXPECTED_LABELS = [
    "established_fact",
    "hedged",
    "traced_weak_origin",
    "debunked",
    "absent",
]

SOURCE_FILES = {
    "canonical100": "outputs/society/canonical100_extension_summary.md",
    "agent_frameworks": "outputs/society/agent_framework_extension_summary.md",
    "ecological_validity": "outputs/society/ecological_validity_summary.md",
    "harness_validity": "outputs/society/harness_validity_rows.md",
    "external_extension": "outputs/society/external_extension_summary.md",
    "external100_extension": "outputs/society/external100_extension_summary.md",
    "open_weight": "outputs/society/open_model_summary.md",
    "human_judge_panel": "outputs/human_eval/judge_panel_summary.md",
    "typed_handoff_control": "outputs/society/prov_handoff_condition_summary_20260710.md",
    "prose_handoff_control": "outputs/society/prov_handoff_prose_condition_summary_20260710.md",
    "scenario_construct_audit": "outputs/human_eval/scenario_audit_summary.json",
}


def read_json(rel_path: str) -> dict[str, Any]:
    path = ROOT / rel_path
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def read_first_json(*rel_paths: str) -> dict[str, Any]:
    for rel_path in rel_paths:
        path = ROOT / rel_path
        if path.exists():
            return json.loads(path.read_text())
    return {}


def read_external_summary() -> dict[str, Any]:
    hundred = read_json("outputs/society/external100_extension_summary.json")
    if hundred.get("citation"):
        return hundred
    return read_json("outputs/society/external_extension_summary.json")


def clean_judge_panel(raw: dict[str, Any]) -> dict[str, Any]:
    clean_distributions: dict[str, dict[str, int]] = {}
    for judge, dist in raw.get("judge_distributions", {}).items():
        clean = {label: int(dist.get(label, 0)) for label in EXPECTED_LABELS}
        error_count = sum(int(v) for k, v in dist.items() if k not in EXPECTED_LABELS)
        if error_count:
            clean["errors"] = error_count
        clean_distributions[judge] = clean

    return {
        "judges": raw.get("judges", []),
        "n_items": raw.get("n_items", 0),
        "judge_distributions": clean_distributions,
        "pairwise": raw.get("pairwise", []),
        "majority_distribution": raw.get("majority_distribution", {}),
        "by_source": raw.get("by_source", []),
        "disagreement_sample_count": len(raw.get("disagreements", [])),
    }


def rate_cell(rate: Any, n: Any) -> str:
    if rate is None:
        return "-"
    return f"{float(rate):.2f} ({n})"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---" for _ in headers]) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def native_extra(row: dict[str, Any]) -> str:
    if "exact_match" in row:
        return f"exact={row['exact_match']:.2f}"
    if "unsupported_recall" in row:
        return (
            f"recall={row['unsupported_recall']:.2f}, "
            f"specificity={row['supported_specificity']:.2f}"
        )
    return "-"


def open_weight_read(row: dict[str, Any]) -> str:
    if row.get("solid") == "30/30" and row.get("flagged") == "0/30":
        return "valid discriminator; trace laundering remains"
    return "adherence/solid-control confounded"


def build_markdown(overview: dict[str, Any]) -> str:
    canonical = overview["main_canonical100"]
    agent = overview["agent_frameworks"]
    external = overview["external_benchmark_extensions"]
    open_weight = overview["open_weight_appendix"]
    human = overview["human_judge_panel"]
    handoff = overview["handoff_controls"]
    construct = overview["scenario_construct_audit"]

    lines: list[str] = []
    lines.append("# All Results Overview")
    lines.append("")
    lines.append(
        "This is a navigation-level summary of the current LaunderBench / "
        "source-dependence-laundering experiment state. It links to the detailed run "
        "artifacts rather than replacing them."
    )
    lines.append("")
    lines.append("## Terminology")
    lines.append("")
    lines.extend(
        [
            "- `trace`: unflagged weak-origin condition; the citation chain is present, but not explicitly warned.",
            "- `flagged`: same weak-origin condition with an explicit warning.",
            "- `solid`: positive-control condition with independent strong support.",
            "- `void`: agentic condition where the origin is not reachable.",
            "- `reachable`: agentic condition where the origin can be found only if the agent asks for it.",
        ]
    )

    panel = canonical.get("panel", {})
    lines.append("")
    lines.append("## 1. Main Canonical100 Result")
    lines.append("")
    lines.append(
        "Panel: {rows} validated weak-origin scenarios; {solid_ok} validated solid controls. "
        "Solid missing: {missing}.".format(
            rows=panel.get("rows"),
            solid_ok=panel.get("solid_ok"),
            missing=", ".join(panel.get("solid_missing", [])) or "none",
        )
    )
    lines.append("")
    lines.append(
        md_table(
            [
                "model",
                "trace laundering",
                "flagged laundering",
                "solid established",
                "discrimination",
            ],
            [
                [
                    row["model"],
                    rate_cell(row["trace_launder"], row["trace_n"]),
                    rate_cell(row["flagged_launder"], row["flagged_n"]),
                    rate_cell(row["solid_established"], row["solid_n"]),
                    f"{row['discrimination']:.2f}",
                ]
                for row in canonical.get("models", [])
            ],
        )
    )
    lines.append("")
    lines.append(
        "Read: the split is specifically in unflagged provenance tracing. Explicit flags almost "
        "eliminate laundering, and solid controls are accepted, so the task is discriminative "
        "rather than just inducing blanket skepticism."
    )
    lines.append("")
    lines.append("Trace laundering by topology:")
    lines.append("")
    lines.append(
        md_table(
            ["model", "linear", "star", "fake_corroboration"],
            [
                [
                    row["model"],
                    row["trace_by_topology"]["linear"]["launder_n"],
                    row["trace_by_topology"]["star"]["launder_n"],
                    row["trace_by_topology"]["fake_corroboration"]["launder_n"],
                ]
                for row in canonical.get("models", [])
            ],
        )
    )

    lines.append("")
    lines.append("## 2. Framework And Agent Results")
    lines.append("")
    lines.append(
        md_table(
            [
                "framework / mode",
                "trace",
                "flagged",
                "solid",
                "void",
                "reachable",
                "search tool called",
                "origin retrieved",
            ],
            [
                [
                    row["label"],
                    row.get("trace", "-"),
                    row.get("flagged", "-"),
                    row.get("solid", "-"),
                    row.get("void", "-"),
                    row.get("reachable", "-"),
                    (
                        f"void {row.get('void_tool_called', '-')}/{row.get('void_n', '-')}; "
                        f"reach {row.get('reachable_tool_called', '-')}/{row.get('reachable_n', '-')}"
                        if "void_tool_called" in row
                        else "-"
                    ),
                    (
                        f"void {row.get('void_origin_retrieved_rows', '-')}/{row.get('void_n', '-')}; "
                        f"reach {row.get('reachable_origin_retrieved_rows', '-')}/{row.get('reachable_n', '-')}"
                        if "void_origin_retrieved_rows" in row
                        else "-"
                    ),
                ]
                for row in agent.get("rows", [])
            ],
        )
    )
    lines.append("")
    lines.append(
        "Read: framework wrappers do not automatically repair the failure. In complete-context RAG, "
        "gpt-4o-mini still launders nearly/all trace items. In agentic reachable-origin settings, "
        "tools are available but agents rarely retrieve the origin page."
    )

    lines.append("")
    lines.append("## 3. Harness Validity And Mechanism Checks")
    lines.append("")
    for item in overview["harness_validity"]["highlights"]:
        lines.append(f"- {item}")

    lines.append("")
    lines.append("Bidirectional handoff controls:")
    lines.append("")
    typed = handoff.get("typed", {}).get("carriers", {})
    prose = handoff.get("prose", {}).get("carriers", {})
    lines.append(
        md_table(
            ["carrier", "typed trace", "typed solid", "typed disc.", "prose trace", "prose solid"],
            [
                [
                    model,
                    f"{row['trace_established']}/{row['matched_n']}",
                    f"{row['solid_established']}/{row['matched_n']}",
                    f"{row['paired_discrimination']:.3f}",
                    f"{prose[model]['trace_established']}/{prose[model]['matched_n']}",
                    f"{prose[model]['solid_established']}/{prose[model]['matched_n']}",
                ]
                for model, row in typed.items()
            ],
        )
    )
    lines.append("")
    lines.append(
        "Read: with a correct upstream type, the structured record preserves all matched negative "
        "and positive licenses (0/384 weak, 384/384 solid). Explicit prose remains weak-safe but "
        "establishes 344/384 solids."
    )
    lines.append("")
    lines.append(
        f"Blinded construct audit: {construct.get('full_construct_majority_pass', 0)}/40 majority "
        f"passes; removing {len(construct.get('one_rater_dissent_scenario_ids', []))} items with "
        "any one-rater dissent changes no headline trace rate by more than .029."
    )

    lines.append("")
    lines.append("## 4. External Benchmark Extensions")
    lines.append("")
    lines.append("Seeded LaunderBench panels:")
    lines.append("")
    lines.append(
        md_table(
            ["benchmark", "model", "trace", "flagged", "solid", "discrimination"],
            [
                [
                    row["benchmark"],
                    row["model"],
                    row["trace"],
                    row["flagged"],
                    row["solid"],
                    f"{row['discrimination']:.2f}",
                ]
                for row in external.get("citation", [])
            ],
        )
    )
    lines.append("")
    lines.append("Type-once-carry handoff:")
    lines.append("")
    lines.append(
        md_table(
            [
                "benchmark",
                "Opus typer weak/circular",
                "mini self-downstream",
                "mini self-carry",
                "Opus typed-handoff",
                "mini own field correct",
            ],
            [
                [
                    row["benchmark"],
                    f"{row['typer_weak_circular']}/{row['n']}",
                    row["self_downstream"],
                    row["self_carry"],
                    row["typed_handoff"],
                    row["own_field_correct"],
                ]
                for row in external.get("handoff", [])
            ],
        )
    )
    lines.append("")
    lines.append("Original native-control panels:")
    lines.append("")
    native_rows = external.get("native", [])
    if any("render" in row for row in native_rows):
        lines.append(
            md_table(
                ["benchmark", "model", "render", "metric", "value", "extra"],
                [
                    [
                        row["benchmark"],
                        row["model"],
                        row.get("render", "plain"),
                        row["metric"],
                        f"{row['value']:.2f}",
                        native_extra(row),
                    ]
                    for row in native_rows
                ],
            )
        )
    else:
        lines.append(
            md_table(
                ["benchmark", "model", "metric", "value", "extra"],
                [
                    [
                        row["benchmark"],
                        row["model"],
                        row["metric"],
                        f"{row['value']:.2f}",
                        native_extra(row),
                    ]
                    for row in native_rows
                ],
            )
        )
    lines.append("")
    lines.append(
        "Read: the seeded ExpertQA/ALCE/RAGTruth panels reproduce the same direction: "
        "gpt-4o-mini launders many trace items; Claude Opus is near zero on trace while "
        "still accepting most solid controls."
    )

    lines.append("")
    lines.append("## 5. Open-Weight Appendix")
    lines.append("")
    if open_weight.get("canonical100"):
        lines.append("Canonical100 open-weight rows:")
        lines.append("")
        lines.append(
            md_table(
                ["model", "n", "trace", "flagged", "solid", "discrimination", "read"],
                [
                    [
                        row["model"],
                        row.get("n_scenarios", "-"),
                        row["trace"],
                        row["flagged"],
                        row["solid"],
                        f"{row['discrimination']:.2f}",
                        row.get("read") or open_weight_read(row),
                    ]
                    for row in open_weight.get("canonical100", [])
                ],
            )
        )
        lines.append("")
    if open_weight.get("small_canonical100"):
        lines.append("Smaller Canonical100 controls:")
        lines.append("")
        lines.append(
            md_table(
                ["model", "n", "trace", "flagged", "solid", "discrimination", "read"],
                [
                    [
                        row["model"],
                        row.get("n_scenarios", "-"),
                        row["trace"],
                        row["flagged"],
                        row["solid"],
                        f"{row['discrimination']:.2f}",
                        row.get("read") or open_weight_read(row),
                    ]
                    for row in open_weight.get("small_canonical100", [])
                ],
            )
        )
        lines.append("")
    legacy_rows = open_weight.get("canonical30_appendix") or open_weight.get("rows", [])
    if legacy_rows:
        lines.append("Canonical30 legacy appendix:")
        lines.append("")
        lines.append(
            md_table(
                ["model", "trace", "flagged", "solid", "discrimination", "read"],
                [
                    [
                        row["model"],
                        row["trace"],
                        row["flagged"],
                        row["solid"],
                        f"{row['discrimination']:.2f}",
                        row.get("read") or open_weight_read(row),
                    ]
                    for row in legacy_rows
                ],
            )
        )
        lines.append("")
    lines.append(
        "Read: open-weight models add a useful appendix. Qwen 32B and Qwen3 30B pass "
        "explicit-warning and solid controls but still launder many trace items."
    )

    lines.append("")
    lines.append("## 6. Human/Judge Reliability")
    lines.append("")
    lines.append(
        "Human-audit judge panel: {n} items, judges = {judges}.".format(
            n=human.get("n_items"),
            judges=", ".join(human.get("judges", [])),
        )
    )
    lines.append("")
    lines.append(
        md_table(
            ["judge", "established", "hedged", "weak/circular", "debunked", "absent", "errors"],
            [
                [
                    judge,
                    dist.get("established_fact", 0),
                    dist.get("hedged", 0),
                    dist.get("traced_weak_origin", 0),
                    dist.get("debunked", 0),
                    dist.get("absent", 0),
                    dist.get("errors", 0),
                ]
                for judge, dist in human.get("judge_distributions", {}).items()
            ],
        )
    )
    lines.append("")
    lines.append(
        md_table(
            ["pair", "n", "exact", "binary established"],
            [
                [
                    f"{row['a']} vs {row['b']}",
                    row["n"],
                    f"{row['exact_agreement']:.3f}",
                    f"{row['binary_established_agreement']:.3f}",
                ]
                for row in human.get("pairwise", [])
            ],
        )
    )
    maj = human.get("majority_distribution", {})
    lines.append("")
    lines.append(
        "Majority distribution: "
        + ", ".join(f"{k}={v}" for k, v in sorted(maj.items()))
        + "."
    )
    if human.get("by_source"):
        lines.append("")
        lines.append("Majority distribution by source:")
        lines.append("")
        lines.append(
            md_table(
                ["source", "n", "established", "hedged", "weak/circular", "debunked", "tie"],
                [
                    [
                        row["source"],
                        row["n"],
                        row.get("majority_dist", {}).get("established_fact", 0),
                        row.get("majority_dist", {}).get("hedged", 0),
                        row.get("majority_dist", {}).get("traced_weak_origin", 0),
                        row.get("majority_dist", {}).get("debunked", 0),
                        row.get("majority_dist", {}).get("tie", 0),
                    ]
                    for row in human.get("by_source", [])
                ],
            )
        )
    lines.append("")
    lines.append(
        "Read: exact weak/hedged/debunk distinctions vary somewhat, but binary "
        "established-vs-not-established agreement is high, especially among strong judges."
    )

    lines.append("")
    lines.append("## 7. Artifact Index")
    lines.append("")
    for label, path in SOURCE_FILES.items():
        lines.append(f"- {label}: `{path}`")
    lines.extend(
        [
            "- canonical100 scenarios: `outputs/society/citation_scenarios_100_v2.jsonl`",
            "- human annotation items: `outputs/human_eval/study_items.jsonl`",
            "- web annotation app: `web/app/index.html`",
        ]
    )
    lines.append("")
    return "\n".join(lines)


def build_overview() -> dict[str, Any]:
    return {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "terminology": {
            "trace": "Unflagged weak-origin condition; citation chain is present but not explicitly warned.",
            "flagged": "Weak-origin condition with an explicit warning.",
            "solid": "Positive-control condition with independent strong support.",
            "void": "Agentic condition where the origin is not reachable.",
            "reachable": "Agentic condition where the origin can be found only if the agent asks for it.",
        },
        "main_canonical100": read_json("outputs/society/canonical100_extension_summary.json"),
        "agent_frameworks": read_json("outputs/society/agent_framework_extension_summary.json"),
        "harness_validity": {
            "source": SOURCE_FILES["harness_validity"],
            "highlights": [
                "Low-cost reasoning helps but does not fully erase the issue: o4-mini still launders 32/100 on Canonical100; gpt-5.4-mini/nano launder 7/100; gpt-5-mini reaches 0/100.",
                "Adding a neutral scratchpad field does not fix low-cost non-reasoning models: gpt-4o-mini remains 89/100 and gpt-4.1-mini remains 63/100.",
                "Stronger intervention results from provenance-directed instructions or typed provenance fields, not generic reasoning space.",
            ],
        },
        "external_benchmark_extensions": read_external_summary(),
        "open_weight_appendix": read_json("outputs/society/open_model_summary.json"),
        "human_judge_panel": clean_judge_panel(
            read_json("outputs/human_eval/judge_panel_summary.json")
        ),
        "handoff_controls": {
            "typed": read_json("outputs/society/prov_handoff_condition_summary_20260710.json"),
            "prose": read_json("outputs/society/prov_handoff_prose_condition_summary_20260710.json"),
        },
        "scenario_construct_audit": read_json(
            "outputs/human_eval/scenario_audit_summary.json"
        ),
        "source_files": SOURCE_FILES,
        "generated_files": {
            "markdown": "outputs/society/ALL_RESULTS_OVERVIEW.md",
            "json": "outputs/society/ALL_RESULTS_OVERVIEW.json",
        },
    }


def main() -> None:
    overview = build_overview()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ALL_RESULTS_OVERVIEW.json").write_text(
        json.dumps(overview, indent=2, sort_keys=False) + "\n"
    )
    (OUT_DIR / "ALL_RESULTS_OVERVIEW.md").write_text(build_markdown(overview))
    print("wrote outputs/society/ALL_RESULTS_OVERVIEW.md")
    print("wrote outputs/society/ALL_RESULTS_OVERVIEW.json")


if __name__ == "__main__":
    main()
