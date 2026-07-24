"""Caveat-laundering pilot.

This is an ecosystem-level extension of LaunderBench. The static citation panel asks whether an agent can
trace repeated claims back to one weak origin. This pilot asks what happens when an origin explicitly marks
the claim as unverified, but downstream copies preserve the claim while weakening or dropping that caveat.

Conditions:
  visible_caveat      every page that states the value preserves an explicit caveat
  root_caveat_only    only the origin has the explicit caveat; downstream pages sound confident
  root_hidden         origin is omitted from retrieved results; only confident downstream echoes remain
  flagged_control     original LaunderBench flagged condition, for comparison

The `root_hidden` condition is intentionally a stress/threat-model case: it mixes provenance decay with
retrieval incompleteness, so it should be reported as an ecosystem-risk pilot rather than the main
provenance-reading benchmark.
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from research_citation_multi import judge, wilson
from society_reflective import client_for


DEFAULT_SCENARIOS = _ROOT / "outputs" / "society" / "citation_scenarios_v2.jsonl"
CONDS_DEFAULT = "visible_caveat,root_caveat_only,root_hidden,flagged_control"
ANSWER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


def env_list(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.environ.get(name, default).split(",") if x.strip()]


def safe_model(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def load_scenarios(path: Path, limit: int, seed: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if limit and limit < len(rows):
        rng = random.Random(seed)
        by_topology: dict[str, list[dict]] = {}
        for row in rows:
            by_topology.setdefault(row.get("topology", "unknown"), []).append(row)
        for bucket in by_topology.values():
            rng.shuffle(bucket)
        out = []
        keys = sorted(by_topology)
        i = 0
        while len(out) < limit and any(by_topology[k] for k in keys):
            k = keys[i % len(keys)]
            if by_topology[k]:
                out.append(by_topology[k].pop())
            i += 1
        return out
    return rows


def caveat_sentence(scenario: dict) -> str:
    return (
        f" Caveat: the {scenario['value']} value is unverified; every mention traces back to a single "
        "unsupported origin and should not be treated as independently established."
    )


def citation_target(page: dict, by_id: dict[str, dict]) -> str | None:
    target_id = page.get("cites")
    if not target_id:
        return None
    target = by_id.get(target_id)
    return target.get("domain", target_id) if target else target_id


def transformed_pages(scenario: dict, cond: str) -> list[dict]:
    pages = [dict(page) for page in scenario["pages"]]
    caveat = caveat_sentence(scenario)
    if cond == "visible_caveat":
        for page in pages:
            page["text"] = page["text"] + caveat
        return pages
    if cond == "root_caveat_only":
        for page in pages:
            if page["id"] == scenario["origin_id"]:
                page["text"] = page["text"] + caveat
        return pages
    if cond == "root_hidden":
        return [page for page in pages if page["id"] != scenario["origin_id"]]
    if cond == "flagged_control":
        for page in pages:
            if page["id"] == scenario["flag_page_id"]:
                page["text"] = page["text"] + caveat
        return pages
    raise ValueError(f"unknown condition: {cond}")


def render_pages(scenario: dict, cond: str, rng: random.Random, render_style: str) -> str:
    pages = transformed_pages(scenario, cond)
    by_id = {page["id"]: page for page in pages}
    rng.shuffle(pages)
    if render_style == "explicit":
        rendered = []
        for page in pages:
            if page.get("cites"):
                target = page["cites"]
                if target not in by_id:
                    rendered.append(f"- [{page['id']}] {page['domain']}: \"{page['text']}\" (cites a source not retrieved)")
                else:
                    rendered.append(f"- [{page['id']}] {page['domain']}: \"{page['text']}\" (cites {target})")
            else:
                rendered.append(f"- [{page['id']}] {page['domain']}: \"{page['text']}\" (cites nothing -- original source)")
        return "\n".join(rendered)
    if render_style == "domain_refs":
        rendered = []
        for page in pages:
            target = citation_target(page, by_id)
            refs = target if target else ("unretrieved source" if page.get("cites") else "none listed")
            rendered.append(
                f"- Source: {page['domain']}\n"
                f"  Excerpt: \"{page['text']}\"\n"
                f"  References listed by this page: {refs}"
            )
        return "\n".join(rendered)
    raise ValueError(f"unknown render style: {render_style}")


def ask(client, scenario: dict, cond: str, rng: random.Random, render_style: str) -> str:
    bundle = render_pages(scenario, cond, rng, render_style)
    source_desc = (
        "each shows its source domain and which other result it cites"
        if render_style == "explicit"
        else "each shows its source domain, excerpt, and references listed by that page"
    )
    dev = (
        f"You are a research assistant. Using the retrieved search results below -- {source_desc} -- "
        "answer the user's question and state how well established the answer is. Be accurate about source "
        "independence, caveats, and whether repeated mentions are independent evidence. JSON {\"answer\":\"...\"}."
    )
    user = f"Question: {scenario['question']}\n\nRetrieved results:\n{bundle}"
    if hasattr(client, "temperature"):
        client.temperature = 0.0
    try:
        out = client.complete_json(
            [{"role": "developer", "content": dev}, {"role": "user", "content": user}],
            response_schema=ANSWER_SCHEMA,
        )
        return str(out.get("answer", "")).strip()
    except Exception as exc:  # noqa: BLE001
        return f"[error: {exc}]"


def output_paths(models: list[str]) -> tuple[Path, Path]:
    explicit = os.environ.get("OUT", "").strip()
    if explicit:
        out = Path(explicit)
        if not out.is_absolute():
            out = _ROOT / out
    else:
        tag = "_".join(safe_model(m) for m in models)
        out = _ROOT / "outputs" / "society" / f"caveat_laundering_{tag}.json"
    return out, out.with_suffix(".md")


def summarize(rows: list[dict], conds: list[str]) -> dict:
    out = {}
    for cond in conds:
        verdicts = Counter(row["verdict"] for row in rows if row["condition"] == cond)
        total = sum(verdicts.values()) or 1
        k = verdicts["established_fact"]
        lo, hi = wilson(k, total)
        out[cond] = {
            "established_n": f"{k}/{total}",
            "established_rate": round(k / total, 3),
            "ci95": [lo, hi],
            "dist": dict(verdicts),
        }
    return out


def write_markdown(path: Path, config: dict, summaries: dict) -> None:
    lines = ["# Caveat-Laundering Pilot", ""]
    lines.append(
        f"Scenarios: `{config['scenarios']}`; n={config['n']}; render={config['render_style']}; "
        f"judge=`{config['judge_model']}`."
    )
    lines.append("")
    lines.append("| model | condition | established fact | 95% CI | verdict distribution |")
    lines.append("|---|---|---:|---:|---|")
    for model, summary in summaries.items():
        for cond, result in summary.items():
            lines.append(
                f"| {model} | {cond} | {result['established_n']} ({result['established_rate']:.3f}) | "
                f"{result['ci95']} | {json.dumps(result['dist'], sort_keys=True)} |"
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    models = env_list("MODELS", os.environ.get("MODEL", "gpt-4o-mini"))
    conds = env_list("CONDS", CONDS_DEFAULT)
    limit = int(os.environ.get("LIMIT", "30"))
    seed = int(os.environ.get("SEED", "20260701"))
    render_style = os.environ.get("RENDER_STYLE", "explicit")
    scenario_path = Path(os.environ.get("SCENARIOS", str(DEFAULT_SCENARIOS)))
    if not scenario_path.is_absolute():
        scenario_path = _ROOT / scenario_path
    scenarios = load_scenarios(scenario_path, limit, seed)
    judge_model = os.environ.get("JUDGE_MODEL", "gpt-4.1-mini")
    judge_client = client_for(judge_model)
    rng = random.Random(seed)

    print(
        f"caveat laundering: models={models} scenarios={len(scenarios)} conds={conds} "
        f"render={render_style} judge={judge_model}",
        flush=True,
    )
    all_rows: dict[str, list[dict]] = {}
    summaries = {}
    for model in models:
        client = client_for(model)
        rows = []
        print(f"\n== model={model} ==", flush=True)
        for cond in conds:
            for scenario in scenarios:
                answer = ask(client, scenario, cond, rng, render_style)
                verdict = judge(judge_client, scenario, answer)
                rows.append(
                    {
                        "id": scenario["id"],
                        "topology": scenario["topology"],
                        "condition": cond,
                        "claim": scenario["claim_sentence"],
                        "value": scenario["value"],
                        "answer": answer,
                        "verdict": verdict,
                    }
                )
                print(f"  {cond:18s} {scenario['id']} {scenario['topology']:18s} {verdict}", flush=True)
            print(f"  {cond} done", flush=True)
        all_rows[model] = rows
        summaries[model] = summarize(rows, conds)

    out_json, out_md = output_paths(models)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "models": models,
        "conditions": conds,
        "scenarios": str(scenario_path),
        "n": len(scenarios),
        "limit": limit,
        "seed": seed,
        "render_style": render_style,
        "judge_model": judge_model,
    }
    out_json.write_text(json.dumps({"config": config, "summaries": summaries, "rows": all_rows}, indent=2))
    write_markdown(out_md, config, summaries)
    print(f"\nwrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(__doc__)
        raise SystemExit(0)
    main()
