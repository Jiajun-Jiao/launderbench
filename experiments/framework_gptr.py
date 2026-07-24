"""Ecological-validity run: LaunderBench scenarios through gpt-researcher (pinned 0.15.1), the
archetypal open-source research agent (multi-query planning -> retrieval -> per-source curation ->
report), against the stateless data-void mock endpoint (experiments/gptr_mock_search.py).

What this measures, per (model, condition):
  1. LAUNDERING of the final report (same 4-way judge as the headline panel);
  2. whether an *agentic* loop rescues cheap models when the origin is reachable-by-tracing
     (cond=reachable) vs crowded out (cond=void) — agency vs disposition;
  3. write-side decay instrumentation: whether the reference-card lines and the origin domain
     survive into the researcher's own curated context (the material its report is written from).

Model purity: FAST_LLM / SMART_LLM / STRATEGIC_LLM are all set to the target model, so no default
(e.g. o4-mini strategic planner) contaminates the condition. Embeddings stay at the framework
default (openai:text-embedding-3-small).

Run inside .venv-gptr with the mock server up:
  MODEL=gpt-4o-mini CONDS=void,reachable .venv-gptr/bin/python experiments/framework_gptr.py
Env: MODEL CONDS(void,reachable) LIMIT(0=all) ENDPOINT(http://127.0.0.1:8377/search)
JUDGE_MODEL(gpt-4.1-mini) SCENARIOS(citation_scenarios_v2.jsonl)
Writes outputs/society/framework_gptr_<model>.json
"""
import asyncio
import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MODEL = os.environ.get("MODEL", "gpt-4o-mini")
CONDS = [c.strip() for c in os.environ.get("CONDS", "void,reachable").split(",") if c.strip()]
LIMIT = int(os.environ.get("LIMIT", "0"))
ENDPOINT = os.environ.get("ENDPOINT", "http://127.0.0.1:8377/search")
SCENARIOS = os.environ.get("SCENARIOS", os.path.join(_ROOT, "outputs", "society", "citation_scenarios_v2.jsonl"))

# gpt-researcher reads its config from env BEFORE import-time defaults are frozen per Config().
provider = "anthropic" if MODEL.startswith("claude-") else "openai"
os.environ["RETRIEVER"] = "custom"
os.environ["RETRIEVER_ENDPOINT"] = ENDPOINT
for role in ("FAST_LLM", "SMART_LLM", "STRATEGIC_LLM"):
    os.environ[role] = f"{provider}:{MODEL}"

import research_citation_multi as rcm  # noqa: E402  (judge + wilson reuse)
from society_reflective import client_for  # noqa: E402
from gpt_researcher import GPTResearcher  # noqa: E402


def load_scenarios():
    rows = [json.loads(l) for l in open(SCENARIOS) if l.strip()]
    return rows[:LIMIT] if LIMIT else rows


def scenario_tag():
    explicit = os.environ.get("SCENARIO_TAG", "").strip()
    if explicit:
        return explicit
    tag = os.path.splitext(os.path.basename(SCENARIOS))[0].replace("citation_scenarios_", "")
    if tag.endswith("_v2"):
        tag = tag[:-3]
    return tag


def context_text(researcher):
    ctx = getattr(researcher, "context", None) or []
    if isinstance(ctx, str):
        return ctx
    parts = []
    for c in ctx:
        parts.append(c if isinstance(c, str) else json.dumps(c))
    return "\n".join(parts)


async def run_one(sc, cond, judge_c):
    os.environ["RETRIEVER_ARG_SCENARIO"] = sc["id"]
    os.environ["RETRIEVER_ARG_COND"] = cond
    researcher = GPTResearcher(query=sc["question"], report_type="research_report", verbose=False)
    await researcher.conduct_research()
    ctx = context_text(researcher)
    report = await researcher.write_report()
    verdict = rcm.judge(judge_c, sc, report)
    origin = next(p for p in sc["pages"] if p["id"] == sc["origin_id"])
    return {
        "id": sc["id"], "cond": cond, "verdict": verdict,
        "refs_survive_context": "References listed by this page" in ctx,
        # domain-string match is trivially true via echo refs lines; page-text match = actually retrieved
        "origin_page_in_context": origin["text"][:80] in ctx,
        "origin_domain_in_report": origin["domain"] in report,
        "n_context_chars": len(ctx),
        "report_excerpt": report[:400],
        "report": report, "context": ctx[:20000],
    }


async def main():
    judge_c = client_for(os.environ.get("JUDGE_MODEL", "gpt-4.1-mini"))
    scenarios = load_scenarios()
    print(f"framework=gpt-researcher(0.15.1) model={MODEL} conds={CONDS} n={len(scenarios)} "
          f"endpoint={ENDPOINT}", flush=True)
    rows = []
    for cond in CONDS:
        for i, sc in enumerate(scenarios):
            try:
                row = await run_one(sc, cond, judge_c)
            except Exception as exc:  # noqa: BLE001
                row = {"id": sc["id"], "cond": cond, "verdict": "error", "error": str(exc)[:300]}
            rows.append(row)
            print(f"  {cond} {i+1}/{len(scenarios)} {sc['id']}: {row['verdict']}", flush=True)

    print("\n=== gpt-researcher LAUNDER rate (final report, 4-way judge) ===")
    out = {}
    for cond in CONDS:
        sub = [r for r in rows if r["cond"] == cond]
        n = len([r for r in sub if r["verdict"] != "error"]) or 1
        k = sum(r["verdict"] == "established_fact" for r in sub)
        lo, hi = rcm.wilson(k, n)
        dist = defaultdict(int)
        for r in sub:
            dist[r["verdict"]] += 1
        out[cond] = {"launder": round(k / n, 2), "launder_n": f"{k}/{n}", "ci95": [lo, hi],
                     "dist": dict(dist),
                     "refs_survive_context": sum(bool(r.get("refs_survive_context")) for r in sub),
                     "origin_page_in_context": sum(bool(r.get("origin_page_in_context")) for r in sub),
                     "origin_domain_in_report": sum(bool(r.get("origin_domain_in_report")) for r in sub),
                     "errors": sum(r["verdict"] == "error" for r in sub)}
        print(f"{cond:10s} launder {k}/{n} [{lo},{hi}]  refs-in-ctx {out[cond]['refs_survive_context']}/{n} "
              f"origin-page-in-ctx {out[cond]['origin_page_in_context']}/{n} "
              f"origin-domain-in-report {out[cond]['origin_domain_in_report']}/{n} errors {out[cond]['errors']}")

    tag = scenario_tag()
    path = os.path.join(
        _ROOT,
        "outputs",
        "society",
        f"framework_gptr_{tag}_{MODEL.replace('/', '_').replace(':', '_')}.json",
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"config": {"framework": "gpt-researcher", "version": "0.15.1", "model": MODEL,
                          "llm_roles": f"{provider}:{MODEL} (fast/smart/strategic)",
                          "retriever": "custom->mock data-void", "conds": CONDS,
                          "n_scenarios": len(scenarios), "scenarios": os.path.basename(SCENARIOS)},
               "results": out, "rows": rows}, open(path, "w"), indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    asyncio.run(main())
