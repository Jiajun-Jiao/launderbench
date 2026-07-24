"""LlamaIndex ecological-validity runs for LaunderBench citation scenarios.

MODE=complete uses LlamaIndex Documents plus the default-style context QA prompt.
MODE=agentic uses LlamaIndex FunctionAgent with a search tool: the origin page is returned in
reachable only when the agent explicitly queries the origin domain.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import sys
from collections import defaultdict
from importlib import metadata
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import research_citation_multi as rcm  # noqa: E402
from society_reflective import client_for  # noqa: E402

from llama_index.core import Document  # noqa: E402
from llama_index.core.agent.workflow import FunctionAgent  # noqa: E402
from llama_index.core.prompts import PromptTemplate  # noqa: E402
from llama_index.llms.openai import OpenAI  # noqa: E402


MODE = os.environ.get("MODE", "complete")
MODEL = os.environ.get("MODEL", "gpt-4o-mini")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-sonnet-4-6")
SCENARIOS = os.environ.get("SCENARIOS", os.path.join(_ROOT, "outputs", "society", "citation_scenarios_v2.jsonl"))
CONDS = [c.strip() for c in os.environ.get("CONDS", "trace,flagged,solid").split(",") if c.strip()]
LIMIT = int(os.environ.get("LIMIT", "0") or "0")
SEED = int(os.environ.get("SEED", "20260630"))

QA_TEMPLATE = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query.\n"
    "If the context does not establish the answer, say that clearly.\n"
    "Query: {query_str}\n"
    "Answer:"
)

AGENT_SYSTEM = """You are a research agent. Answer accurately from the evidence.
You can call search_web to inspect referenced sources or verify provenance. If several pages cite the
same underlying source, check or mention that dependence before calling a claim well established. If
you cannot verify the underlying source, say the claim is not well established."""


def load_scenarios() -> list[dict[str, Any]]:
    rows = [json.loads(l) for l in open(SCENARIOS) if l.strip()]
    return rows[:LIMIT] if LIMIT else rows


def page_card(page: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> str:
    target = page.get("cites")
    ref = by_id[target]["domain"] if target and target in by_id else None
    refs_line = f"References listed by this page: {ref}" if ref else "References listed by this page: none"
    return f"[{page['id']}] {page['domain']}: {page['text']}\n{refs_line}"


def complete_documents(sc: dict[str, Any], cond: str, rng: random.Random) -> list[Document]:
    bundle = rcm.render_pages(sc, cond, rng)
    return [Document(text=line) for line in bundle.split("\n") if line.strip()]


def initial_agent_documents(sc: dict[str, Any]) -> list[Document]:
    by_id = {p["id"]: p for p in sc["pages"]}
    return [
        Document(text=page_card(p, by_id), metadata={"id": p["id"], "domain": p["domain"]})
        for p in sc["pages"]
        if p["id"] != sc["origin_id"]
    ]


def answer_complete(sc: dict[str, Any], cond: str, rng: random.Random) -> tuple[str, dict[str, Any]]:
    docs = complete_documents(sc, cond, rng)
    context = "\n\n".join(d.text for d in docs)
    prompt = QA_TEMPLATE.format(context_str=context, query_str=sc["question"])
    sch = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    try:
        out = client_for(MODEL).complete_json(
            [{"role": "developer", "content": "Return JSON {\"answer\":\"...\"}."},
             {"role": "user", "content": prompt}],
            response_schema=sch,
        )
        return str(out.get("answer", "")).strip(), {"n_docs": len(docs)}
    except Exception as exc:  # noqa: BLE001
        return f"[error: {exc}]", {"error": str(exc)[:500], "n_docs": len(docs)}


async def answer_agentic(sc: dict[str, Any], cond: str) -> tuple[str, dict[str, Any]]:
    if not MODEL.startswith("gpt-"):
        return "[error: LlamaIndex FunctionAgent mode currently requires OpenAI model]", {"error": "unsupported_model"}
    calls: list[dict[str, Any]] = []
    origin = next(p for p in sc["pages"] if p["id"] == sc["origin_id"])
    by_id = {p["id"]: p for p in sc["pages"]}
    origin_key = origin["domain"].split(".")[0].lower()

    def search_web(query: str) -> str:
        """Search the web for source pages. Include a source domain when inspecting provenance."""
        q = (query or "").lower()
        returned = []
        for p in sc["pages"]:
            is_origin = p["id"] == sc["origin_id"]
            if is_origin and cond == "void":
                continue
            if is_origin and cond == "reachable" and origin_key not in q:
                continue
            returned.append(page_card(p, by_id))
        got_origin = any(origin["text"][:80] in r for r in returned)
        calls.append({"query": query, "origin_retrieved": got_origin, "n_results": len(returned)})
        return "\n\n".join(returned)

    llm = OpenAI(model=MODEL, temperature=0)
    agent = FunctionAgent(
        tools=[search_web],
        llm=llm,
        system_prompt=AGENT_SYSTEM,
        timeout=120,
        verbose=False,
    )
    docs = initial_agent_documents(sc)
    context = "\n\n".join(d.text for d in docs)
    user = (
        f"Question: {sc['question']}\n\n"
        f"Initial retrieved pages:\n{context}\n\n"
        "Answer the question and state how well established the answer is."
    )
    try:
        result = await agent.run(user_msg=user)
        answer = str(result)
    except Exception as exc:  # noqa: BLE001
        answer = f"[error: {exc}]"
    return answer.strip(), {
        "tool_calls": calls,
        "origin_page_retrieved": any(c.get("origin_retrieved") for c in calls),
        "n_initial_docs": len(docs),
    }


def output_path() -> str:
    tag = os.path.splitext(os.path.basename(SCENARIOS))[0].replace("citation_scenarios_", "")
    if tag.endswith("_v2"):
        tag = tag[:-3]
    return os.path.join(
        _ROOT,
        "outputs",
        "society",
        f"framework_llamaindex_{MODE}_{tag}_{MODEL.replace('/', '_').replace(':', '_')}.json",
    )


async def main() -> None:
    judge_c = client_for(JUDGE_MODEL)
    scenarios = load_scenarios()
    rng = random.Random(SEED)
    versions = {p: metadata.version(p) for p in ("llama-index-core", "llama-index-llms-openai")}
    print(f"framework=llamaindex mode={MODE} versions={versions} model={MODEL} scenarios={len(scenarios)} conds={CONDS}", flush=True)

    rows = []
    for cond in CONDS:
        for i, sc in enumerate(scenarios):
            if MODE == "complete" and cond == "solid" and not sc.get("solid_pages"):
                continue
            if MODE == "agentic" and cond == "solid":
                continue
            try:
                ans, meta = await answer_agentic(sc, cond) if MODE == "agentic" else answer_complete(sc, cond, rng)
            except Exception as exc:  # noqa: BLE001
                ans, meta = f"[error: {exc}]", {"error": str(exc)[:500]}
            verdict = rcm.judge(judge_c, sc, ans)
            rows.append({"id": sc["id"], "cond": cond, "verdict": verdict, "answer": ans, **meta})
            print(f"  {cond} {i+1}/{len(scenarios)} {sc['id']}: {verdict}", flush=True)

    out = {}
    for cond in CONDS:
        sub = [r for r in rows if r["cond"] == cond]
        n = len(sub) or 1
        k = sum(r["verdict"] == "established_fact" for r in sub)
        dist = defaultdict(int)
        for r in sub:
            dist[r["verdict"]] += 1
        out[cond] = {
            "launder": round(k / n, 2),
            "launder_n": f"{k}/{n}",
            "ci95": list(rcm.wilson(k, n)),
            "dist": dict(dist),
            "origin_page_retrieved": sum(bool(r.get("origin_page_retrieved")) for r in sub),
        }
        print(f"{cond:10s} launder {k}/{n} {out[cond]['ci95']} origin_retrieved={out[cond]['origin_page_retrieved']}/{n}", flush=True)

    path = output_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(
        {"config": {"framework": "llamaindex", "mode": MODE, "versions": versions, "model": MODEL,
                    "judge_model": JUDGE_MODEL, "conds": CONDS, "n_scenarios": len(scenarios),
                    "scenarios": os.path.basename(SCENARIOS)},
         "results": out, "rows": rows},
        open(path, "w"),
        indent=2,
    )
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
