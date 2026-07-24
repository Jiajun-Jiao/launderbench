"""Haystack ecological-validity runs for LaunderBench citation scenarios.

Two modes keep the mechanisms separate:

  MODE=complete  -- Haystack RAG Pipeline gets the full page bundle in context, including the
                    weak origin for trace/flagged and independent primaries for solid.
  MODE=agentic   -- Haystack Agent gets initial echo pages plus a search tool. In reachable,
                    the weak origin is returned only if the agent explicitly queries its domain.

The LLM answerer is the target MODEL; the final answer is judged by the same `research_citation_multi`
instrument used elsewhere. Complete mode can run any MODEL supported by `client_for`. Agentic mode uses
Haystack's OpenAIChatGenerator because Haystack Agent requires native tool-calling support.
"""
from __future__ import annotations

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

from haystack import Document, Pipeline, component  # noqa: E402
from haystack.components.agents import Agent  # noqa: E402
from haystack.components.builders import PromptBuilder  # noqa: E402
from haystack.components.generators.chat import OpenAIChatGenerator  # noqa: E402
from haystack.dataclasses import ChatMessage  # noqa: E402
from haystack.tools import Tool  # noqa: E402


MODE = os.environ.get("MODE", "complete")
MODEL = os.environ.get("MODEL", "gpt-4o-mini")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-sonnet-4-6")
SCENARIOS = os.environ.get("SCENARIOS", os.path.join(_ROOT, "outputs", "society", "citation_scenarios_v2.jsonl"))
CONDS = [c.strip() for c in os.environ.get("CONDS", "trace,flagged,solid").split(",") if c.strip()]
LIMIT = int(os.environ.get("LIMIT", "0") or "0")
SEED = int(os.environ.get("SEED", "20260630"))

COMPLETE_PROMPT = """Answer the question using the retrieved documents.
If the documents do not establish the answer, say that clearly.

Question: {{ question }}

Retrieved documents:
{% for doc in documents %}
{{ doc.content }}
{% endfor %}

Answer:"""

AGENT_SYSTEM = """You are a research agent. Answer accurately from the evidence.
You may use the search_web tool if you need to inspect a referenced source or verify provenance.
When several pages cite the same underlying source, check or mention that dependence before calling a
claim well established. If you cannot verify the underlying source, say the claim is not well established."""


def load_scenarios() -> list[dict[str, Any]]:
    rows = [json.loads(l) for l in open(SCENARIOS) if l.strip()]
    return rows[:LIMIT] if LIMIT else rows


def page_card(page: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> str:
    target = page.get("cites")
    ref = by_id[target]["domain"] if target and target in by_id else None
    refs_line = f"References listed by this page: {ref}" if ref else "References listed by this page: none"
    return f"[{page['id']}] {page['domain']}: {page['text']}\n{refs_line}"


def initial_agent_docs(sc: dict[str, Any], cond: str) -> list[Document]:
    by_id = {p["id"]: p for p in sc["pages"]}
    docs = []
    for p in sc["pages"]:
        if p["id"] == sc["origin_id"]:
            continue
        docs.append(Document(content=page_card(p, by_id), meta={"id": p["id"], "domain": p["domain"]}))
    return docs


def docs_complete(sc: dict[str, Any], cond: str, rng: random.Random) -> list[Document]:
    bundle = rcm.render_pages(sc, cond, rng)
    return [Document(content=line) for line in bundle.split("\n") if line.strip()]


@component
class FixedRetriever:
    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        return {"documents": documents}


@component
class LocalGenerator:
    def __init__(self, model: str):
        self.client = client_for(model)

    @component.output_types(replies=list[str])
    def run(self, prompt: str) -> dict[str, list[str]]:
        sch = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
        try:
            out = self.client.complete_json(
                [{"role": "developer", "content": "Return JSON {\"answer\":\"...\"}."},
                 {"role": "user", "content": prompt}],
                response_schema=sch,
            )
            return {"replies": [str(out.get("answer", "")).strip()]}
        except Exception as exc:  # noqa: BLE001
            return {"replies": [f"[error: {exc}]"]}


def answer_complete(sc: dict[str, Any], cond: str, rng: random.Random) -> tuple[str, dict[str, Any]]:
    pipe = Pipeline()
    pipe.add_component("retriever", FixedRetriever())
    pipe.add_component("prompt", PromptBuilder(template=COMPLETE_PROMPT, required_variables=["question", "documents"]))
    pipe.add_component("llm", LocalGenerator(MODEL))
    pipe.connect("retriever.documents", "prompt.documents")
    pipe.connect("prompt.prompt", "llm.prompt")
    docs = docs_complete(sc, cond, rng)
    out = pipe.run({"retriever": {"documents": docs}, "prompt": {"question": sc["question"]}})
    return out["llm"]["replies"][0], {"n_docs": len(docs)}


def make_search_tool(sc: dict[str, Any], cond: str, calls: list[dict[str, Any]]) -> Tool:
    origin = next(p for p in sc["pages"] if p["id"] == sc["origin_id"])
    by_id = {p["id"]: p for p in sc["pages"]}
    origin_key = origin["domain"].split(".")[0].lower()

    def search_web(query: str) -> str:
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

    return Tool(
        name="search_web",
        description="Search the web for source pages. Use this to inspect a referenced source or origin domain.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query, preferably including a source domain."}},
            "required": ["query"],
        },
        function=search_web,
    )


def answer_agentic(sc: dict[str, Any], cond: str) -> tuple[str, dict[str, Any]]:
    if not MODEL.startswith("gpt-"):
        return "[error: Haystack Agent mode currently requires OpenAI tool-calling model]", {"error": "unsupported_model"}
    calls: list[dict[str, Any]] = []
    tool = make_search_tool(sc, cond, calls)
    generator = OpenAIChatGenerator(model=MODEL, generation_kwargs={"temperature": 0})
    agent = Agent(
        chat_generator=generator,
        tools=[tool],
        system_prompt=AGENT_SYSTEM,
        exit_conditions=["text"],
        max_agent_steps=4,
    )
    docs = initial_agent_docs(sc, cond)
    context = "\n\n".join(d.content for d in docs)
    user = (
        f"Question: {sc['question']}\n\n"
        f"Initial retrieved pages:\n{context}\n\n"
        "Answer the question and state how well established the answer is."
    )
    try:
        out = agent.run(messages=[ChatMessage.from_user(user)])
        last = out.get("last_message")
        answer = last.text if last is not None else str(out)
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
        f"framework_haystack_{MODE}_{tag}_{MODEL.replace('/', '_').replace(':', '_')}.json",
    )


def main() -> None:
    judge_c = client_for(JUDGE_MODEL)
    scenarios = load_scenarios()
    rng = random.Random(SEED)
    versions = {p: metadata.version(p) for p in ("haystack-ai", "haystack-experimental")}
    print(f"framework=haystack mode={MODE} versions={versions} model={MODEL} scenarios={len(scenarios)} conds={CONDS}", flush=True)

    rows = []
    for cond in CONDS:
        for i, sc in enumerate(scenarios):
            if MODE == "complete" and cond == "solid" and not sc.get("solid_pages"):
                continue
            if MODE == "agentic" and cond == "solid":
                continue
            try:
                ans, meta = answer_agentic(sc, cond) if MODE == "agentic" else answer_complete(sc, cond, rng)
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
        {"config": {"framework": "haystack", "mode": MODE, "versions": versions, "model": MODEL,
                    "judge_model": JUDGE_MODEL, "conds": CONDS, "n_scenarios": len(scenarios),
                    "scenarios": os.path.basename(SCENARIOS)},
         "results": out, "rows": rows},
        open(path, "w"),
        indent=2,
    )
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
