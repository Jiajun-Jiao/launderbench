"""Ecological-validity replication: run the canonical citation-laundering scenarios through an
off-the-shelf LangChain RAG chain with the framework's DEFAULT tutorial prompt (rlm/rag-prompt),
instead of our hand-built harness prompt.

Design: everything that defines the benchmark item is byte-identical to research_citation_multi.py
(page text, citation rendering, shuffle rng, judge, Wilson CI). The only substitutions are the
ecological ones a real default deployment would make:
  - pages arrive as retrieved Documents (one page per Document), injected at the point where the
    tutorial chain pipes `retriever | format_docs` (fixed doc set per item, so a query-ignoring
    retriever and direct injection are equivalent);
  - the answer prompt is LangChain's canonical RAG tutorial prompt (hub "rlm/rag-prompt"), which
    notably does NOT ask for evidence-strength accuracy (our harness prompt does);
  - output is free text (no JSON schema constraint).
So a reviewer objection "your self-built single-pass harness is a strawman" is answered by showing
the same split under the most popular framework's default chain — which is *less* provenance-charitable
than our harness.

Run inside .venv-frameworks (pinned langchain deps + system site packages):
  .venv-frameworks/bin/python experiments/framework_langchain.py
Env: MODEL(gpt-4o-mini) CONDS(trace,flagged,solid) JUDGE_MODEL(gpt-4.1-mini)
SCENARIOS(outputs/society/citation_scenarios_v2.jsonl) SEED(20260630)
Writes outputs/society/framework_langchain_<model>.json (+ .md table row source).
"""
import json
import os
import random
import sys
from collections import defaultdict
from importlib import metadata

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import research_citation_multi as rcm  # noqa: E402  (render_pages / judge / wilson reuse)
from society_reflective import client_for  # noqa: E402

from langchain_core.documents import Document  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_core.runnables import RunnablePassthrough  # noqa: E402

CONDS = [c.strip() for c in os.environ.get("CONDS", "trace,flagged,solid").split(",") if c.strip()]
SEED = int(os.environ.get("SEED", "20260630"))
SCENARIOS = os.environ.get("SCENARIOS", os.path.join(_ROOT, "outputs", "society", "citation_scenarios_v2.jsonl"))

# LangChain's canonical RAG tutorial prompt ("rlm/rag-prompt" on the LangChain hub). We try to pull the
# live hub version first; on any failure we fall back to this verbatim copy so runs are reproducible
# offline. The run output records which path was taken and the exact text used.
RAG_PROMPT_FALLBACK = (
    "You are an assistant for question-answering tasks. Use the following pieces of retrieved context "
    "to answer the question. If you don't know the answer, just say that you don't know. Use three "
    "sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context} \n"
    "Answer:"
)


LIMIT = int(os.environ.get("LIMIT", "0"))  # 0 = all


def load_scenarios():
    rows = [json.loads(l) for l in open(SCENARIOS) if l.strip()]
    return rows[:LIMIT] if LIMIT else rows


def page_documents(scenario, cond, rng):
    """One retrieved Document per page, text byte-identical to the canonical bundle lines."""
    bundle = rcm.render_pages(scenario, cond, rng)
    return [Document(page_content=line) for line in bundle.split("\n") if line.strip()]


def get_prompt():
    try:
        from langchain import hub  # noqa: PLC0415
        prompt = hub.pull("rlm/rag-prompt")
        return prompt, "hub:rlm/rag-prompt"
    except Exception:  # noqa: BLE001  (no hub access — use the verbatim copy)
        return ChatPromptTemplate.from_template(RAG_PROMPT_FALLBACK), "fallback:rlm/rag-prompt-verbatim"


def make_llm(model):
    if model.startswith("claude-"):
        from langchain_anthropic import ChatAnthropic  # noqa: PLC0415
        # no temperature: the current Anthropic API 400s on `temperature` for claude-opus-4-8
        return ChatAnthropic(model=model, max_tokens=1024)
    from langchain_openai import ChatOpenAI  # noqa: PLC0415
    return ChatOpenAI(model=model, temperature=0.0)


def scenario_tag():
    explicit = os.environ.get("SCENARIO_TAG", "").strip()
    if explicit:
        return explicit
    tag = os.path.splitext(os.path.basename(SCENARIOS))[0].replace("citation_scenarios_", "")
    if tag.endswith("_v2"):
        tag = tag[:-3]
    return tag


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def main():
    model = os.environ.get("MODEL", "gpt-4o-mini")
    llm = make_llm(model)
    prompt, prompt_source = get_prompt()
    judge_c = client_for(os.environ.get("JUDGE_MODEL", "gpt-4.1-mini"))
    scenarios = load_scenarios()
    rng = random.Random(SEED)
    versions = {p: metadata.version(p) for p in
                ("langchain", "langchain-core", "langchain-openai", "langchain-anthropic")}
    print(f"framework=langchain {versions} model={model} prompt={prompt_source} "
          f"scenarios={len(scenarios)} conds={CONDS}", flush=True)

    agg = {c: defaultdict(int) for c in CONDS}
    examples = []
    for cond in CONDS:
        for sc in scenarios:
            if cond == "solid" and not sc.get("solid_pages"):
                continue
            docs = page_documents(sc, cond, rng)
            chain = ({"context": (lambda _q, _d=docs: format_docs(_d)), "question": RunnablePassthrough()}
                     | prompt | llm | StrOutputParser())
            try:
                ans = chain.invoke(sc["question"]).strip()
            except Exception as exc:  # noqa: BLE001
                ans = f"[error: {exc}]"
            p = rcm.judge(judge_c, sc, ans)
            agg[cond][p] += 1
            if p == "established_fact" and cond == "trace" and len(examples) < 6:
                examples.append({"id": sc["id"], "claim": sc["claim_sentence"], "answer": ans[:300]})
        print(f"  {cond} done", flush=True)

    print("\n=== LangChain default-chain LAUNDER rate ===")
    out = {}
    for cond in CONDS:
        n = sum(agg[cond].values()) or 1
        k = agg[cond]["established_fact"]
        lo, hi = rcm.wilson(k, n)
        out[cond] = {"launder": round(k / n, 2), "launder_n": f"{k}/{n}", "ci95": [lo, hi],
                     "dist": {kk: round(vv / n, 2) for kk, vv in agg[cond].items()}}
        print(f"{cond:9s}{k/n:>9.2f}  [{lo},{hi}]  n={n}")

    tag = scenario_tag()
    path = os.path.join(
        _ROOT,
        "outputs",
        "society",
        f"framework_langchain_{tag}_{model.replace('/', '_').replace(':', '_')}.json",
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prompt_text = RAG_PROMPT_FALLBACK if prompt_source.startswith("fallback") else str(prompt)
    json.dump({"config": {"framework": "langchain", "versions": versions, "model": model,
                          "prompt_source": prompt_source, "prompt_text": prompt_text,
                          "conds": CONDS, "n_scenarios": len(scenarios), "scenarios": os.path.basename(SCENARIOS)},
               "results": out, "launder_examples": examples}, open(path, "w"), indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
