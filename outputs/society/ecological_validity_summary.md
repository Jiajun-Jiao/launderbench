# Ecological-validity wing: off-the-shelf agent stacks (vs "self-built harness" objection)

Canonical 30 scenarios (`citation_scenarios_v2.jsonl`), same judge (`gpt-4.1-mini`) and Wilson CIs as
the headline §7.2 panel. Everything item-level is byte-identical to the controlled harness; only the
surrounding stack is swapped for an off-the-shelf framework at default settings.

## A. LangChain default tutorial RAG chain (pinned: langchain 1.3.11 / -core 1.4.8 / -openai 1.3.3 / -anthropic 1.4.8)

Chain = one Document per page → tutorial `retriever | format_docs` piping → the canonical LangChain RAG
prompt (`rlm/rag-prompt`, verbatim fallback recorded in the run JSON) → free-text answer (no JSON
constraint). Note the default prompt does NOT ask for evidence-strength accuracy (ours does) and asks
for a concise ≤3-sentence answer. Script: `experiments/framework_langchain.py` (`.venv-frameworks/`).

| model | trace launder | flagged | solid accept | note |
|---|---|---|---|---|
| gpt-4o-mini | **29/30 (.97)** [.83,.99] | **4/30 (.13)** | 30/30 (1.00) | worse than our harness (.90 / .00): default prompt is *less* provenance-charitable |
| claude-opus-4-8 | **0/30 (.00)** [0,.11] | 0/30 | 30/30 (1.00) | 30/30 trace verdicts are `traced_weak_origin`; perfect discrimination (+1.00) under the same default prompt |

Reading: the mini/Opus split is not an artifact of our hand-built harness — it reproduces, slightly
amplified, under the most popular framework's default chain. Under the default prompt gpt-4o-mini even
ignores the explicit unverified flag 4/30 times (flagged .13 vs .00 in our harness), i.e. the headline
panel *understates* deployment risk. Opus spends its three allowed sentences flagging the weak origin.
Fix applied during runs: current Anthropic API rejects `temperature` for claude-opus-4-8 (400); the
first Opus pass silently judged as `absent` 90/90 from caught exceptions — rerun cleanly after removing
the param (lesson: audit `[error:` answers before trusting `absent` rates).

## B. gpt-researcher 0.15.1 data-void runs (`experiments/framework_gptr.py` + `gptr_mock_search.py`)

The archetypal open research agent (initial search → evidence-conditioned sub-query planning →
retrieval → per-source curation → report). Custom-retriever mock (stateless, port 8377) serves the
echo pages for any query; `void` = origin page crowded out entirely; `reachable` = origin returned iff
a sub-query names the origin site. FAST/SMART/STRATEGIC_LLM all pinned to the target model; embeddings
at framework default. n=30 each cell, same 4-way judge on the final report.

| model | cond | launder | refs-line in own context | origin page retrieved | origin domain named in report |
|---|---|---|---|---|---|
| gpt-4o-mini | void | **30/30 (1.00)** | 30/30 | 0/30 | 30/30 |
| gpt-4o-mini | reachable | **30/30 (1.00)** | 30/30 | **0/30** | 30/30 |
| claude-haiku-4-5 | void | **4/30 (.13)** | 30/30 | 0/30 | 30/30 |
| claude-haiku-4-5 | reachable | **0/30 (.00)** | 30/30 | **0/30** | 30/30 |

Readings.
1. **The full agentic stack makes the cheap model worse, not better**: gpt-4o-mini goes .90
   (single-pass) → .97 (LangChain default) → **1.00** (research agent), laundering every scenario in
   both conditions.
2. **Agency does not substitute for disposition — and the choice not to trace is the model's.** The
   sub-query planner is evidence-conditioned (gpt-researcher feeds the initial search results,
   including every "References listed by this page: <origin>" line, into the planning prompt, which
   says to use specific details from the context). Yet across 120 runs neither model EVER issued a
   query naming the origin site (origin retrieval 0/60 and 0/60) — so `reachable` collapses to `void`
   ex post, which is precisely the point: the affordance to trace existed and was never exercised.
   gpt-4o-mini names the origin domain in 30/30 of its reports while never having checked it and
   still asserts the claim as established.
3. **Haiku's discipline is textual provenance-reading, not retrieval**: it never fetches the origin
   either, but reads the circular citation structure out of its own curated context and refuses to
   launder (51/60 traced_weak_origin or hedged in final reports; void-vs-reachable difference 4/30 vs
   0/30 is noise around a low rate — served content is identical given zero tracing).
4. **Pipeline amplification, write-side**: Haiku .03 (single-pass) → .13 (research-agent void) is
   consistent with the §7.3 prose-relay decay mechanism (its report is written from its own compressed
   curation), though CIs overlap ([.01,.17] vs [.05,.3]) — report as directional, not a firm claim.

Fix applied during runs: `origin_in_context` originally matched the origin *domain string*, which the
echo refs lines make trivially true; replaced with origin *page-text* matching (actual retrieval)
before the full runs.

## C. Haystack 2.30.2 and LlamaIndex 0.14.23 extension

Two added framework probes split the discrepancy explicitly:

* `complete` = full weak-origin page already in context, as in the canonical harness.
* `agentic` = initial context excludes the origin; in `reachable`, the origin is returned only if the
  agent explicitly queries its domain.

| framework / mode | model | trace | flagged | solid | void | reachable | origin retrieved in reachable |
|---|---|---:|---:|---:|---:|---:|---:|
| Haystack complete | gpt-4o-mini | **30/30** | 6/30 | 30/30 | - | - | - |
| Haystack complete | claude-haiku-4-5 | 8/30 | 0/30 | 30/30 | - | - | - |
| Haystack agentic | gpt-4o-mini | - | - | - | **26/30** | **24/30** | 3/30 |
| LlamaIndex complete | gpt-4o-mini | **30/30** | 2/30 | 30/30 | - | - | - |
| LlamaIndex complete | claude-haiku-4-5 | 3/30 | 0/30 | 30/30 | - | - | - |
| LlamaIndex agentic | gpt-4o-mini | - | - | - | **30/30** | **30/30** | 0/30 |

Reading: default-ish Haystack and LlamaIndex RAG wrappers do not repair the cheap-model failure when
the weak origin is already present (gpt-4o-mini launders 30/30 trace in both). A more disciplined model
still performs much better in the same wrappers (Haiku 8/30 and 3/30), so the split is not merely a
framework artifact. In agentic/tool mode, the models often call search, but rarely query the specific
origin domain; `reachable` therefore remains close to `void`.

Detailed output: `outputs/society/agent_framework_extension_summary.md`.
