# All Results Overview

This is a navigation-level summary of the current LaunderBench / source-dependence-laundering experiment state. It links to the detailed run artifacts rather than replacing them.

## Terminology

- `trace`: unflagged weak-origin condition; the citation chain is present, but not explicitly warned.
- `flagged`: same weak-origin condition with an explicit warning.
- `solid`: positive-control condition with independent strong support.
- `void`: agentic condition where the origin is not reachable.
- `reachable`: agentic condition where the origin can be found only if the agent asks for it.

## 1. Main Canonical100 Result

Panel: 100 validated weak-origin scenarios; 96 validated solid controls. Solid missing: S041, S085, S088, S094.

| model | trace laundering | flagged laundering | solid established | discrimination |
|---|---|---|---|---|
| claude-haiku-4-5 | 0.02 (2/100) | 0.00 (0/100) | 1.00 (96/96) | 0.98 |
| gpt-5.5 | 0.00 (0/100) | 0.00 (0/100) | 0.96 (92/96) | 0.96 |
| o3 | 0.05 (5/100) | 0.00 (0/100) | 1.00 (96/96) | 0.95 |
| claude-opus-4-8 | 0.00 (0/100) | 0.00 (0/100) | 0.92 (88/96) | 0.92 |
| claude-sonnet-4-6 | 0.00 (0/100) | 0.00 (0/100) | 0.79 (76/96) | 0.79 |
| gemini-2.5-flash | 0.29 (29/100) | 0.00 (0/100) | 1.00 (96/96) | 0.71 |
| qwen3:30b-a3b-instruct-2507-q4_K_M | 0.53 (53/100) | 0.01 (1/100) | 0.97 (93/96) | 0.43 |
| gpt-4.1 | 0.62 (62/100) | 0.00 (0/100) | 1.00 (96/96) | 0.38 |
| qwen2.5:32b-instruct | 0.70 (70/100) | 0.01 (1/100) | 1.00 (96/96) | 0.28 |
| gpt-4o | 0.72 (72/100) | 0.01 (1/100) | 1.00 (96/96) | 0.27 |
| gpt-4.1-mini | 0.74 (74/100) | 0.01 (1/100) | 1.00 (96/96) | 0.24 |
| gpt-4o-mini | 0.96 (96/100) | 0.04 (4/100) | 1.00 (96/96) | 0.03 |

Read: the split is specifically in unflagged provenance tracing. Explicit flags almost eliminate laundering, and solid controls are accepted, so the task is discriminative rather than just inducing blanket skepticism.

Trace laundering by topology:

| model | linear | star | fake_corroboration |
|---|---|---|---|
| claude-haiku-4-5 | 1/34 | 0/33 | 1/33 |
| gpt-5.5 | 0/34 | 0/33 | 0/33 |
| o3 | 4/34 | 0/33 | 1/33 |
| claude-opus-4-8 | 0/34 | 0/33 | 0/33 |
| claude-sonnet-4-6 | 0/34 | 0/33 | 0/33 |
| gemini-2.5-flash | 16/34 | 4/33 | 9/33 |
| qwen3:30b-a3b-instruct-2507-q4_K_M | 22/34 | 12/33 | 19/33 |
| gpt-4.1 | 33/34 | 7/33 | 22/33 |
| qwen2.5:32b-instruct | 28/34 | 19/33 | 23/33 |
| gpt-4o | 32/34 | 11/33 | 29/33 |
| gpt-4.1-mini | 32/34 | 14/33 | 28/33 |
| gpt-4o-mini | 34/34 | 29/33 | 33/33 |

## 2. Framework And Agent Results

| framework / mode | trace | flagged | solid | void | reachable | search tool called | origin retrieved |
|---|---|---|---|---|---|---|---|
| haystack complete n=100 gpt-4o-mini | 99/100 | 20/100 | 96/96 | - | - | - | - |
| langchain complete n=100 gpt-4o-mini | 98/100 | 14/100 | 96/96 | - | - | - | - |
| llamaindex complete n=100 gpt-4o-mini | 97/100 | 5/100 | 96/96 | - | - | - | - |
| gpt-researcher agentic n=30 claude-haiku-4-5 | - | - | - | 4/30 | 0/30 | void 0/30; reach 0/30 | void 0/30; reach 0/30 |
| gpt-researcher agentic n=30 gpt-4o-mini | - | - | - | 30/30 | 30/30 | void 0/30; reach 0/30 | void 0/30; reach 0/30 |
| haystack agentic n=30 gpt-4o-mini | - | - | - | 26/30 | 24/30 | void 29/30; reach 29/30 | void 0/30; reach 3/30 |
| haystack complete n=30 claude-haiku-4-5 | 8/30 | 0/30 | 30/30 | - | - | - | - |
| haystack complete n=30 gpt-4o-mini | 30/30 | 6/30 | 30/30 | - | - | - | - |
| langchain complete n=30 claude-opus-4-8 | 0/30 | 0/30 | 30/30 | - | - | - | - |
| langchain complete n=30 gpt-4o-mini | 29/30 | 4/30 | 30/30 | - | - | - | - |
| llamaindex agentic n=30 gpt-4o-mini | - | - | - | 30/30 | 30/30 | void 29/30; reach 29/30 | void 0/30; reach 0/30 |
| llamaindex complete n=30 claude-haiku-4-5 | 3/30 | 0/30 | 30/30 | - | - | - | - |
| llamaindex complete n=30 gpt-4o-mini | 30/30 | 2/30 | 30/30 | - | - | - | - |

Read: framework wrappers do not automatically repair the failure. In complete-context RAG, gpt-4o-mini still launders nearly/all trace items. In agentic reachable-origin settings, tools are available but agents rarely retrieve the origin page.

## 3. Harness Validity And Mechanism Checks

- Low-cost reasoning helps but does not fully erase the issue: o4-mini still launders 32/100 on Canonical100; gpt-5.4-mini/nano launder 7/100; gpt-5-mini reaches 0/100.
- Adding a neutral scratchpad field does not fix low-cost non-reasoning models: gpt-4o-mini remains 89/100 and gpt-4.1-mini remains 63/100.
- Stronger intervention results from provenance-directed instructions or typed provenance fields, not generic reasoning space.

Bidirectional handoff controls:

| carrier | typed trace | typed solid | typed disc. | prose trace | prose solid |
|---|---|---|---|---|---|
| gpt-4o-mini | 0/96 | 96/96 | 1.000 | 0/96 | 85/96 |
| gpt-4.1-mini | 0/96 | 96/96 | 1.000 | 0/96 | 86/96 |
| gpt-4o | 0/96 | 96/96 | 1.000 | 0/96 | 86/96 |
| gpt-4.1 | 0/96 | 96/96 | 1.000 | 0/96 | 87/96 |

Read: with a correct upstream type, the structured record preserves all matched negative and positive licenses (0/384 weak, 384/384 solid). Explicit prose remains weak-safe but establishes 344/384 solids.

Blinded construct audit: 40/40 majority passes; removing 9 items with any one-rater dissent changes no headline trace rate by more than .029.

## 4. External Benchmark Extensions

Seeded LaunderBench panels:

| benchmark | model | trace | flagged | solid | discrimination |
|---|---|---|---|---|---|
| expertqa | gpt-4o-mini | 17/30 | 0/30 | 30/30 | 0.43 |
| expertqa | claude-opus-4-8 | 0/30 | 0/30 | 21/30 | 0.70 |
| alce | gpt-4o-mini | 21/30 | 0/30 | 30/30 | 0.30 |
| alce | claude-opus-4-8 | 0/30 | 0/30 | 30/30 | 1.00 |
| ragtruth | gpt-4o-mini | 13/30 | 0/30 | 30/30 | 0.57 |
| ragtruth | claude-opus-4-8 | 0/30 | 0/30 | 29/30 | 0.97 |

Type-once-carry handoff:

| benchmark | Opus typer weak/circular | mini self-downstream | mini self-carry | Opus typed-handoff | mini own field correct |
|---|---|---|---|---|---|
| expertqa | 29/30 | 26/30 | 15/30 | 0/30 | 14/30 |
| alce | 30/30 | 27/30 | 18/30 | 0/30 | 9/30 |
| ragtruth | 29/30 | 22/30 | 11/30 | 0/30 | 17/30 |

Original native-control panels:

| benchmark | model | metric | value | extra |
|---|---|---|---|---|
| expertqa | gpt-4o-mini | accuracy | 0.60 | recall=0.87, specificity=0.33 |
| expertqa | claude-opus-4-8 | accuracy | 0.53 | recall=1.00, specificity=0.07 |
| alce | gpt-4o-mini | match_or_partial | 0.77 | exact=0.20 |
| alce | claude-opus-4-8 | match_or_partial | 0.73 | exact=0.30 |
| ragtruth | gpt-4o-mini | accuracy | 0.73 | recall=0.67, specificity=0.80 |
| ragtruth | claude-opus-4-8 | accuracy | 0.80 | recall=1.00, specificity=0.60 |

Read: the seeded ExpertQA/ALCE/RAGTruth panels reproduce the same direction: gpt-4o-mini launders many trace items; Claude Opus is near zero on trace while still accepting most solid controls.

## 5. Open-Weight Appendix

Canonical100 open-weight rows:

| model | n | trace | flagged | solid | discrimination | read |
|---|---|---|---|---|---|---|
| qwen2.5:32b-instruct | 100 | 70/100 | 1/100 | 96/96 | 0.30 | valid discriminator; trace laundering remains |
| qwen3:30b-a3b-instruct-2507-q4_K_M | 100 | 53/100 | 1/100 | 93/96 | 0.44 | valid discriminator; trace laundering remains |

Smaller Canonical100 controls:

| model | n | trace | flagged | solid | discrimination | read |
|---|---|---|---|---|---|---|
| qwen2.5:7b-instruct | 100 | 88/100 | 22/100 | 94/96 | 0.10 | blanket-trust / weak explicit-warning control |
| llama3.1:8b | 100 | 59/100 | 14/100 | 76/96 | 0.20 | adherence/solid-control confounded |
| mistral:7b | 100 | 72/100 | 20/100 | 89/96 | 0.21 | blanket-trust / weak explicit-warning control |
| gemma2:9b | 100 | 62/100 | 0/100 | 95/96 | 0.37 | valid discriminator; trace laundering remains |

Canonical30 legacy appendix:

| model | trace | flagged | solid | discrimination | read |
|---|---|---|---|---|---|
| qwen2.5:7b-instruct | 28/30 | 6/30 | 28/30 | 0.00 | blanket-trust / weak explicit-warning control |
| qwen2.5:32b-instruct | 15/30 | 0/30 | 30/30 | 0.50 | valid discriminator; trace laundering remains |
| llama3.1:8b | 13/30 | 4/30 | 21/30 | 0.27 | adherence/solid-control confounded |
| qwen3:30b-a3b-instruct-2507-q4_K_M | 17/30 | 0/30 | 30/30 | 0.43 | valid discriminator; trace laundering remains |

Read: open-weight models add a useful appendix. Qwen 32B and Qwen3 30B pass explicit-warning and solid controls but still launder many trace items.

## 6. Human/Judge Reliability

Human-audit judge panel: 112 items, judges = gpt-4.1-mini, gpt-5.5, claude-opus-4-8, gemini-2.5-flash.

| judge | established | hedged | weak/circular | debunked | absent | errors |
|---|---|---|---|---|---|---|
| gpt-4.1-mini | 29 | 22 | 28 | 33 | 0 | 0 |
| gpt-5.5 | 32 | 6 | 37 | 37 | 0 | 0 |
| claude-opus-4-8 | 31 | 5 | 40 | 36 | 0 | 0 |
| gemini-2.5-flash | 31 | 4 | 36 | 35 | 0 | 6 |

| pair | n | exact | binary established |
|---|---|---|---|
| gpt-4.1-mini vs gpt-5.5 | 112 | 0.812 | 0.973 |
| gpt-4.1-mini vs claude-opus-4-8 | 112 | 0.795 | 0.982 |
| gpt-4.1-mini vs gemini-2.5-flash | 106 | 0.830 | 0.981 |
| gpt-5.5 vs claude-opus-4-8 | 112 | 0.964 | 0.991 |
| gpt-5.5 vs gemini-2.5-flash | 106 | 0.934 | 0.991 |
| claude-opus-4-8 vs gemini-2.5-flash | 106 | 0.953 | 1.000 |

Majority distribution: debunked=35, established_fact=31, hedged=5, tie=4, traced_weak_origin=37.

Majority distribution by source:

| source | n | established | hedged | weak/circular | debunked | tie |
|---|---|---|---|---|---|---|
| real_web_case_bank | 40 | 4 | 2 | 7 | 23 | 4 |
| stratified_citation_judge_audit | 72 | 27 | 3 | 30 | 12 | 0 |

Read: exact weak/hedged/debunk distinctions vary somewhat, but binary established-vs-not-established agreement is high, especially among strong judges.

## 7. Artifact Index

- canonical100: `outputs/society/canonical100_extension_summary.md`
- agent_frameworks: `outputs/society/agent_framework_extension_summary.md`
- ecological_validity: `outputs/society/ecological_validity_summary.md`
- harness_validity: `outputs/society/harness_validity_rows.md`
- external_extension: `outputs/society/external_extension_summary.md`
- external100_extension: `outputs/society/external100_extension_summary.md`
- open_weight: `outputs/society/open_model_summary.md`
- human_judge_panel: `outputs/human_eval/judge_panel_summary.md`
- typed_handoff_control: `outputs/society/prov_handoff_condition_summary_20260710.md`
- prose_handoff_control: `outputs/society/prov_handoff_prose_condition_summary_20260710.md`
- scenario_construct_audit: `outputs/human_eval/scenario_audit_summary.json`
- canonical100 scenarios: `outputs/society/citation_scenarios_100_v2.jsonl`
- human annotation items: `outputs/human_eval/study_items.jsonl`
- web annotation app: `web/app/index.html`
