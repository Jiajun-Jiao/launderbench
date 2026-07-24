# Code Map — LaunderBench / Provenance-Licensed Assertion

> **Release scope note.** The `paper/` LaTeX tree and legacy E-NPC world-set outputs
> (`outputs/eco_*` etc.) are workspace-side; the anonymous release ships the code, scenario sets,
> and every LaunderBench headline artifact, with legacy outputs regenerable via `REPRODUCE.md`.

> **CURRENT (ARR LaTeX, 2026-07-10).** Canonical scripts behind `paper/main.tex` and
> `paper/sections/*.tex`:
> `research_citation_multi.py` (citation laundering / flagged / solid) · `provenance_complexity.py`
> (typing gradient) · `provenance_trace.py` (clean-chain control) · `provenance_pipeline.py`
> (self-compression pilot) · `provenance_relay.py` (multi-hop relay negative) ·
> `provenance_pressure.py` (authority/social pressure negative) · `provenance_handoff.py`
> (type-once-carry fix) · `prov_handoff_analyze.py` / `handoff_condition_analyze.py`
> (single-arm and paired weak/solid handoff tables) ·
> `provenance_mechanism_diagnostic.py` (activation/state-use intervention ladder) ·
> `provenance_mechanism_analyze.py` (mechanism-gradient table) · `caveat_laundering.py`
> (ecosystem caveat-decay pilot) · `audit_gold.py`
> (definition-sensitivity) · `benchmark_seed_adapter.py` / `citation_seeded.py` /
> `external_native_prepare.py` / `external_native_eval.py` / `external_extension_analyze.py` /
> `rendering_ablation_analyze.py` / `native_citation_wrapper_analyze.py` / `open_model_analyze.py`
> / `judge_robustness_audit.py` / `real_web_pilot.py` / `graph_only_rewire.py` /
> `graph_only_rejudge.py` / `graph_only_analyze.py`
> (external-benchmark seeding/native-control extension, judge robustness, real-web pilot,
> graph-only isolation/robustness analysis) · `society_ga.py` (GA-reflection negative) · `failure_breakdown.py`
> (.00 breakdown) + deps (society_reflective, society_sim,
> society_propagation, graph_extract3, eco_worlds, llm_controller, epistemic_state, evaluator).
> Deflated dead-ends → `experiments/_archive/`; superseded logs/json → `outputs/_archive/`; old papers →
> `drafts/`. See `PROGRESS.md` for the honest "what survived / what didn't".

Layout after the 2026-06-28 cleanup. **Core library lives at the repo root; all experiment/method
scripts live in `experiments/`.** Run experiment scripts from the repo root as
`python3 experiments/<name>.py` (Python puts `experiments/` on the path so the scripts import each
other; they also insert the repo root for the core libs; `outputs/` paths are relative to the repo
root). Reproduction commands: `REPRODUCE.md`. The sole paper source of truth is `paper/`
(`main.tex` + `sections/*.tex` + `refs.bib`, official `acl.sty`); build and refresh deliverables only
with `bash scratchpad/final_build_gate.sh` from the repository root. Markdown drafts are historical.

## Core library (repo root — imported by everything; no run needed)
| file | role |
|---|---|
| `world_generator.py` | worlds: locations, NPCs, events, observation assignment, info propagation |
| `epistemic_state.py` | claim tuple `(p,s,a,τ,ρ,v)`, source taxonomy, `allowed_modes_for_claim`, `source_supports_mode` |
| `query_generator.py` | query items + gold (acceptable modes, forbidden claims, true value) |
| `evaluator.py` | metrics: WAA/EC/OLR/FWR/SAA/RAF/FPR/ISU; `evaluate_response`, `aggregate_results` |
| `baselines.py` | deterministic controllers (Global-State, Local-Transcript, RAG, Prompt-only, oracle) |
| `llm_controller.py` | LLM clients (OpenAI-compat + Anthropic), prompt policies, `StructuredSourceCalibratedController` |
| `run_benchmark.py` | CLI entry point |

## experiments/ — dataset generation
| file | role |
|---|---|
| `eco_worlds.py` | LLM-author + strict-validate worlds (the 12-world ecological set); `validate_and_build`, `build_queries` (imported widely) |
| `eco_scale.py` | scale authoring to 250 worlds → `outputs/eco_worlds_full.jsonl` (resumable) |
| `eco_dataset_card.py` | stats + `outputs/eco_dataset_card.md` |
| `eco_fp.py` | false-presupposition worlds (FPR) → `outputs/eco_worlds_fp.jsonl` |
| `eco_narrative.py` | **narrative worlds** (who-saw/who-told embedded in prose) → `outputs/eco_narrative.jsonl` |

## experiments/ — controllers & main results
| file | role |
|---|---|
| `multi_seed_run.py` | multi-model × multi-seed main table (§5) |
| `multi_seed_ablation.py`, `exp2_ablation.py` | clean ablation (§6/§8.2) |
| `combined_exp134.py`, `exp3_pressure.py` | cross-family + pressure + new scenarios (§8.1/8.3/8.4) |
| `claude_run.py`, `gpt41_run.py` | Claude / GPT-4.1-mini families (§8.1; need API keys) |
| `eco_run_llm.py` | run LLM controllers on an authored set (reads `.json` or `.jsonl`; `LIMIT`, `ECO_PATH`) |

## experiments/ — LaunderBench / provenance stress tests
| file | role |
|---|---|
| `citation_scenarios.py`, `citation_solid.py` | scenario synthesis + solid positive-control generation |
| `research_citation_multi.py` | citation laundering panel over `trace` / `flagged` / `solid`; persists row-level answers/errors and reports rates over valid answers (`attempted_n`/`valid_n`/`excluded_n` explicit); `COT=1` adds a neutral scratchpad field |
| `citation_rows_summary.py` | API-free, hash-stamped reaggregation of a persisted citation-panel row file using valid-answer denominators |
| `framework_langchain.py` | ecological-validity replication: same scenarios through LangChain's default tutorial RAG chain (run with `.venv-frameworks/bin/python`) |
| `framework_gptr.py`, `gptr_mock_search.py` | ecological data-void runs: gpt-researcher 0.15.1 against a stateless mock search endpoint (`void` / `reachable` origin conditions, context instrumentation; run with `.venv-gptr/bin/python`) |
| `provenance_complexity.py`, `provenance_trace.py` | controlled source-typing gradient + clean-chain controls |
| `provenance_pipeline.py` | self-compression pilot: direct / briefing / downstream / self-carry |
| `provenance_relay.py` | repeated relay stress test for frontier robustness |
| `provenance_pressure.py` | social/authority pressure stress test for frontier robustness |
| `provenance_handoff.py` | type once with a strong model, carry structured provenance through cheaper carriers; schema v2 preserves full carrier answers and carrier-authored briefings for offline audit |
| `prov_handoff_analyze.py` | side-effect-free Markdown summary table plus stored-answer audit coverage for `prov_handoff*.json` |
| `handoff_condition_analyze.py` | pair trace and solid handoff rows by scenario, then report per-carrier discrimination and paired-bootstrap intervals for typed or explicit-prose arms |
| `scenario_audit_score.py` / `scenario_audit_sensitivity.py` | score the blinded 40-item construct audit and recompute headline trace rates after removing every one-rater dissent item |
| `provenance_mechanism_diagnostic.py` | decomposes citation laundering into edge parsing / root tracing / source typing / speech license, plus raw→instruction→diagnose_answer→self_structured→graph→roots→type→license interventions |
| `provenance_mechanism_analyze.py` | aggregate mechanism diagnostic JSONs into the cross-model mechanism-gradient table |
| `caveat_laundering.py` | ecosystem pilot: visible caveats vs root-only caveat vs root-hidden caveat-stripped echoes |
| `benchmark_seed_adapter.py` | convert ALCE / ExpertQA / RAGTruth / generic JSON(L) into LaunderBench seed specs |
| `citation_seeded.py` | generate prior-free citation-laundering scenarios from external benchmark seed specs |
| `external_native_prepare.py` | prepare original ALCE / ExpertQA / RAGTruth native-control cases without LaunderBench graph rewriting |
| `external_native_eval.py` | run source-grounded answer/support audits on original native-control cases |
| `external_extension_analyze.py` | aggregate seeded-extension, handoff, native-control results and frozen-input hashes |
| `rendering_ablation_analyze.py` | compare explicit page-id citation rendering against domain-level reference-card rendering |
| `native_citation_wrapper_analyze.py` | compare original native controls against verbatim source/evidence plus lightweight citation metadata |
| `open_model_analyze.py` | aggregate local/open-weight citation-panel runs and mark control/adherence caveats |
| `judge_robustness_audit.py` | rerun deterministic citation-panel answers with per-sample records and relabel with base/strong judges |
| `real_web_pilot.py` | real-web circular-reporting case-bank runner; defaults to `outputs/society/real_web_case_bank.json` and compares repeated-only vs provenance/debunk retrieval |
| `build_citation_human_audit.py` | build the blinded 112-item human audit from stratified judge-robustness records plus real-web case-bank answers |
| `citation_human_audit_score.py` | score human provenance labels against base/strong LLM judges and report exact/binary agreement |
| `panel_stats.py` | paired McNemar/model comparisons, paired-96 matched-control discrimination bootstrap (legacy independent bootstrap retained), domain-clustered intervals, and per-domain rates |
| `graph_only_rewire.py` | deterministic five-page graph-only causal instrument and resumable runner: shared within-item bodies/order/warrant, order counterbalanced across items, and matched edge/terminal/indegree counts; one-vs-two claim roots use a fixed non-claim placeholder; build/dry-run are API-free and actual execution requires an explicit spend confirmation; completed outputs are `graph_only_rewire_<model>.{json,rows.jsonl}` |
| `graph_only_rejudge.py` | rejudge the 400 stored graph-only answers with the paper's frozen production rubric without repeating answer/root calls; preserves the original factorized labels, writes row-level audit records, and requires explicit spend confirmation |
| `graph_only_analyze.py` | API-free integrity/statistics/cost aggregation for the completed graph-only run and production rejudge; checks key coverage, duplicates, errors, and instrument hashes, then writes `graph_only_rewire_analysis.{json,md}` |

## experiments/ — validity & reviewer-fixes
| file | role |
|---|---|
| `eco3_metric_validity.py` | LLM-judge vs auto metrics (§11); exports `render_knowledge`, judge schema (imported by others) |
| `eco4_human_sample.py` / `eco4_human_score.py` | human-study kit: sample items / score IAA+agreement |
| `eco5_freetext.py` | **free-text re-scoring** (the controller-debunk: judge prose not struct fields) |
| `eco6_stats.py` | bootstrap CIs + McNemar (structured vs best prompt) |
| `eco_rag.py` | transfer-beyond-games (RAG-assistant framing) |
| `eco2_paraphrase.py` | paraphrase-robustness (weak/secondary check) |
| `eco_overnight_summarize.py` | aggregate `outputs/overnight/*` |

## experiments/ — THE METHOD (provenance extraction; graph_extract series)
| file | role | status |
|---|---|---|
| `provenance_experiment.py` | one-shot LLM extractor (§9 baseline) + extractor-quality curve (decoupled `EXTRACTOR_MODEL`/`CTRL_MODEL`) | baseline |
| `graph_extract.py` | **step-1**: structure (given graph) decides presence+type, LLM grounds value → beats one-shot (.56→.75, OLR .20→.05) | ✓ |
| `graph_extract2.py` | ablation: per-sentence evidentiality from prose is weak (.44) → signal is in relational structure | ablation |
| `graph_extract3.py` | **from narrative**: narrative_oneshot vs narrative_graph vs gold ceiling | ✓ |
| `graph_extract4.py` | **narrative_struct** (decouple reconstruct-graph + propagate) → EC .48→.59, OLR .25→.12 | ✓ best |
| `graph_extract5.py` | self-consistency (K=5) on flow-graph → source-type .39→.44 (modest; residual systematic) | ✓ |

## experiments/ — society level (provenance under propagation)
| file | role |
|---|---|
| `society_propagation.py` | **§7.8 society experiment**: one fact relayed agent-to-agent down a social chain; independent judge reads each agent's prose; measures OLR/FWR/provenance-fidelity by hop for **naive / careful / disciplined** societies. Lifts the construct from single-agent to a population (the C-level generalization). |

## web/app/ — annotation tool (deployed via nginx + pm2)
`server.py` (stdlib HTTP, per-annotator login + assigned subset + server-side collection),
`make_assignments.py` (hashed passwords + item assignment), `ecosystem.config.js` (pm2),
`index.html` (UI). Currently serves the citation/provenance answer audit at `/whosaw/`.
See `outputs/human_eval/annotation_guide.md`.

## Orchestrators
`run_overnight.sh` (extractor curve → FP → scaled-100), `run_refine.sh` (free-text → RAG → stats → full-250).

## Docs
`paper/` (current LaTeX source), `enpc_paper_v3.md` and `drafts/` (historical snapshots),
`enpc_paper_draft_v0.md` (historical working notes), `results_qwen7b.md` (§0–§14 legacy results),
`DATASHEET.md`, `REPRODUCE.md`, `novelty_assessment.md`,
`PROGRESS.md` (status), this file.

## outputs/
`eco_worlds_full.jsonl` (250), `eco_narrative.jsonl` (40), `eco_worlds_fp.jsonl` (30), result JSONs
(`graph_extract*_results.json`, `freetext_results.json`, `rag_transfer_results.json`,
`stats_results.json`, `metric_validity_results.json`, `gpt41_results.json`, …), `human_eval/` (kit),
`overnight/` + `refine/` (pipeline logs), `eco_dataset_card.md`.
