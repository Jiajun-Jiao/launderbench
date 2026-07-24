# Reproducibility — LaunderBench / Provenance-Licensed Assertion

> **Release scope note.** Every LaunderBench headline artifact referenced below ships in the
> anonymous release. Paths that belong to the legacy E-NPC world-set experiments (`outputs/eco_*`,
> `outputs/exp*`, `outputs/graph_extract*`, `outputs/claude_results.json`,
> `outputs/gpt41_results.json`, `outputs/freetext_results.json`, and similar) are
> workspace-side only: they support no headline number and are regenerated exactly by the commands
> listed in the legacy section below. The `paper/` LaTeX tree ships separately in the ARR source
> archive, not in the code/data release.

> **Paper build (full submission workspace only):** from the repository root, run
> `bash scratchpad/final_build_gate.sh`. This is the sole publication path: it runs the pinned
> pdflatex/BibTeX sequence, enforces the 8-page main-text and integrity gates, and refreshes the PDF
> plus both source/release archives only after all checks pass. Never edit a generated PDF or zip.

> **CURRENT (ARR LaTeX, 2026-07-10) — reproduce the `paper/` headline results.** First
> load provider keys into the environment (a local `.env` is optional and is excluded from the
> release); OpenAI egress needs sandbox disabled; o3 / gpt-5.5 need
> `OPENAI_API_KEY="$OPENAI_API_KEY_2"`.
> - **Source-dependence laundering headline (Canonical100, trace/flagged/solid, Wilson CI):** per model `M` in
>   {gpt-4o-mini, gpt-4.1-mini, gpt-4o, gpt-4.1, gpt-5.5, o3, claude-haiku-4-5, claude-sonnet-4-6,
>   claude-opus-4-8, gemini-2.5-flash, qwen2.5:32b-instruct, qwen3:30b-a3b-instruct-2507-q4_K_M}:
>   `SCENARIOS=outputs/society/citation_scenarios_100_v2.jsonl SCENARIO_TAG=canonical100 CONDS=trace,flagged,solid MODEL=$M python3 experiments/research_citation_multi.py`
>   (`SCENARIO_TAG=canonical100` keeps the `research_citation_multi_canonical100_*` output naming; the
>   script now aborts if `SCENARIOS` lacks `solid_pages`, e.g. the non-`_v2` v1 file.)
>   Summary: `outputs/society/canonical100_extension_summary.md`.
>   Every new run reports rates over valid answers and separately records `attempted_n`, `valid_n`,
>   `excluded_n`, and `errors_by_type`; provider/format failures and model refusals are not scored as
>   the substantive `absent` label. Recompute this valid-denominator summary from any persisted row
>   file, without API calls, using `python3 experiments/citation_rows_summary.py --scenarios <panel.jsonl> --rows <run.rows.jsonl>`.
>   When both `trace` and `solid` are present, `discrimination` is computed over valid matched
>   scenario/rep pairs; `discrimination_independent_legacy` preserves the former unmatched rate
>   difference for audit.
> - **Datasheet diversity statistics (App.\ A "Diversity"):** `python3 experiments/scenario_diversity_stats.py`
>   → `outputs/society/scenario_diversity_summary.json` (entity/value uniqueness, pairwise Jaccard,
>   distinct n-gram ratios, cross-scenario sentence reuse).
> - **Blinded scenario-construct audit (App. E):** `python3 experiments/scenario_audit_score.py`
>   scores the 40-item stratified audit and writes
>   `outputs/human_eval/scenario_audit_summary.json`; then
>   `python3 experiments/scenario_audit_sensitivity.py` removes every item with any one-rater
>   construct dissent and writes `outputs/human_eval/scenario_audit_sensitivity.json`.
> - **Laundering rationale / failure-mode table (App. F, `tab:error`):** `python3 experiments/laundering_error_analysis.py`
>   — lexical categorization of every `established_fact` `trace` answer into multiplicity / establishment /
>   authority / bare (reads the canonical100 `.rows.jsonl`); prints the per-model table + pooled 84% multiplicity.
> - **Decoding-variance spot-check (Limitations):** `TEMP=0.7 REPS=3 SEED=20260703 SCENARIOS=outputs/society/citation_scenarios_100_v2.jsonl SCENARIO_TAG=var_t07 CONDS=trace MODEL=$M python3 experiments/research_citation_multi.py`
>   for `M` in {gpt-4o-mini, gpt-4.1-mini}; summarize per-rep laundering-rate spread with
>   `python3 experiments/decoding_variance_summary.py`.
> - **Canonical30 subset (mechanism / handoff / framework wings; historical panel):** per model `M`:
>   `SCENARIOS=outputs/society/citation_scenarios_v2.jsonl CONDS=trace,flagged,solid MODEL=$M python3 experiments/research_citation_multi.py`.
> - **Harness-validity rows (low-cost reasoning + CoT-allowed):** same script/panel; Canonical100 rows for
>   `M` in {o4-mini, gpt-5-mini, gpt-5.4-mini, gpt-5.4-nano}:
>   `SCENARIOS=outputs/society/citation_scenarios_100_v2.jsonl SCENARIO_TAG=canonical100 CONDS=trace,flagged,solid MODEL=$M python3 experiments/research_citation_multi.py`;
>   CoT-allowed Canonical100 rows add `COT=1` for `M` in {gpt-4o-mini, gpt-4.1-mini}.
>   Summary: `outputs/society/harness_validity_rows.md`.
> - **Ecological-validity wing (off-the-shelf stacks):** LangChain default tutorial chain (venv
>   `.venv-frameworks/`, pinned versions in run JSON):
>   `CONDS=trace,flagged,solid MODEL=$M .venv-frameworks/bin/python experiments/framework_langchain.py`
>   for `M` in {gpt-4o-mini, claude-opus-4-8}. gpt-researcher 0.15.1 data-void runs (venv `.venv-gptr/`):
>   start `python3 experiments/gptr_mock_search.py` (port 8377), then
>   `CONDS=void,reachable MODEL=$M .venv-gptr/bin/python experiments/framework_gptr.py`
>   for `M` in {gpt-4o-mini, claude-haiku-4-5}. Summary: `outputs/society/ecological_validity_summary.md`.
> - **Source-typing gradient:** `NATURALISTIC=1 K=2 DEPTH=5 REPS=5 MODEL=$M python3 experiments/provenance_complexity.py`
> - **Clean-chain control:** `MODEL=$M python3 experiments/provenance_trace.py` (single chain to depth 6)
> - **Frontier robustness / negative stress tests:** `MODELS=claude-opus-4-8,gpt-5.5,o3 LIMIT=10 python3 experiments/provenance_pressure.py`;
>   `MODEL=claude-opus-4-8 LIMIT=8 HOPS=4 python3 experiments/provenance_relay.py`;
>   `MODELS=claude-opus-4-8,gpt-5.5,o3,gpt-4o-mini LIMIT=10 python3 experiments/provenance_pipeline.py`
> - **Type-once-carry handoff (Canonical100, `tab:handoff`):** the schema-v2 headline replication is
>   `OPENAI_API_KEY="$OPENAI_API_KEY_2" TYPER=claude-opus-4-8 COND=trace ARMS=typed_handoff CARRIERS=gpt-4o-mini,gpt-4.1-mini,gpt-4o,gpt-4.1 SCENARIOS=outputs/society/citation_scenarios_100_v2.jsonl SCENARIO_TAG=canonical100 LIMIT=100 SEED=20260703 OUT=outputs/society/prov_handoff_canonical100_trace_schema2_20260710.json python3 experiments/provenance_handoff.py`.
>   The matched positive-control run changes `COND=solid`, `LIMIT=96`, and
>   `OUT=outputs/society/prov_handoff_canonical100_solid_schema2_20260710.json`.
>   Summarize either artifact with `HANDOFF_JSON=<path> python3 experiments/prov_handoff_analyze.py`,
>   and compute the paired trace/solid result with `python3 experiments/handoff_condition_analyze.py`.
>   The two stored schema-v2 artifacts preserve all 784 complete carrier answers and yield 0/384
>   established on the matched weak-origin side versus 384/384 on the solid side; the additional four
>   unmatched trace items give the reported 0/400 headline. To reproduce the prose-channel positive
>   control, run `OPENAI_API_KEY="$OPENAI_API_KEY_2" TYPER=claude-opus-4-8 COND=solid ARMS=prose_warning CARRIERS=gpt-4o-mini,gpt-4.1-mini,gpt-4o,gpt-4.1 SCENARIOS=outputs/society/citation_scenarios_100_v2.jsonl SCENARIO_TAG=canonical100 LIMIT=96 SEED=20260703 TYPER_FIELDS=outputs/society/prov_handoff_canonical100_solid_schema2_20260710.json OUT=outputs/society/prov_handoff_prose_canonical100_solid_schema2_20260710.json python3 experiments/provenance_handoff_prose.py`.
>   Pair it with the complete stored trace artifact using `TRACE_HANDOFF=outputs/society/prov_handoff_prose_canonical100.json SOLID_HANDOFF=outputs/society/prov_handoff_prose_canonical100_solid_schema2_20260710.json ARM=prose_warning TITLE='Explicit prose handoff' HANDOFF_CONDITION_JSON=outputs/society/prov_handoff_prose_condition_summary_20260710.json HANDOFF_CONDITION_MD=outputs/society/prov_handoff_prose_condition_summary_20260710.md python3 experiments/handoff_condition_analyze.py`;
>   the current result is 0/384 matched weak-origin versus 344/384 solid establishments.
>   The original all-arm artifact remains available for the self-downstream and self-carry rows; a fresh
>   all-arm schema-v2 run is obtained by omitting `ARMS` and `OUT`.
> - **Mechanism diagnostic:** `OPENAI_API_KEY="$OPENAI_API_KEY_2" LIMIT=30 MODELS=gpt-4o-mini JUDGE_MODEL=gpt-4.1-mini ARMS=raw,instruction,diagnose_answer,self_structured,graph,roots,type,license OUT=outputs/society/provenance_mechanism_activation_gpt4omini.json python3 experiments/provenance_mechanism_diagnostic.py`.
>   This prints the activation/state-use ladder and writes
>   `outputs/society/provenance_mechanism_activation_gpt4omini.md`.
>   Repeat for `gpt-4.1-mini` and `claude-haiku-4-5` with `OUT=outputs/society/provenance_mechanism_activation_gpt41mini.json` and
>   `OUT=outputs/society/provenance_mechanism_activation_claude_haiku.json`; aggregate with
>   `python3 experiments/provenance_mechanism_analyze.py`.
> - **Caveat-laundering ecosystem pilot:** `OPENAI_API_KEY="$OPENAI_API_KEY_2" LIMIT=30 MODELS=gpt-4o-mini,claude-haiku-4-5 JUDGE_MODEL=gpt-4.1-mini OUT=outputs/society/caveat_laundering_gpt4omini_haiku.json python3 experiments/caveat_laundering.py`.
> - **Definition-sensitivity audit:** `MODEL=claude-opus-4-8 MAXW=2 python3 experiments/audit_gold.py`
> - **GA-reflection negative result:** `MODEL=$M JUDGE_MODEL=gpt-4.1-mini python3 experiments/society_ga.py`
> - **the .00 breakdown:** `MODEL=$M python3 experiments/failure_breakdown.py`
> - **External-benchmark seeding extension:** convert ALCE / ExpertQA / RAGTruth files into seed specs,
>   generate prior-free controlled scenarios, add `solid` controls, and evaluate:
>   `python3 experiments/benchmark_seed_adapter.py --benchmark expertqa --input outputs/external/expertqa_r2_compiled_anon.jsonl --output outputs/society/seeds_expertqa.jsonl --limit 60 --shuffle`;
>   `SEEDS=outputs/society/seeds_expertqa.jsonl N=30 ID_PREFIX=Q OUT=outputs/society/citation_scenarios_expertqa.jsonl python3 experiments/citation_seeded.py`;
>   `IN=outputs/society/citation_scenarios_expertqa.jsonl OUT=outputs/society/citation_scenarios_expertqa_v2.jsonl python3 experiments/citation_solid.py`;
>   `SCENARIO_TAG=expertqa SCENARIOS=outputs/society/citation_scenarios_expertqa_v2.jsonl CONDS=trace,flagged,solid MODEL=$M python3 experiments/research_citation_multi.py`;
>   `SCENARIO_TAG=expertqa SCENARIOS=outputs/society/citation_scenarios_expertqa_v2.jsonl TYPER=claude-opus-4-8 CARRIERS=gpt-4o-mini LIMIT=30 python3 experiments/provenance_handoff.py`.
>   Repeat with `alce` / `ragtruth` seed files. Aggregate with
>   `python3 experiments/external_extension_analyze.py`; current summary:
>   `outputs/society/external_extension_summary.md`.
> - **Rendering ablation:** rerun the citation panel with natural domain-level reference cards instead of
>   page-id cite syntax:
>   `RENDER_STYLE=domain_refs SCENARIO_TAG=$B SCENARIOS=outputs/society/citation_scenarios_${B}_v2.jsonl CONDS=trace,flagged,solid MODEL=$M python3 experiments/research_citation_multi.py`;
>   aggregate with `python3 experiments/rendering_ablation_analyze.py`.
> - **Original native-control panels:** prepare original-distribution controls without LaunderBench graph
>   rewriting using `python3 experiments/external_native_prepare.py --benchmark $B --output outputs/society/native_cases_$B.jsonl --n 30`, then run
>   `python3 experiments/external_native_eval.py --cases outputs/society/native_cases_$B.jsonl --model $M --tag $B`.
> - **Verbatim native citation wrapper:** keep original source/evidence text unchanged and add only simple
>   reference metadata:
>   `NATIVE_RENDER_STYLE=citation_wrapper python3 experiments/external_native_eval.py --cases outputs/society/native_cases_$B.jsonl --model $M --tag $B`;
>   aggregate with `python3 experiments/native_citation_wrapper_analyze.py`.
> - **Open-weight appendix panel:** for a local Ollama model `M`, run
>   `MODEL=$M SCENARIOS=outputs/society/citation_scenarios_v2.jsonl CONDS=trace,flagged,solid python3 experiments/research_citation_multi.py`;
>   aggregate the current open-weight set with `python3 experiments/open_model_analyze.py`.
> - **Judge robustness audit:** with API keys loaded, run
>   `PANELS=canonical,expertqa,alce,ragtruth MODELS=gpt-4o-mini,claude-opus-4-8 CONDS=trace,flagged,solid BASE_JUDGE=gpt-4.1-mini STRONG_JUDGES=gpt-5.5 python3 experiments/judge_robustness_audit.py`.
>   For the completed stratified smoke used in `PROGRESS.md`, add
>   `MAX_PER_COND=3 CACHE=outputs/society/judge_robustness_records_strat3.jsonl SUMMARY_JSON=outputs/society/judge_robustness_summary_strat3.json SUMMARY_MD=outputs/society/judge_robustness_summary_strat3.md`.
> - **Exploratory real-web pilot:** with API keys loaded, run
>   `MODELS=gpt-4o-mini,claude-opus-4-8 JUDGE_MODEL=gpt-5.5 OUT_JSON=outputs/society/real_web_case_bank_results.json OUT_MD=outputs/society/real_web_case_bank_results.md python3 experiments/real_web_pilot.py`.
>   The case bank is `outputs/society/real_web_case_bank.json`; override with `CASE_BANK=/path/to/cases.json`.
> - **Human citation/provenance audit:** build the blinded audit served by `/whosaw/` with
>   `python3 experiments/build_citation_human_audit.py`, assign annotators with
>   `python3 web/app/make_assignments.py`, restart `pm2 restart whosaw`, then score completed labels with
>   `python3 experiments/citation_human_audit_score.py`. Robustness variants (App G annotator-identity
>   sentence): `EXCLUDE=annot1 python3 experiments/citation_human_audit_score.py` (study designer
>   excluded; binary k=1.000, n=71/111) and `EXCLUDE=annot1,annot2,annot5 ...` (strictest subset:
>   only the two annotators with no prior project involvement remain; binary-unanimous 110/112,
>   k=1.000, n=71/110); snapshot in `outputs/human_eval/audit_score_loo_20260704.txt`.
> - **Disjoint-family re-gate (gate entanglement):** with `GOOGLE_API_KEY` loaded, run
>   `VALIDATOR=gemini-2.5-pro python3 experiments/citation_regate.py` — re-runs G2+G3+solid gates on
>   the canonical 100 with Gemini 2.5 Pro (identical prompts/schemas, imported from the generators)
>   and re-tabulates every model's panel rates on the survivor subset from the existing
>   `.rows.jsonl` records (no new panel calls). Output
>   `outputs/society/citation_regate_gemini25pro.{rows.jsonl,json}`.
> - **Paired/clustered statistics + per-domain table (App E):**
>   `python3 experiments/panel_stats.py` → `outputs/society/panel_stats.json` (exact McNemar between
>   paired models, paired bootstrap CIs on discrimination over the 96 matched trace/solid controls,
>   domain-clustered bootstrap on trace rates, per-domain rates for `tab:domains`). The superseded
>   100-trace-vs-96-solid independent bootstrap remains in
>   `discrimination_ci_independent_legacy`; the preferred paired result is `discrimination_ci`.
>   Set `PANEL_STATS_OUT=/tmp/panel_stats.json` to verify without overwriting the frozen derived file.
> - **Solid-rejection error analysis (Finding 4 / App H):**
>   `python3 experiments/solid_rejection_analysis.py` → `outputs/society/solid_rejection_analysis.json`
>   (categorizes all 28 Sonnet/Opus canonical `solid` non-established answers; also recomputes the
>   sandbox-audit summary rates quoted in the validity analysis).
> - **Seeded-panel text-overlap check (Ethics):** `python3 experiments/seed_overlap_check.py` →
>   `outputs/society/seed_overlap_check.json` (longest shared word n-gram between model-visible
>   seeded-scenario content and its source-dataset seed text).
> - **Prose-warning + field-vs-prose handoff controls:** with keys loaded,
>   `OPENAI_API_KEY=$OPENAI_API_KEY_2 SEED=20260703 LIMIT=100
>   SCENARIOS=outputs/society/citation_scenarios_100_v2.jsonl SCENARIO_TAG=canonical100
>   python3 experiments/provenance_handoff_prose.py` — Opus writes prose-ONLY briefings explicitly
>   instructed to state the provenance conclusion (arm `prose_warning`), and carriers answer from
>   their own confident plain briefing plus Opus's typed field (arm `field_vs_prose`, conflict test;
>   typed fields reused verbatim from `prov_handoff_canonical100.json`). Output
>   `outputs/society/prov_handoff_prose_canonical100.json`.
> - **ExpertQA treat-as-genuine valid-denominator audit (App H):** recompute directly from the frozen
>   row records (no API calls):
>   `python3 experiments/citation_rows_summary.py --scenarios outputs/society/citation_scenarios_expertqa_v2.jsonl --rows outputs/society/research_citation_multi_expertqa_genuine_claude-opus-4-8.rows.jsonl`.
>   This yields `trace` 0/29 valid (one excluded error) and `solid` 24/28 valid (two excluded errors),
>   while retaining attempted n=30 for both. The earlier baseline file predates row persistence and
>   therefore supports only its frozen all-attempt summary (21/30); its 21/28 reinterpretation relies
>   on the contemporaneous refusal audit recorded in `PROGRESS.md`, not on a recoverable baseline row file.
> - **Human SCENARIO construct-validity audit (Appendix G):** build items with
>   `python3 experiments/build_scenario_audit.py` (40 stratified canonical scenarios; gold kept
>   separately in `outputs/human_eval/scenario_audit_gold.jsonl`), assign annotators with
>   `python3 web/scenario_audit/make_assignments.py`, serve with
>   `cd web/scenario_audit && pm2 start ecosystem.config.js` (port 8093, nginx path `/scenaudit/`),
>   score with `python3 experiments/scenario_audit_score.py`. See `web/scenario_audit/README.md`.
>   Chinese reading aids (machine translation, register-preserving; English authoritative) are added
>   by `python3 experiments/translate_scenario_audit.py` (claude-sonnet-4-6, plain-text per-segment
>   calls); restart the server afterwards.
> - **Closed-book prior probe (per evaluated model; Appendix E):** with keys loaded, per lane
>   `SCENARIOS=outputs/society/citation_scenarios_100_v2.jsonl MODELS=<comma list> python3
>   experiments/closed_book_prior.py` (reasoning models on `OPENAI_API_KEY=$OPENAI_API_KEY_2`; local
>   models via Ollama). Forced-guess, no pages; generous lexical hit rule (all-numbers or any
>   distinctive value token not present in the question). Outputs
>   `outputs/society/closed_book_prior_<model>.{json,rows.jsonl}`; every stored hit row is manually
>   inspected.
> - **Instruction-neutrality ablation (Appendix E):** `INSTR=neutral` (task prompt says nothing
>   about evidence strength) or `INSTR=noaccuracy` (drops only the accuracy sentence) with the
>   canonical runner, e.g. `SCENARIOS=outputs/society/citation_scenarios_100_v2.jsonl
>   SCENARIO_TAG=canonical100 CONDS=trace,solid,flagged INSTR=neutral MODEL=gpt-4o-mini python3
>   experiments/research_citation_multi.py` (run for gpt-4o-mini, gpt-4.1-mini, gpt-5.5,
>   claude-sonnet-4-6; noaccuracy for the two minis). Output
>   `research_citation_multi_canonical100_instr_<variant>_<model>.json`.
> - **Low-cost-typer replication (type-once-carry, Appendix E):**
>   `OPENAI_API_KEY=$OPENAI_API_KEY_3 TYPER=gpt-5-mini ARMS=typed_handoff
>   CARRIERS=gpt-4o-mini,gpt-4.1-mini,gpt-4o,gpt-4.1 LIMIT=100 SEED=20260703
>   SCENARIOS=outputs/society/citation_scenarios_100_v2.jsonl
>   OUT=outputs/society/prov_handoff_canonical100_typer_gpt5mini.json python3
>   experiments/provenance_handoff.py` (ARMS=typed_handoff reruns only the typed wing; self arms
>   come from `prov_handoff_canonical100.json`).
> - **Typer cost accounting (Appendix E):** `python3 experiments/typer_cost_calibration.py`
>   (30 measured calls: Opus direct / Opus typing / carrier consume) then
>   `python3 experiments/typer_cost_report.py` (joins calibration, the exact gpt-5-mini run usage,
>   and list prices into per-query costs and break-even k). Outputs
>   `outputs/society/typer_cost_{calibration,report}.json`.
> - **Origin-kind breakdown (App E):** `python3 experiments/origin_kind_breakdown.py` → per-model
>   trace laundering by weak-origin kind, `outputs/society/origin_kind_breakdown.json` (no API calls).
> - **Typer-eligibility sweep (App E):** per lane
>   `SCENARIOS=outputs/society/citation_scenarios_100_v2.jsonl MODELS=<comma list> python3
>   experiments/typing_sweep.py` (reasoning models on `OPENAI_API_KEY=$OPENAI_API_KEY_2`) — the
>   verbatim briefing+field typer instrument on trace (n=100, gold weak/circular) and solid (n=96,
>   gold independent), machine-scored against gold with no carriers or judge; typing correctness is
>   the sufficient statistic for the protocol outcome (validated on three full typer runs: Opus and
>   gpt-5-mini at 100/100 fields -> 0/400 laundering; gpt-4.1-mini at 59/100 ->
>   `prov_handoff_canonical100_typer_gpt41mini.json`, run with `TYPER=gpt-4.1-mini
>   ARMS=typed_handoff ... OUT=...typer_gpt41mini.json` as for the gpt-5-mini variant). Outputs
>   `outputs/society/typing_sweep_<model>.{json,rows.jsonl}`; aggregate the App-E table with
>   `python3 experiments/typing_sweep_report.py` → `outputs/society/typing_sweep_table.json`.
> - **Mixed-graph counting probe (terminal-node dissociation, App E):** per lane
>   `SCENARIOS=outputs/society/citation_scenarios_100_v2.jsonl LIMIT=30 MODELS=<comma list> python3
>   experiments/mixed_graph_probe.py` (reasoning models on `OPENAI_API_KEY=$OPENAI_API_KEY_2`).
>   Three programmatically composed conditions per scenario from already-gated material (trace
>   as-is; mixed = trace minus one in-degree-0 echo plus one solid primary renamed P6; solid
>   as-is), gold independent-primary counts 0/1/3 vs terminal-page counts 1/2/3; the counting
>   question is machine-scored (count + exact page ids), no judge. Outputs
>   `outputs/society/mixed_graph_probe_<model>.{json,rows.jsonl}`.
> - **Graph-only rewiring instrument (completed two-model causal-isolation probe):**
>   `python3 experiments/graph_only_rewire.py build` deterministically creates and validates
>   `outputs/society/graph_only_rewire_instrument.json` from the 100 canonical claims. Each arm has
>   the same claim, question, five page bodies/domains, within-scenario page order, three edges, two
>   terminals, per-page indegrees, and symmetric P2/P3 warrant profile; order is counterbalanced
>   across items, and only the P2/P5 `Reference pointer` lines change within an item. Dependent claim
>   paths end at `{P3}`; independent paths end at `{P2,P3}`. A fixed non-claim
>   P5 is mathematically necessary: if all pages bear the claim, fixed terminal count fixes the
>   number of terminal claim origins. Validate and estimate without creating clients via
>   `python3 experiments/graph_only_rewire.py dry-run --models gpt-4o-mini,claude-opus-4-8 --tasks answer,roots`.
>   The completed main run was
>   `python3 experiments/graph_only_rewire.py run --models gpt-4o-mini,claude-opus-4-8 --tasks answer,roots --confirm-api-spend I_UNDERSTAND`.
>   It produced 800 task rows (400/model: 200 answers and 200 root diagnostics) from 1,200 logical
>   API calls (400 natural answers + 400 blind factorized judges + 400 root diagnostics), with no
>   answer, root, or judge errors. Rejudge the stored answers, without repeating answer/root calls,
>   under the frozen canonical-panel rubric via
>   `python3 experiments/graph_only_rejudge.py --models gpt-4o-mini,claude-opus-4-8 --confirm-api-spend I_UNDERSTAND`;
>   this completed 400 additional judge calls with no errors. Run the deterministic, API-free
>   integrity/statistics/cost aggregation with `python3 experiments/graph_only_analyze.py`.
>
>   The eight-model extension (2026-07-24) adds six snapshots to the original mini/Opus pair:
>   `python3 experiments/graph_only_rewire.py run --models gpt-4.1,claude-haiku-4-5,claude-sonnet-4-6,gemini-2.5-flash --tasks answer,roots --confirm-api-spend I_UNDERSTAND`,
>   then `OPENAI_API_KEY="$OPENAI_API_KEY_2" python3 experiments/graph_only_rewire.py run --models o3,gpt-5.5 --tasks answer,roots --confirm-api-spend I_UNDERSTAND`,
>   then `python3 experiments/graph_only_rejudge.py --models gpt-4.1,claude-haiku-4-5,claude-sonnet-4-6,gemini-2.5-flash,o3,gpt-5.5 --confirm-api-spend I_UNDERSTAND`,
>   then `python3 experiments/graph_only_analyze.py --models gpt-4o-mini,claude-opus-4-8,gpt-4.1,claude-haiku-4-5,claude-sonnet-4-6,gemini-2.5-flash,o3,gpt-5.5`.
>
>   Under the production rubric (dependent/independent established, paired difference): GPT-5.5
>   4/93 (+.89 [.83,.95], exact McNemar p=3.2e-27), o3 12/71 (+.59 [.49,.68], p=3.5e-18), Opus 0/50
>   (+.50 [.40,.60], p=1.8e-15); GPT-4.1 89/100 (+.11), Sonnet 53/62 (+.09, p=.049), Haiku 68/75
>   (+.07, p=.09); Gemini 2.5 Flash 98/98 and gpt-4o-mini 99/99 (+.00). Factorized and production
>   judges agree on .905-.990 of binary verdicts per model. Exact root-ID-set recovery is 88-100%
>   for every model except mini (0/100 both arms); joint IDs-plus-count is 100/100 in both arms for
>   o3 and GPT-5.5, 88/100-100/100 for Opus. Haiku and Sonnet (canonical trace <= .05) establish
>   53-68% of dependent arms under the uniform registry register, so their canonical restraint is
>   register-conditioned; edge sensitivity is established for GPT-5.5, o3, and Opus only.
>
>   Recorded successful main+rejudge cost over priced models (mini, Opus, GPT-4.1, Haiku, Sonnet)
>   is $7.480836 at the 2026-07-07 price table; gemini-2.5-flash, o3, and gpt-5.5 are outside the
>   frozen table, so their API-reported token usage is recorded without an invented price. This
>   excludes successful calls in the discarded compatibility-failure partial run (usage not
>   persisted), assumes invalid HTTP-400 schema requests unbilled, and excludes a discarded
>   all-error Claude attempt during an Anthropic account-limit window
>   (`scratchpad/graph_only_claude_apilimit_20260724/`, zero successful calls). Outputs are
>   `outputs/society/graph_only_rewire_<model>.{json,rows.jsonl}`,
>   `outputs/society/graph_only_rejudge_production_<model>.{json,rows.jsonl}`, and the consolidated
>   `outputs/society/graph_only_rewire_analysis.{json,md}`.
>
> - **Register x edges factorial (confessed-origin row, 2026-07-24):**
>   `python3 experiments/graph_only_rewire.py build --register confessed` creates
>   `outputs/society/graph_only_rewire_confessed_instrument.json`: the same 100 pairs, with only the
>   symmetric P2/P3 root pages rewritten as self-described unverified personal notes (domain plus a
>   constant 46-character confession appended to the body); claim sentence, remaining pages, wiring,
>   order, and counts are unchanged and all invariants revalidate. Run all eight snapshots with
>   `python3 experiments/graph_only_rewire.py run --instrument outputs/society/graph_only_rewire_confessed_instrument.json --models gpt-4o-mini,gpt-4.1,claude-haiku-4-5,claude-sonnet-4-6,claude-opus-4-8,gemini-2.5-flash --tasks answer,roots --confirm-api-spend I_UNDERSTAND`,
>   then `OPENAI_API_KEY="$OPENAI_API_KEY_2" python3 experiments/graph_only_rewire.py run --instrument outputs/society/graph_only_rewire_confessed_instrument.json --models o3,gpt-5.5 --tasks answer,roots --confirm-api-spend I_UNDERSTAND`,
>   then `python3 experiments/graph_only_rejudge.py --instrument outputs/society/graph_only_rewire_confessed_instrument.json --models <all eight> --confirm-api-spend I_UNDERSTAND`,
>   then `python3 experiments/graph_only_register_analyze.py`.
>
>   Production-rubric result: every snapshot establishes 0/100 dependent and 0/100 independent
>   confessed bundles (gpt-4o-mini 2/100 dependent; Opus 0/99 and 0/99, see refusal note), so the
>   register channel moves all eight models while the edge channel (registry row) moves only
>   GPT-5.5, o3, and Opus. Dependent-column register effects: mini +.97, Gemini +.98, GPT-4.1 +.89,
>   Haiku +.68, Sonnet +.53 (exact McNemar p<=2.2e-16), Opus .00, GPT-5.5 +.04, o3 +.12.
>   Factorized and production judges agree on 1.000 of confessed-row binary verdicts per model.
>   Refusal disclosure: claude-opus-4-8 returns a deterministic Anthropic safety refusal on all four
>   G001 calls (the fictional compound name collides with a real explosive); the pair is excluded
>   pairwise and reported (`scratchpad/graph_only_confessed_opus_refusal_20260724/`), and two
>   phantom rejudge rows that had scored the empty answers as `absent` were removed. Recorded
>   confessed-row cost over priced models is $4.457794 at the 2026-07-07 table; gemini-2.5-flash,
>   o3, and gpt-5.5 record token usage only. Outputs are
>   `outputs/society/graph_only_rewire_confessed_<model>.{json,rows.jsonl}`,
>   `outputs/society/graph_only_rejudge_production_confessed_<model>.{json,rows.jsonl}`, and the
>   consolidated `outputs/society/graph_only_register_analysis.{json,md}`.
>
> The section below maps the older v1 (`results_qwen7b.md`) results.

Maps every result in `results_qwen7b.md` and the paper draft (`enpc_paper_draft_v0.md`, Section 20) to the
script, command, and output file that produces it. Findings, tables, and honest caveats live in
`results_qwen7b.md` (Sections 0--12); this file is the *how-to-run*.

## Environment

- **Local (free, default):** Ollama OpenAI-compatible endpoint `http://localhost:11434/v1`.
  Chat models: `qwen2.5:7b-instruct`, `qwen2.5:32b-instruct`, `qwen3:30b-a3b-instruct-2507-q4_K_M`,
  `llama3.1:8b`, `mistral:7b`, `gemma2:9b`. Localhost works inside the sandbox.
- **API (paid, low-cost):** put `OPENAI_API_KEY=sk-...` in repo-root `.env` (one line, no spaces around `=`).
  Source it at runtime: `set -a; . ./.env; set +a`. This project's key grants `gpt-4.1`/`gpt-4.1-mini`/
  `gpt-4.1-nano` (not `gpt-4o-mini`). Anthropic (`claude-haiku-4-5`, `claude-sonnet-4-6`) via an
  `ANTHROPIC_API_KEY` env var supplied ad-hoc. External egress needs the run un-sandboxed.
- No secret is ever written to disk by these scripts; keys are read from env only.
- External benchmark source files are not vendored in this repo. Download ALCE / ExpertQA / RAGTruth
  data separately if you want to run the seeded extension; the adapter only copies style/metadata into a
  seed spec and the generated LaunderBench claims remain prior-free and fictional.

## Core library (no run needed)

`world_generator.py`, `epistemic_state.py`, `query_generator.py`, `evaluator.py`, `baselines.py`,
`llm_controller.py` — generators, metrics, deterministic baselines, and the LLM controllers
(incl. `StructuredSourceCalibratedController`, the proposed method). `run_benchmark.py` is the CLI.

## Result → script → command

| Paper / results section | Script | Command (prefix `set -a; . ./.env; set +a` for API runs) | Output |
|---|---|---|---|
| Sections 2 and 7: main table and structured policy (single-seed) | `run_benchmark.py` | `python3 run_benchmark.py --worlds 12 --queries-per-world 5 --seed 7 --include-llm --llm-model qwen2.5:7b-instruct --llm-base-url http://localhost:11434/v1 --llm-policies global_state,local_state,source_calibrated --by-scenario --jsonl outputs/run.jsonl` | stdout + jsonl |
| Section 5: multi-model × multi-seed (mean±std) | `multi_seed_run.py` | `python3 experiments/multi_seed_run.py` | `outputs/multiseed_results.json` |
| Sections 6 and 8.2: clean ablation | `multi_seed_ablation.py`, `exp2_ablation.py` | `python3 experiments/exp2_ablation.py` | `outputs/exp2_ablation_results.json` |
| Sections 8.1, 8.3, and 8.4: cross-family, pressure, and new scenarios | `combined_exp134.py` (+ `exp3_pressure.py`) | `python3 experiments/combined_exp134.py` | `outputs/exp134_results.json` |
| Section 8.1: Claude family (API) | `claude_run.py` | `ANTHROPIC_API_KEY=… python3 experiments/claude_run.py` | `outputs/claude_results.json` |
| Section 8.1: GPT-4.1-mini family (API) | `gpt41_run.py` | `python3 experiments/gpt41_run.py` | `outputs/gpt41_results.json` |
| Sections 9 and 20.7: provenance extraction + **extractor curve** | `provenance_experiment.py` | per-extractor: `EXTRACTOR_MODEL=… EXTRACTOR_BASE=… CTRL_MODEL=qwen2.5:32b-instruct SEEDS=7,11 WORLDS=5 QPW=4 python3 experiments/provenance_experiment.py` (emits a `CURVE_JSON` line) | stdout / `outputs/overnight/curve_*.txt` |
| Section 10: ecological worlds (12) | `eco_worlds.py` → `eco_run_llm.py` | `BASE=https://api.openai.com/v1 MODEL=gpt-4.1-mini N=12 python3 experiments/eco_worlds.py` then `python3 experiments/eco_run_llm.py` | `outputs/eco_worlds.json` |
| Section 10.1: **scaled 250-world dataset** + card | `eco_scale.py` → `eco_dataset_card.py` | `N=250 MODEL=gpt-4.1-mini python3 experiments/eco_scale.py` then `python3 experiments/eco_dataset_card.py` | `outputs/eco_worlds_full.jsonl`, `outputs/eco_dataset_card.md` |
| Section 10.1: method at scale | `eco_run_llm.py` | `GEN_MODEL=qwen2.5:32b-instruct ECO_PATH=outputs/eco_worlds_full.jsonl LIMIT=30 python3 experiments/eco_run_llm.py` | stdout / `outputs/overnight/full250.txt` |
| Section 11: metric validity (LLM judge) | `eco3_metric_validity.py` | `SEEDS=7,11 WORLDS=6 QPW=4 N_JUDGE=150 python3 experiments/eco3_metric_validity.py` | `outputs/metric_validity_results.json` |
| Section 20.x: FPR coverage (false-presupposition) | `eco_fp.py` | `N=30 GEN_MODEL=qwen2.5:32b-instruct python3 experiments/eco_fp.py` | `outputs/eco_worlds_fp.jsonl` |
| #1 human validation kit | `eco4_human_sample.py` → web tool → `eco4_human_score.py` | `python3 experiments/eco4_human_sample.py` (items + key + LLM judge); annotate; `python3 experiments/eco4_human_score.py` | `outputs/human_eval/` |

## Unattended pipeline

`bash run_overnight.sh` runs, sequentially (one GPU): the extractor curve (3 extractors, fixed
controller), the FP scenarios, and the full 250-world LLM eval. Logs land in `outputs/overnight/`;
`outputs/overnight/DONE.txt` appears on completion.

## Reviewer-fix experiments (v1, addressing the 3-reviewer adversarial pass)

| Fix | Script | Command (prefix `set -a; . ./.env; set +a` for the judge) | Output |
|---|---|---|---|
| Free-text re-scoring (judge prose, not struct fields) | `eco5_freetext.py` | `SEEDS=7,11 WORLDS=8 QPW=5 python3 experiments/eco5_freetext.py` | `outputs/freetext_results.json` |
| Transfer beyond games (RAG assistant) | `eco_rag.py` | `SEEDS=7,11 WORLDS=8 QPW=5 python3 experiments/eco_rag.py` | `outputs/rag_transfer_results.json` |
| Bootstrap CIs + McNemar (structured vs best prompt) | `eco6_stats.py` | `SEEDS=7,11,17 WORLDS=8 QPW=5 python3 experiments/eco6_stats.py` | `outputs/stats_results.json` |
| Full-250 all LLM controllers | `eco_run_llm.py` | `GEN_MODEL=qwen2.5:32b-instruct ECO_PATH=outputs/eco_worlds_full.jsonl LIMIT=0 python3 experiments/eco_run_llm.py` | `outputs/refine/full250_all.txt` |
| All four, sequential (one GPU) | `run_refine.sh` | `bash run_refine.sh` | `outputs/refine/` + `DONE.txt` |

Paper artifacts: clean repositioned draft **`enpc_paper_v1.md`**; **`DATASHEET.md`** (Gebru format);
working notes `enpc_paper_draft_v0.md`.

## Method — provenance extraction (the graph_extract series)

All default to local `MODEL=qwen2.5:32b-instruct`; narrative scripts read `outputs/eco_narrative.jsonl`.

| step | script | command (from repo root) | output |
|---|---|---|---|
| one-shot baseline + extractor curve | `provenance_experiment.py` | `EXTRACTOR_MODEL=… CTRL_MODEL=qwen2.5:32b-instruct python3 experiments/provenance_experiment.py` | stdout (`CURVE_JSON`) |
| step-1 graph-grounded (given graph) | `graph_extract.py` | `SEEDS=7,11 WORLDS=5 QPW=4 python3 experiments/graph_extract.py` | `outputs/graph_extract_results.json` |
| prose-evidentiality ablation | `graph_extract2.py` | `python3 experiments/graph_extract2.py` | `outputs/graph_extract2_results.json` |
| narrative worlds (data) | `eco_narrative.py` | `N=40 python3 experiments/eco_narrative.py` | `outputs/eco_narrative.jsonl` |
| from-narrative (oneshot / graph / gold) | `graph_extract3.py` | `python3 experiments/graph_extract3.py` | `outputs/graph_extract3_results.json` |
| narrative_struct (decouple + propagate) | `graph_extract4.py` | `python3 experiments/graph_extract4.py` | `outputs/graph_extract4_results.json` |
| self-consistency (K=5) | `graph_extract5.py` | `K=5 python3 experiments/graph_extract5.py` | `outputs/graph_extract5_results.json` |
| Section 7.8: society (provenance under propagation) | `society_propagation.py` | `set -a; . ./.env; set +a` then `CHAIN_LEN=6 N_FACTS=0 SEEDS=7,11 MODEL=qwen2.5:32b-instruct JUDGE_MODEL=gpt-4.1-mini JUDGE_BASE=https://api.openai.com/v1 JUDGE_API_KEY=$OPENAI_API_KEY OUT=outputs/society/society_propagation_results.json python3 experiments/society_propagation.py` (judge needs egress / un-sandboxed; agents local) | `outputs/society/society_propagation_results.json` |

## Notes / known data caveats

- LLM controllers are evaluated **gold-free** (never see labels).
- Local 32B cannot reliably author clean atom-valued worlds (~1/9); use `gpt-4.1-mini` as author (≈12/13).
- The scaled set's questions for code-answer worlds were terse labels; the human tool renders them as
  natural questions, and `source_agent` ids are resolved to names (`eco3_metric_validity.render_knowledge`).
- `_answer_matches` (evaluator) uses whole-token containment; names/codes are unambiguous.
