# Datasheet — LaunderBench / Provenance-Licensed Assertion

Version: 2026-07-10.

## Motivation

LaunderBench evaluates whether LLM research agents distinguish independent support from citation
echoes. The intended use is evaluation and diagnosis of provenance-licensed assertion in research,
RAG, and agent pipelines. It is not a fact-veracity benchmark: the headline claims are fictional and
prior-free by construction so the measured skill is source-graph reading, not parametric knowledge.

## Composition

- **Canonical100 citation panel.** 100 validated weak-origin scenarios across 20 domains and three
  topologies: `linear`, `star`, and `fake_corroboration`.
- **Controls.** Each scenario has a `flagged` weak-origin control. 96 scenarios have validated `solid`
  minimal pairs with independent primary-source support; the four missing solid controls are excluded
  from solid denominators.
- **Seeded external-style panels.** ExpertQA, ALCE, and RAGTruth seed adapters each produce 30
  prior-free fictional scenarios plus solid controls, preserving question/source style without reusing
  real benchmark facts.
- **Legacy/ecological world set.** The repository also includes the earlier Who-Saw-What/E-NPC set:
  250 LLM-authored worlds, 1,575 queries, 126 answer atoms, and provenance-typed local evidence states.

Each citation scenario stores the claim, value, question, page texts, source domains, citation edges,
origin id, flag-page id, topology, and validation metadata. Solid controls add `solid_pages` with
independent primary roots.

## Collection And Validation

Weak-origin citation scenarios are authored by a strong model and structurally wired in code. Admission
requires three gates:

- Programmatic validation: no value leakage in the question, no duplicate domains, no launderer
  self-flags, no smuggled independent primary evidence.
- Prior-freeness validation: an independent strong model must report no prior knowledge of the entity
  or claim.
- Construct validation: an independent strong model confirms that launderer pages are non-doubting,
  the origin is weak, every page traces to a single origin, and no independent primary source is
  present.

Solid controls use analogous gates requiring at least two independent primary sources and no circular
single-origin support.

A separate gold-blinded construct audit covers 40 canonical scenarios, stratified by topology and
shown in shuffled page order. Four complete annotators (three hypothesis-unaware plus the
hypothesis-aware study designer) independently check the common origin, origin weakness, absence of
leaked independent support and self-flagging tells, and matched solid sources. 39/40 pass by
majority (one 2-2 tie on the no-leakage check; the three hypothesis-unaware annotators alone pass
all 40) and all 40 gold origins are recovered; removing the ten scenarios with any annotator dissent
changes no headline trace rate by more than 0.022. The audit records and scoring
scripts are included under `web/scenario_audit/`, `outputs/human_eval/`, and `experiments/`.

## Labels And Metrics

The main label is whether the final answer presents the value as an `established_fact`. Other verdicts
are `hedged`, `traced_weak_origin`, and `absent`. The headline metrics are laundering rate on `trace`,
laundering on `flagged`, establishment on `solid`, and discrimination:

```text
mean over matched scenarios i of
  1[established(solid_i)] - 1[established(trace_i)]
```

Only scenarios with both valid `trace` and `solid` answers enter this paired metric (96 pairs in the
canonical panel before any provider-error exclusions).

Judge robustness is checked with stronger relabeling and a blinded 112-item judge panel; binary
established/not-established agreement is high, while exact hedged-vs-traced labels vary more.

## Sensitive Content

The synthetic benchmark contains fictional entities and no real persons or PII. The real-web pilot is
manually curated from documented public circular-reporting incidents and is reported only as
exploratory ecological evidence.

## Distribution

The intended public release includes code, scenario JSONL files, summary artifacts, and reproducibility
commands. Planned licenses: MIT for code and CC-BY-4.0 for generated data, subject to final repository
packaging and model-output redistribution checks. Runtime secrets such as `.env` and `web/app/secret`
must not be included in release archives.

## Maintenance

The benchmark is evaluation-only and regenerable from scripts and seeds. Future additions should go
through the same prior-freeness and construct-validity gates and should report missing-control
denominators explicitly.
