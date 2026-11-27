# LaunderBench

Code, data, and row-level outputs for the paper **LaunderBench: A Controlled Diagnostic of Provenance
Use and Handoffs in LLM Research Agents** (Jiajun Jiao, Bowen Liao, Mingqian Sun; EACL 2027).

Benchmark + experiments for *source-dependence citation laundering*: whether LLM research agents distinguish
independent corroboration from circular reporting that traces to a single weak origin.

- `outputs/society/citation_scenarios_100_v2.jsonl` — the canonical 100 model-gated, prior-free
  weak-origin scenarios (20 domains x 3 topologies) with 96 matched `solid` controls.
- `outputs/society/citation_scenarios_{expertqa,alce,ragtruth}_v2.jsonl` — seeded external panels.
- `experiments/` — generators, gates, panels, mechanism/handoff diagnostics, statistics, audits.
- `outputs/society/research_citation_multi_canonical100_*.rows.jsonl` — per-scenario row-level
  records for every model in the headline table.
- `outputs/society/graph_only_rewire_analysis.{json,md}` — consolidated graph-only rewiring
  integrity/statistics report for the twelve panel models; its raw answer/root and
  production-rejudge rows are alongside it.
- `outputs/society/prov_handoff_condition_summary_20260710.{json,md}` — paired weak/solid
  typed-handoff control; the adjacent schema-v2 artifacts preserve all 784 carrier answers.
- `outputs/society/prov_handoff_prose_condition_summary_20260710.{json,md}` — the matched
  explicit-prose channel control and its complete solid-answer artifact.
- `experiments/provenance_pipeline.py` — defines the handoff record (`value`, `source_type`,
  `verified`, `basis`). `PROV_SOURCE_COUNT=1` adds the optional `independent_source_count` field,
  which covers the case the four-value `source_type` cannot express: one real source that other
  pages repeat. The field is off by default and no reported number uses it (see `REPRODUCE.md`).
- `outputs/human_eval/` + `web/app/`, `web/scenario_audit/` — blinded answer-label audits,
  the completed scenario-construct audit and sensitivity analysis, and the annotation tooling.
- `REPRODUCE.md` — the exact command for every number in the paper.
- `tests/` — the release-level regression and artifact-integrity tests.
- `requirements*.txt` — isolated top-level dependency pins for core and framework replications.

## Environment and tests

The core benchmark and tests use only the Python standard library and were tested with CPython
3.13.13. Create a clean environment, run `python -m pip install -r requirements.txt`, then run
`python -m unittest discover -s tests`. Framework replications use separate environments because
their provider-SDK versions differ; install the matching `requirements-*.txt` file before running
the command documented in `REPRODUCE.md`.

## Licenses

Code is released under the MIT License (`LICENSE`). Data under `outputs/` are released under
CC BY 4.0 (`LICENSE-DATA`). Annotator identities are pseudonymized (`annot1..5`, `audit1..3`);
API keys, live credentials, and personal data are not included.

## Citation

```bibtex
@inproceedings{jiao2027launderbench,
  title     = {{LaunderBench}: A Controlled Diagnostic of Provenance Use and Handoffs in {LLM} Research Agents},
  author    = {Jiao, Jiajun and Liao, Bowen and Sun, Mingqian},
  booktitle = {Proceedings of EACL 2027},
  year      = {2027},
  note      = {To appear}
}
```
