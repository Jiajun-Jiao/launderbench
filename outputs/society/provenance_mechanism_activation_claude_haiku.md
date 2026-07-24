# Provenance Mechanism Diagnostic

Scenarios: `/path/to/launderbench/outputs/society/citation_scenarios_v2.jsonl`; n=30; arms=raw, instruction, diagnose_answer, self_structured, graph, roots, type, license; judge=`gpt-4.1-mini`.

## claude-haiku-4-5

| diagnostic step | accuracy |
|---|---:|
| edge_accuracy | 0.800 |
| all_edges_correct | 0.000 |
| cited_edge_accuracy | 1.000 |
| all_cited_edges_correct | 1.000 |
| root_set_correct | 1.000 |
| root_count_correct | 1.000 |
| origin_quality_correct | 1.000 |
| source_type_correct | 1.000 |
| license_non_established | 1.000 |

| first failed step | n |
|---|---:|
| none | 30 |

Raw laundering despite fully correct explicit diagnostic: 1/1.

| first rescue arm | n |
|---|---:|
| instruction | 1 |
| raw_non_launder | 29 |

| intervention | laundering | 95% CI |
|---|---:|---:|
| raw | 1/30 (0.033) | [0.01, 0.17] |
| instruction | 0/30 (0.000) | [0.0, 0.11] |
| diagnose_answer | 0/30 (0.000) | [0.0, 0.11] |
| self_structured | 0/30 (0.000) | [0.0, 0.11] |
| graph | 0/30 (0.000) | [0.0, 0.11] |
| roots | 0/30 (0.000) | [0.0, 0.11] |
| type | 0/30 (0.000) | [0.0, 0.11] |
| license | 0/30 (0.000) | [0.0, 0.11] |

| topology | n | first failed | raw launder | instruction launder | diagnose_answer launder | self_structured launder | graph launder | roots launder | type launder | license launder |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fake_corroboration | 10 | {'none': 10} | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| linear | 10 | {'none': 10} | 1/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| star | 10 | {'none': 10} | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |

