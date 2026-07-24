# Provenance Mechanism Diagnostic

Scenarios: `/path/to/launderbench/outputs/society/citation_scenarios_v2.jsonl`; n=10; arms=raw, instruction, diagnose_answer, self_structured, graph, roots, type, license; judge=`gpt-4.1-mini`.

## gpt-4.1-mini

| diagnostic step | accuracy |
|---|---:|
| edge_accuracy | 0.780 |
| all_edges_correct | 0.000 |
| cited_edge_accuracy | 0.975 |
| all_cited_edges_correct | 0.900 |
| root_set_correct | 0.900 |
| root_count_correct | 0.900 |
| origin_quality_correct | 1.000 |
| source_type_correct | 1.000 |
| license_non_established | 1.000 |

| first failed step | n |
|---|---:|
| edge_parsing | 1 |
| none | 9 |

Raw laundering despite fully correct explicit diagnostic: 7/8.

| first rescue arm | n |
|---|---:|
| diagnose_answer | 1 |
| instruction | 7 |
| raw_non_launder | 2 |

| intervention | laundering | 95% CI |
|---|---:|---:|
| raw | 8/10 (0.800) | [0.49, 0.94] |
| instruction | 1/10 (0.100) | [0.02, 0.4] |
| diagnose_answer | 0/10 (0.000) | [0.0, 0.28] |
| self_structured | 0/10 (0.000) | [0.0, 0.28] |
| graph | 1/10 (0.100) | [0.02, 0.4] |
| roots | 0/10 (0.000) | [0.0, 0.28] |
| type | 0/10 (0.000) | [0.0, 0.28] |
| license | 0/10 (0.000) | [0.0, 0.28] |

| topology | n | first failed | raw launder | instruction launder | diagnose_answer launder | self_structured launder | graph launder | roots launder | type launder | license launder |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fake_corroboration | 4 | {'none': 4} | 3/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| linear | 3 | {'edge_parsing': 1, 'none': 2} | 3/3 | 1/3 | 0/3 | 0/3 | 1/3 | 0/3 | 0/3 | 0/3 |
| star | 3 | {'none': 3} | 2/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |

