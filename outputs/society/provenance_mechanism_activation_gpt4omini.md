# Provenance Mechanism Diagnostic

Scenarios: `/path/to/launderbench/outputs/society/citation_scenarios_v2.jsonl`; n=30; arms=raw, instruction, diagnose_answer, self_structured, graph, roots, type, license; judge=`gpt-4.1-mini`.

## gpt-4o-mini

| diagnostic step | accuracy |
|---|---:|
| edge_accuracy | 0.807 |
| all_edges_correct | 0.100 |
| cited_edge_accuracy | 0.983 |
| all_cited_edges_correct | 0.967 |
| root_set_correct | 0.767 |
| root_count_correct | 0.767 |
| origin_quality_correct | 0.733 |
| source_type_correct | 0.867 |
| license_non_established | 0.867 |

| first failed step | n |
|---|---:|
| edge_parsing | 1 |
| none | 21 |
| origin_quality | 1 |
| root_tracing | 7 |

Raw laundering despite fully correct explicit diagnostic: 18/26.

| first rescue arm | n |
|---|---:|
| diagnose_answer | 2 |
| instruction | 14 |
| raw_non_launder | 4 |
| self_structured | 7 |
| type | 3 |

| intervention | laundering | 95% CI |
|---|---:|---:|
| raw | 26/30 (0.867) | [0.7, 0.95] |
| instruction | 12/30 (0.400) | [0.25, 0.58] |
| diagnose_answer | 15/30 (0.500) | [0.33, 0.67] |
| self_structured | 4/30 (0.133) | [0.05, 0.3] |
| graph | 22/30 (0.733) | [0.56, 0.86] |
| roots | 6/30 (0.200) | [0.1, 0.37] |
| type | 0/30 (0.000) | [0.0, 0.11] |
| license | 0/30 (0.000) | [0.0, 0.11] |

| topology | n | first failed | raw launder | instruction launder | diagnose_answer launder | self_structured launder | graph launder | roots launder | type launder | license launder |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fake_corroboration | 10 | {'none': 6, 'root_tracing': 3, 'edge_parsing': 1} | 10/10 | 6/10 | 7/10 | 2/10 | 9/10 | 3/10 | 0/10 | 0/10 |
| linear | 10 | {'root_tracing': 4, 'none': 5, 'origin_quality': 1} | 9/10 | 6/10 | 5/10 | 2/10 | 8/10 | 3/10 | 0/10 | 0/10 |
| star | 10 | {'none': 10} | 7/10 | 0/10 | 3/10 | 0/10 | 5/10 | 0/10 | 0/10 | 0/10 |

