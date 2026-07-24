# Provenance Mechanism Diagnostic

Scenarios: `/path/to/launderbench/outputs/society/citation_scenarios_v2.jsonl`; n=30; arms=raw, graph, roots, type, license; judge=`gpt-4.1-mini`.

## gpt-4o-mini

| diagnostic step | accuracy |
|---|---:|
| edge_accuracy | 0.833 |
| all_edges_correct | 0.200 |
| cited_edge_accuracy | 0.992 |
| all_cited_edges_correct | 0.967 |
| root_set_correct | 0.700 |
| root_count_correct | 0.700 |
| origin_quality_correct | 0.700 |
| source_type_correct | 0.800 |
| license_non_established | 0.800 |

| first failed step | n |
|---|---:|
| edge_parsing | 1 |
| none | 21 |
| root_tracing | 8 |

Raw laundering despite fully correct explicit diagnostic: 13/22.

| first rescue arm | n |
|---|---:|
| graph | 2 |
| raw_non_launder | 8 |
| roots | 14 |
| type | 6 |

| intervention | laundering | 95% CI |
|---|---:|---:|
| raw | 22/30 (0.733) | [0.56, 0.86] |
| graph | 22/30 (0.733) | [0.56, 0.86] |
| roots | 6/30 (0.200) | [0.1, 0.37] |
| type | 0/30 (0.000) | [0.0, 0.11] |
| license | 0/30 (0.000) | [0.0, 0.11] |

| topology | n | first failed | raw launder | type launder | license launder |
|---|---:|---|---:|---:|---:|
| fake_corroboration | 10 | {'none': 5, 'root_tracing': 4, 'edge_parsing': 1} | 10/10 | 0/10 | 0/10 |
| linear | 10 | {'root_tracing': 4, 'none': 6} | 8/10 | 0/10 | 0/10 |
| star | 10 | {'none': 10} | 4/10 | 0/10 | 0/10 |

