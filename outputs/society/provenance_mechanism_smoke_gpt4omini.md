# Provenance Mechanism Diagnostic

Scenarios: `/path/to/launderbench/outputs/society/citation_scenarios_v2.jsonl`; n=3; arms=raw, graph, roots, type, license; judge=`gpt-4.1-mini`.

## gpt-4o-mini

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
| none | 3 |

| intervention | laundering | 95% CI |
|---|---:|---:|
| raw | 3/3 (1.000) | [0.44, 1.0] |
| graph | 2/3 (0.667) | [0.21, 0.94] |
| roots | 0/3 (0.000) | [0.0, 0.56] |
| type | 0/3 (0.000) | [0.0, 0.56] |
| license | 0/3 (0.000) | [0.0, 0.56] |

| topology | n | first failed | raw launder | type launder | license launder |
|---|---:|---|---:|---:|---:|
| fake_corroboration | 1 | {'none': 1} | 1/1 | 0/1 | 0/1 |
| linear | 1 | {'none': 1} | 1/1 | 0/1 | 0/1 |
| star | 1 | {'none': 1} | 1/1 | 0/1 | 0/1 |

