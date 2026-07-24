# Caveat-Laundering Pilot

Scenarios: `/path/to/launderbench/outputs/society/citation_scenarios_v2.jsonl`; n=30; render=explicit; judge=`gpt-4.1-mini`.

| model | condition | established fact | 95% CI | verdict distribution |
|---|---|---:|---:|---|
| gpt-4o-mini | visible_caveat | 0/30 (0.000) | [0.0, 0.11] | {"traced_weak_origin": 30} |
| gpt-4o-mini | root_caveat_only | 2/30 (0.067) | [0.02, 0.21] | {"established_fact": 2, "hedged": 15, "traced_weak_origin": 13} |
| gpt-4o-mini | root_hidden | 23/30 (0.767) | [0.59, 0.88] | {"established_fact": 23, "hedged": 6, "traced_weak_origin": 1} |
| gpt-4o-mini | flagged_control | 0/30 (0.000) | [0.0, 0.11] | {"hedged": 5, "traced_weak_origin": 25} |
| claude-haiku-4-5 | visible_caveat | 0/30 (0.000) | [0.0, 0.11] | {"traced_weak_origin": 30} |
| claude-haiku-4-5 | root_caveat_only | 0/30 (0.000) | [0.0, 0.11] | {"traced_weak_origin": 30} |
| claude-haiku-4-5 | root_hidden | 5/30 (0.167) | [0.07, 0.34] | {"established_fact": 5, "hedged": 8, "traced_weak_origin": 17} |
| claude-haiku-4-5 | flagged_control | 0/30 (0.000) | [0.0, 0.11] | {"traced_weak_origin": 30} |
