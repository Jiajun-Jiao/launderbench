# Rendering Ablation Summary

`explicit` is the original `[P1] ... (cites P2)` rendering. `domain_refs` removes page-id citation syntax and renders each result as a source domain, excerpt, and `References listed by this page: <domain>`.

| benchmark | model | explicit trace | domain_refs trace | explicit flagged | domain_refs flagged | explicit solid | domain_refs solid |
|---|---|---:|---:|---:|---:|---:|---:|
| expertqa | gpt-4o-mini | 17/30 | 18/30 | 0/30 | 0/30 | 30/30 | 30/30 |
| expertqa | claude-opus-4-8 | 0/30 | 0/30 | 0/30 | 0/30 | 21/30 | 18/30 |
| alce | gpt-4o-mini | 21/30 | 24/30 | 0/30 | 0/30 | 30/30 | 30/30 |
| alce | claude-opus-4-8 | 0/30 | 0/30 | 0/30 | 0/30 | 30/30 | 27/30 |
| ragtruth | gpt-4o-mini | 13/30 | 16/30 | 0/30 | 0/30 | 30/30 | 30/30 |
| ragtruth | claude-opus-4-8 | 0/30 | 0/30 | 0/30 | 0/30 | 29/30 | 29/30 |

Main read: replacing explicit page-id cites with domain-level references does not remove the core split. GPT-4o-mini still launders circular sourcing; Opus still launders 0/30 on trace across all three panels.
