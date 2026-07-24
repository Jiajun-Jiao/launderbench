# Harness-validity supplement rows

Scenarios: `citation_scenarios_100_v2.jsonl` for Canonical100 rows unless noted. Judge:
`gpt-4.1-mini`. Script: `experiments/research_citation_multi.py`.

## A. Cheap reasoning / newer mini models

| model | panel | trace launder | flagged | solid accept | discrimination |
|---|---|---:|---:|---:|---:|
| gpt-4o-mini | Canonical100, non-reasoning | 96/100 (.96) | 4/100 | 96/96 | +.04 |
| gpt-4.1-mini | Canonical100, non-reasoning | 74/100 (.74) | 1/100 | 96/96 | +.26 |
| gpt-5-mini | Canonical100, cheap reasoning | 0/100 (.00) | 0/100 | 96/96 | +1.00 |
| gpt-5.4-mini | Canonical100, newer cheap | 7/100 (.07) | 0/100 | 83/96 | +.79 |
| gpt-5.4-nano | Canonical100, newer cheapest | 7/100 (.07) | 0/100 | 67/96 | +.63 |
| o4-mini | Canonical100, cheap reasoning | 32/100 (.32) | 0/100 | 96/96 | +.68 |

Reading: newer/cheap reasoning models reduce laundering substantially, but not all rows are simply
solved without tradeoffs. The strongest cheap row here is gpt-5-mini; gpt-5.4-mini and nano are
nonzero on trace, o4-mini remains visibly vulnerable, and nano is substantially conservative on solid
controls.

## B. CoT-allowed rows

`COT=1` adds a free-form `reasoning` field before `answer` in the same schema, with a neutral scaffold
sentence and no extra provenance wording.

| model | trace (no CoT) | trace (CoT) | flagged (CoT) | solid (CoT) |
|---|---:|---:|---:|---:|
| gpt-4o-mini | 96/100 (.96) | 89/100 (.89) | 3/100 | 94/96 |
| gpt-4.1-mini | 74/100 (.74) | 63/100 (.63) | 0/100 | 96/96 |

Reading: generic reasoning room helps somewhat but leaves high laundering. The stronger interventions
are provenance-directed instructions and typed provenance fields, as shown in the mechanism ladder and
type-once-carry experiments.

Repro:

```bash
SCENARIOS=outputs/society/citation_scenarios_100_v2.jsonl \
CONDS=trace,flagged,solid \
MODEL=$M \
python3 experiments/research_citation_multi.py

SCENARIOS=outputs/society/citation_scenarios_100_v2.jsonl \
CONDS=trace,flagged,solid \
COT=1 \
MODEL=$M \
python3 experiments/research_citation_multi.py
```
