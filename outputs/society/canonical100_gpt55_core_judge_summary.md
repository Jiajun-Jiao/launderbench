# Canonical100 Strong-Judge Sensitivity

Judge-only relabeling of existing Canonical100 final answers. Strong judges see only the target claim/value and the final answer, not retrieved evidence or hidden condition.

Records: 2072. Judges: gpt-5.5.

## Usage / Cost

| judge | calls with usage | input tokens | output tokens | estimated cost |
|---|---:|---:|---:|---:|
| gpt-5.5 | 2073 | 761848 | 149158 | $8.2840 |

## Agreement vs production judge

| judge | n | exact label agreement | binary established agreement |
|---|---:|---:|---:|
| gpt-5.5 | 2072 | 0.951 | 0.983 |

## Main Sensitivity Table

| answerer | base trace | gpt-5.5 trace | base flagged | gpt-5.5 flagged | base solid | gpt-5.5 solid |
|---|---:|---:|---:|---:|---:|---:|
| claude-haiku-4-5 | 2/100 | 2/100 | 0/100 | 0/100 | 96/96 | 96/96 |
| gemini-2.5-flash | 29/100 | 28/100 | 0/100 | 0/100 | 96/96 | 96/96 |
| gpt-4.1-mini | 74/100 | 73/100 | 1/100 | 1/100 | 96/96 | 96/96 |
| gpt-4o-mini | 96/100 | 95/100 | 4/100 | 4/100 | 96/96 | 96/96 |
| gpt-5.5 | 0/100 | 0/100 | 0/100 | 0/100 | 92/96 | 94/96 |
| qwen2.5:32b-instruct | 70/100 | 76/100 | 1/100 | 1/100 | 96/96 | 96/96 |
| qwen3:30b-a3b-instruct-2507-q4_K_M | 53/100 | 64/100 | 1/100 | 1/100 | 93/96 | 96/96 |

## Detailed Rates

| answerer | condition | n | base established | gpt-5.5 established |
|---|---|---:|---:|---:|
| claude-haiku-4-5 | flagged | 100 | 0/100 | 0/100 |
| claude-haiku-4-5 | solid | 96 | 96/96 | 96/96 |
| claude-haiku-4-5 | trace | 100 | 2/100 | 2/100 |
| gemini-2.5-flash | flagged | 100 | 0/100 | 0/100 |
| gemini-2.5-flash | solid | 96 | 96/96 | 96/96 |
| gemini-2.5-flash | trace | 100 | 29/100 | 28/100 |
| gpt-4.1-mini | flagged | 100 | 1/100 | 1/100 |
| gpt-4.1-mini | solid | 96 | 96/96 | 96/96 |
| gpt-4.1-mini | trace | 100 | 74/100 | 73/100 |
| gpt-4o-mini | flagged | 100 | 4/100 | 4/100 |
| gpt-4o-mini | solid | 96 | 96/96 | 96/96 |
| gpt-4o-mini | trace | 100 | 96/100 | 95/100 |
| gpt-5.5 | flagged | 100 | 0/100 | 0/100 |
| gpt-5.5 | solid | 96 | 92/96 | 94/96 |
| gpt-5.5 | trace | 100 | 0/100 | 0/100 |
| qwen2.5:32b-instruct | flagged | 100 | 1/100 | 1/100 |
| qwen2.5:32b-instruct | solid | 96 | 96/96 | 96/96 |
| qwen2.5:32b-instruct | trace | 100 | 70/100 | 76/100 |
| qwen3:30b-a3b-instruct-2507-q4_K_M | flagged | 100 | 1/100 | 1/100 |
| qwen3:30b-a3b-instruct-2507-q4_K_M | solid | 96 | 93/96 | 96/96 |
| qwen3:30b-a3b-instruct-2507-q4_K_M | trace | 100 | 53/100 | 64/100 |
