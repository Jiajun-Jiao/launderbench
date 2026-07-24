# Canonical100 Strong-Judge Sensitivity

Judge-only relabeling of existing Canonical100 final answers. Strong judges see only the target claim/value and the final answer, not retrieved evidence or hidden condition.

Records: 10. Judges: gpt-5.5, claude-opus-4-8.

## Usage / Cost

| judge | calls with usage | input tokens | output tokens | estimated cost |
|---|---:|---:|---:|---:|
| gpt-5.5 | 10 | 3982 | 504 | $0.0350 |
| claude-opus-4-8 | 10 | 6623 | 180 | $0.0376 |

## Agreement vs production judge

| judge | n | exact label agreement | binary established agreement |
|---|---:|---:|---:|
| gpt-5.5 | 10 | 1.0 | 1.0 |
| claude-opus-4-8 | 10 | 1.0 | 1.0 |

## Main Sensitivity Table

| answerer | base trace | gpt-5.5 trace | claude-opus-4-8 trace | base flagged | gpt-5.5 flagged | claude-opus-4-8 flagged | base solid | gpt-5.5 solid | claude-opus-4-8 solid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| claude-haiku-4-5 | - | - | - | 0/10 | 0/10 | 0/10 | - | - | - |

## Detailed Rates

| answerer | condition | n | base established | gpt-5.5 established | claude-opus-4-8 established |
|---|---|---:|---:|---:|---:|
| claude-haiku-4-5 | flagged | 10 | 0/10 | 0/10 | 0/10 |
