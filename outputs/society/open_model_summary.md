# Open-Weight Citation Panel Summary

Local Ollama models as answerers, `gpt-4.1-mini` as the output classifier.

## Canonical100 Open-Weight Rows

| model | n | trace | flagged | solid | discrimination | read |
|---|---:|---:|---:|---:|---:|---|
| qwen2.5:32b-instruct | 100 | 70/100 | 1/100 | 96/96 | 0.30 | valid discriminator; trace laundering remains |
| qwen3:30b-a3b-instruct-2507-q4_K_M | 100 | 53/100 | 1/100 | 93/96 | 0.44 | valid discriminator; trace laundering remains |

Main-read rows are those that pass the explicit-warning and solid controls cleanly enough to compare against closed hosted models.

## Smaller / Legacy Canonical100 Rows

| model | n | trace | flagged | solid | discrimination | read |
|---|---:|---:|---:|---:|---:|---|
| qwen2.5:7b-instruct | 100 | 88/100 | 22/100 | 94/96 | 0.10 | blanket-trust / weak explicit-warning control |
| llama3.1:8b | 100 | 59/100 | 14/100 | 76/96 | 0.20 | adherence/solid-control confounded |
| mistral:7b | 100 | 72/100 | 20/100 | 89/96 | 0.21 | blanket-trust / weak explicit-warning control |
| gemma2:9b | 100 | 62/100 | 0/100 | 95/96 | 0.37 | valid discriminator; trace laundering remains |

These are useful controls but should be read cautiously when solid or flagged controls are not clean.

## Canonical30 Legacy Appendix

| model | n | trace | flagged | solid | discrimination | read |
|---|---:|---:|---:|---:|---:|---|
| qwen2.5:7b-instruct | 30 | 28/30 | 6/30 | 28/30 | 0.00 | blanket-trust / weak explicit-warning control |
| qwen2.5:32b-instruct | 30 | 15/30 | 0/30 | 30/30 | 0.50 | valid discriminator; trace laundering remains |
| llama3.1:8b | 30 | 13/30 | 4/30 | 21/30 | 0.27 | adherence/solid-control confounded |
| qwen3:30b-a3b-instruct-2507-q4_K_M | 30 | 17/30 | 0/30 | 30/30 | 0.43 | valid discriminator; trace laundering remains |
