# Provenance Mechanism Gradient

| model | n | cited-edge acc | root-set acc | source-type acc | raw | instruction | diagnose+answer | self-structured | graph | roots | type | license |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-4o-mini | 30 | 0.983 | 0.767 | 0.867 | 26/30 | 12/30 | 15/30 | 4/30 | 22/30 | 6/30 | 0/30 | 0/30 |
| gpt-4.1-mini | 30 | 1.000 | 1.000 | 1.000 | 21/30 | 1/30 | 0/30 | 0/30 | 9/30 | 0/30 | 0/30 | 0/30 |
| claude-haiku-4-5 | 30 | 1.000 | 1.000 | 1.000 | 1/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |

## Raw Laundering Despite Fully Correct Explicit Diagnostic

| model | raw laundering with explicit diagnostic success | first rescue arm counts |
|---|---:|---|
| gpt-4o-mini | 18/26 | {"diagnose_answer": 2, "instruction": 14, "raw_non_launder": 4, "self_structured": 7, "type": 3} |
| gpt-4.1-mini | 21/21 | {"diagnose_answer": 1, "instruction": 20, "raw_non_launder": 9} |
| claude-haiku-4-5 | 1/1 | {"instruction": 1, "raw_non_launder": 29} |
