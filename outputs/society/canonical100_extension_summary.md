# Canonical100 extension summary

Panel: 100 validated weak-origin scenarios; 96 have validated solid positive controls. Solid missing: S041, S085, S088, S094.

| model | trace launder | trace 95% CI | flagged launder | solid established | discrimination (paired) | API errors |
|---|---:|---:|---:|---:|---:|---:|
| claude-haiku-4-5 | 0.02 (2/100) | [0.01, 0.07] | 0.00 (0/100) | 1.00 (96/96) | 0.98 (n=96) | {'trace': 0, 'flagged': 0, 'solid': 0} |
| gpt-5.5 | 0.00 (0/100) | [0.0, 0.04] | 0.00 (0/100) | 0.96 (92/96) | 0.96 (n=96) | {'trace': 0, 'flagged': 0, 'solid': 0} |
| o3 | 0.05 (5/100) | [0.02, 0.11] | 0.00 (0/100) | 1.00 (96/96) | 0.95 (n=96) | {'trace': 0, 'flagged': 0, 'solid': 0} |
| claude-opus-4-8 | 0.00 (0/100) | [0.0, 0.04] | 0.00 (0/100) | 0.92 (88/96) | 0.92 (n=96) | {'trace': 0, 'flagged': 0, 'solid': 0} |
| claude-sonnet-4-6 | 0.00 (0/100) | [0.0, 0.04] | 0.00 (0/100) | 0.79 (76/96) | 0.79 (n=96) | {'trace': 0, 'flagged': 0, 'solid': 0} |
| gemini-2.5-flash | 0.29 (29/100) | [0.21, 0.39] | 0.00 (0/100) | 1.00 (96/96) | 0.71 (n=96) | {'trace': 0, 'flagged': 0, 'solid': 0} |
| qwen3:30b-a3b-instruct-2507-q4_K_M | 0.53 (53/100) | [0.43, 0.62] | 0.01 (1/100) | 0.97 (93/96) | 0.43 (n=96) | {'trace': 0, 'flagged': 0, 'solid': 0} |
| gpt-4.1 | 0.62 (62/100) | [0.52, 0.71] | 0.00 (0/100) | 1.00 (96/96) | 0.38 (n=96) | {'trace': 0, 'flagged': 0, 'solid': 0} |
| qwen2.5:32b-instruct | 0.70 (70/100) | [0.6, 0.78] | 0.01 (1/100) | 1.00 (96/96) | 0.28 (n=96) | {'trace': 0, 'flagged': 0, 'solid': 0} |
| gpt-4o | 0.72 (72/100) | [0.63, 0.8] | 0.01 (1/100) | 1.00 (96/96) | 0.27 (n=96) | {'trace': 0, 'flagged': 0, 'solid': 0} |
| gpt-4.1-mini | 0.74 (74/100) | [0.65, 0.82] | 0.01 (1/100) | 1.00 (96/96) | 0.24 (n=96) | {'trace': 0, 'flagged': 0, 'solid': 0} |
| gpt-4o-mini | 0.96 (96/100) | [0.9, 0.98] | 0.04 (4/100) | 1.00 (96/96) | 0.03 (n=96) | {'trace': 0, 'flagged': 0, 'solid': 0} |

Trace laundering by topology:

## claude-haiku-4-5
- linear: 1/34 (0.03)
- star: 0/33 (0.00)
- fake_corroboration: 1/33 (0.03)

## gpt-5.5
- linear: 0/34 (0.00)
- star: 0/33 (0.00)
- fake_corroboration: 0/33 (0.00)

## o3
- linear: 4/34 (0.12)
- star: 0/33 (0.00)
- fake_corroboration: 1/33 (0.03)

## claude-opus-4-8
- linear: 0/34 (0.00)
- star: 0/33 (0.00)
- fake_corroboration: 0/33 (0.00)

## claude-sonnet-4-6
- linear: 0/34 (0.00)
- star: 0/33 (0.00)
- fake_corroboration: 0/33 (0.00)

## gemini-2.5-flash
- linear: 16/34 (0.47)
- star: 4/33 (0.12)
- fake_corroboration: 9/33 (0.27)

## qwen3:30b-a3b-instruct-2507-q4_K_M
- linear: 22/34 (0.65)
- star: 12/33 (0.36)
- fake_corroboration: 19/33 (0.58)

## gpt-4.1
- linear: 33/34 (0.97)
- star: 7/33 (0.21)
- fake_corroboration: 22/33 (0.67)

## qwen2.5:32b-instruct
- linear: 28/34 (0.82)
- star: 19/33 (0.58)
- fake_corroboration: 23/33 (0.70)

## gpt-4o
- linear: 32/34 (0.94)
- star: 11/33 (0.33)
- fake_corroboration: 29/33 (0.88)

## gpt-4.1-mini
- linear: 32/34 (0.94)
- star: 14/33 (0.42)
- fake_corroboration: 28/33 (0.85)

## gpt-4o-mini
- linear: 34/34 (1.00)
- star: 29/33 (0.88)
- fake_corroboration: 33/33 (1.00)
