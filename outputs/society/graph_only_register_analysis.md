# Register x edges factorial (graph-only probe)

Production-judge established counts per cell; effects are within-scenario paired means.

| model | registry dep/ind | confessed dep/ind | edge eff (registry) | edge eff (confessed) | register eff (dep) | register eff (ind) |
|---|---:|---:|---:|---:|---:|---:|
| gpt-4o-mini | 99/100, 99/100 | 2/100, 0/100 | +0.00 | -0.02 | +0.97 | +0.99 |
| gemini-2.5-flash | 98/100, 98/100 | 0/100, 0/100 | +0.00 | +0.00 | +0.98 | +0.98 |
| claude-haiku-4-5 | 68/100, 75/100 | 0/100, 0/100 | +0.07 | +0.00 | +0.68 | +0.75 |
| claude-sonnet-4-6 | 53/100, 62/100 | 0/100, 0/100 | +0.09 | +0.00 | +0.53 | +0.62 |
| gpt-4.1 | 89/100, 100/100 | 0/100, 0/100 | +0.11 | +0.00 | +0.89 | +1.00 |
| claude-opus-4-8 | 0/100, 50/100 | 0/99, 0/99 | +0.50 | +0.00 | +0.00 | +0.51 |
| o3 | 12/100, 71/100 | 0/100, 0/100 | +0.59 | +0.00 | +0.12 | +0.71 |
| gpt-5.5 | 4/100, 93/100 | 0/100, 0/100 | +0.89 | +0.00 | +0.04 | +0.93 |
| gpt-4.1-mini | 100/100, 100/100 | 1/100, 0/100 | +0.00 | -0.01 | +0.99 | +1.00 |
| gpt-4o | 100/100, 100/100 | 0/100, 0/100 | +0.00 | +0.00 | +1.00 | +1.00 |
| qwen2.5:32b-instruct | 72/100, 66/100 | 1/100, 1/100 | -0.06 | +0.00 | +0.71 | +0.65 |
| qwen3:30b-a3b-instruct-2507-q4_K_M | 76/100, 69/100 | 4/100, 2/100 | -0.07 | -0.02 | +0.72 | +0.67 |

Register effect = registry minus confessed within the same arm (positive = the
self-confessed origin register suppresses establishment). Edge effect = independent
minus dependent within the same register.

Recorded confessed-row cost under the frozen price table: US$5.6983 over priced models (unpriced, usage-only: gemini-2.5-flash, gpt-5.5, o3, qwen2.5:32b-instruct, qwen3:30b-a3b-instruct-2507-q4_K_M).
