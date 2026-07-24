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

Register effect = registry minus confessed within the same arm (positive = the
self-confessed origin register suppresses establishment). Edge effect = independent
minus dependent within the same register.

Recorded confessed-row cost under the frozen price table: US$4.4578 over priced models (unpriced, usage-only: gemini-2.5-flash, gpt-5.5, o3).
