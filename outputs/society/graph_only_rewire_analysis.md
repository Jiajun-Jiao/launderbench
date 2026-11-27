# Graph-only rewiring analysis

Instrument: `9baaf7477028c400f5ecc64524a2ab93a59c69807f6694e41321c4f01b55686a`; 100 matched scenarios; 10,000 paired/domain-cluster bootstraps.

| model | factorized dep/ind | factorized paired diff [95% CI] | production dep/ind | production paired diff [95% CI] | root IDs + count jointly exact dep/ind |
|---|---:|---:|---:|---:|---:|
| gpt-4o-mini | 1.00/1.00 | +0.00 [0.0, 0.0] | 0.99/0.99 | +0.00 [0.0, 0.0] | 0/100, 0/100 |
| claude-opus-4-8 | 0.09/0.56 | +0.47 [0.37, 0.57] | 0.00/0.50 | +0.50 [0.4, 0.6] | 88/100, 100/100 |
| gpt-4.1 | 0.92/1.00 | +0.08 [0.03, 0.14] | 0.89/1.00 | +0.11 [0.05, 0.17] | 91/100, 99/100 |
| claude-haiku-4-5 | 0.69/0.75 | +0.06 [-0.01, 0.13] | 0.68/0.75 | +0.07 [0.0, 0.14] | 17/100, 88/100 |
| claude-sonnet-4-6 | 0.56/0.65 | +0.09 [0.0, 0.18] | 0.53/0.62 | +0.09 [0.01, 0.17] | 69/100, 100/100 |
| gemini-2.5-flash | 0.93/0.98 | +0.05 [0.0, 0.1] | 0.98/0.98 | +0.00 [-0.03, 0.03] | 84/100, 97/100 |
| o3 | 0.14/0.75 | +0.61 [0.51, 0.71] | 0.12/0.71 | +0.59 [0.49, 0.68] | 100/100, 100/100 |
| gpt-5.5 | 0.14/0.94 | +0.80 [0.72, 0.87] | 0.04/0.93 | +0.89 [0.83, 0.95] | 100/100, 100/100 |
| gpt-4.1-mini | 1.00/1.00 | +0.00 [0.0, 0.0] | 1.00/1.00 | +0.00 [0.0, 0.0] | 35/100, 92/100 |
| gpt-4o | 1.00/1.00 | +0.00 [0.0, 0.0] | 1.00/1.00 | +0.00 [0.0, 0.0] | 59/100, 100/100 |
| qwen2.5:32b-instruct | 0.66/0.59 | -0.07 [-0.13, -0.02] | 0.72/0.66 | -0.06 [-0.12, 0.0] | 4/100, 37/100 |
| qwen3:30b-a3b-instruct-2507-q4_K_M | 0.86/0.77 | -0.09 [-0.15, -0.04] | 0.76/0.69 | -0.07 [-0.13, -0.01] | 1/100, 36/100 |

`factorized` is the runner's separate stance/lineage rubric. `production` is the frozen canonical-panel rubric. Both are blind to arm and expected graph.

`root IDs + count jointly exact` requires unique, exact root IDs and a correct separately declared lineage count.

Recorded successful main-run + production-rejudge cost under the frozen price table: US$8.6010 over priced models (unpriced, usage-only: gemini-2.5-flash, gpt-5.5, o3, qwen2.5:32b-instruct, qwen3:30b-a3b-instruct-2507-q4_K_M). This excludes the discarded compatibility-failure partial run, whose successful-call usage was not persisted.
