# Graph-only rewiring analysis

Instrument: `9baaf7477028c400f5ecc64524a2ab93a59c69807f6694e41321c4f01b55686a`; 100 matched scenarios; 10,000 paired/domain-cluster bootstraps.

| model | factorized dep/ind | factorized paired diff [95% CI] | production dep/ind | production paired diff [95% CI] | root IDs + count jointly exact dep/ind |
|---|---:|---:|---:|---:|---:|
| gemini-3.1-pro-preview | 0.69/1.00 | +0.31 [0.22, 0.4] | 0.18/0.99 | +0.81 [0.73, 0.88] | 100/100, 100/100 |

`factorized` is the runner's separate stance/lineage rubric. `production` is the frozen canonical-panel rubric. Both are blind to arm and expected graph.

`root IDs + count jointly exact` requires unique, exact root IDs and a correct separately declared lineage count.

Recorded successful main-run + production-rejudge cost under the frozen price table: US$0.0000 over priced models (unpriced, usage-only: gemini-3.1-pro-preview). This excludes the discarded compatibility-failure partial run, whose successful-call usage was not persisted.
