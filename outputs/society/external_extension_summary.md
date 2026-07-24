# External Benchmark Extension Summary

## Seeded LaunderBench Panels

| benchmark | model | trace | flagged | solid | discrimination |
|---|---|---:|---:|---:|---:|
| expertqa | gpt-4o-mini | 17/30 | 0/30 | 30/30 | 0.43 |
| expertqa | claude-opus-4-8 | 0/30 | 0/30 | 21/30 | 0.70 |
| alce | gpt-4o-mini | 21/30 | 0/30 | 30/30 | 0.30 |
| alce | claude-opus-4-8 | 0/30 | 0/30 | 30/30 | 1.00 |
| ragtruth | gpt-4o-mini | 13/30 | 0/30 | 30/30 | 0.57 |
| ragtruth | claude-opus-4-8 | 0/30 | 0/30 | 29/30 | 0.97 |

## Type-Once-Carry Handoff

| benchmark | Opus typer weak/circular | mini self-downstream | mini self-carry | Opus typed-handoff | mini own field correct |
|---|---:|---:|---:|---:|---:|
| expertqa | 29/30 | 26/30 | 15/30 | 0/30 | 14/30 |
| alce | 30/30 | 27/30 | 18/30 | 0/30 | 9/30 |
| ragtruth | 29/30 | 22/30 | 11/30 | 0/30 | 17/30 |

## Original Native-Control Panels

| benchmark | model | metric | value | extra |
|---|---|---|---:|---|
| expertqa | gpt-4o-mini | accuracy | 0.60 | recall=0.87, specificity=0.33 |
| expertqa | claude-opus-4-8 | accuracy | 0.53 | recall=1.00, specificity=0.07 |
| alce | gpt-4o-mini | match_or_partial | 0.77 | exact=0.20 |
| alce | claude-opus-4-8 | match_or_partial | 0.73 | exact=0.30 |
| ragtruth | gpt-4o-mini | accuracy | 0.73 | recall=0.67, specificity=0.80 |
| ragtruth | claude-opus-4-8 | accuracy | 0.80 | recall=1.00, specificity=0.60 |

## Frozen Inputs

- `outputs/society/seeds_expertqa.jsonl`: `a74020bb9d92e9a64ed94b4def81138b7f20095e350bc1d7cef3683615e5e253`
- `outputs/society/citation_scenarios_expertqa.jsonl`: `1c87f2b14e521d5c03c58bf2a868b4370225194cad5121186b1c84a8113e44a7`
- `outputs/society/citation_scenarios_expertqa_v2.jsonl`: `734ac0a9883ca652ed828ceeda3b18d52b610e045d38dfcb78b65eff007f0a9e`
- `outputs/society/native_cases_expertqa.jsonl`: `0a8dc4cb691e760c399367b453e40f89bc4b003b6d3b86fdee4942aaed08a59f`
- `outputs/society/seeds_alce.jsonl`: `71faa7860898bcfe034d81978ed2d3a2df8454138b7f5871a79dc66c2af52c13`
- `outputs/society/citation_scenarios_alce.jsonl`: `70363e418035f0616f91427198204f50d14154114ddf248a4540ae3c88877b40`
- `outputs/society/citation_scenarios_alce_v2.jsonl`: `6dc93ad72135c96c9855c0a6ef60e2abed73802024fe44ee4034871a9128b5f6`
- `outputs/society/native_cases_alce.jsonl`: `8bdb144e9d8938c21666db14eee7797a5e94acd9e7bef063fde5f5c0b9a56dbb`
- `outputs/society/seeds_ragtruth.jsonl`: `ae5db476760dcd5ccac4924966544470d4e83a8fa7d91ad1084264315e9945fb`
- `outputs/society/citation_scenarios_ragtruth.jsonl`: `6a4ffd4775e3481e1881d5fcf9d3467778e2620f8a4b90fc97dfd377da9faa02`
- `outputs/society/citation_scenarios_ragtruth_v2.jsonl`: `b88d57a516744cef3ac212df4dc9536c10a9ebdb0dfa842dc186e0fe95ce8def`
- `outputs/society/native_cases_ragtruth.jsonl`: `a11133ff3b53081c05c341261e1357eb91788e43a7ec455b2bfa2f426511f6a9`
