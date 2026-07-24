# Native Citation Wrapper Summary

Original benchmark text is kept verbatim. The wrapper only adds `References listed by this source: ...` metadata to each original source/evidence item. Metrics remain each benchmark's native control metric, not LaunderBench laundering.

| benchmark | model | metric | plain | citation_wrapper | extra plain | extra wrapper |
|---|---|---|---:|---:|---|---|
| expertqa | gpt-4o-mini | accuracy | 0.60 | 0.67 | recall=0.87, specificity=0.33 | recall=0.87, specificity=0.47 |
| expertqa | claude-opus-4-8 | accuracy | 0.53 | 0.57 | recall=1.00, specificity=0.07 | recall=1.00, specificity=0.13 |
| alce | gpt-4o-mini | match_or_partial | 0.77 | 0.70 | exact=0.20 | exact=0.27 |
| alce | claude-opus-4-8 | match_or_partial | 0.73 | 0.73 | exact=0.30 | exact=0.40 |
| ragtruth | gpt-4o-mini | accuracy | 0.73 | 0.80 | recall=0.67, specificity=0.80 | recall=0.80, specificity=0.80 |
| ragtruth | claude-opus-4-8 | accuracy | 0.80 | 0.80 | recall=1.00, specificity=0.60 | recall=1.00, specificity=0.60 |

Main read: adding lightweight citation metadata to original, non-rephrased benchmark inputs does not create the seeded LaunderBench split; native performance remains broadly similar.
