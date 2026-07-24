# Third-Party Notices

LaunderBench code is released under the MIT License (`LICENSE`) and LaunderBench data under
CC BY 4.0 (`LICENSE-DATA`). The release additionally interacts with the following third-party
artifacts.

## Seeded-panel source datasets (MIT License)

The external seeded panels draw distributional seeds (question style, domain hint, answer type,
evidence format) from the following datasets, each distributed under the MIT License by its
authors. Seed-provenance metadata quotes the source question and short evidence-format snippets
under those same licenses; native-control items reproduce original examples under the same
licenses. No other source text is included in the release.

- **ALCE** — https://github.com/princeton-nlp/ALCE (MIT License)
- **ExpertQA** — https://github.com/chaitanyamalaviya/ExpertQA (MIT License)
- **RAGTruth** — https://github.com/ParticleMedia/RAGTruth (MIT License)

## Software dependencies

Framework replications use LangChain, Haystack, LlamaIndex, and gpt-researcher at the pinned
versions recorded in the paper appendix and the `requirements-*.txt` manifests; local open-weight
inference uses Ollama. These packages are installed from their upstream distributions and remain
under their respective upstream licenses; no third-party package code is redistributed in this
release.

## Open-weight model weights

Open-weight models (Qwen2.5, Qwen3, Llama 3.1, Mistral 7B, Gemma 2) are pulled at run time through
Ollama under each model's own upstream license and are not redistributed here.
