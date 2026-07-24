# P1-P3 Expansion Plan

Run order is intentionally conservative: first produce main-text rows, then mechanism/framework rows, then external/smaller-model robustness rows.

## P1. Main-Text Expansion

1. Open-weight Canonical100
   - Already complete: `qwen2.5:32b-instruct`
   - Run/resume: `qwen3:30b-a3b-instruct-2507-q4_K_M`

2. Harness/mechanism Canonical100
   - CoT/scratchpad: `gpt-4o-mini`, `gpt-4.1-mini`
   - Cheap reasoning controls: `o4-mini`, `gpt-5-mini`, `gpt-5.4-mini`, `gpt-5.4-nano`
   - Provenance mechanism ladder: `gpt-4o-mini`, `gpt-4.1-mini`, `claude-haiku-4-5`
   - Compression / typed-carry pipeline: `gpt-4o-mini`, `claude-opus-4-8`, `gpt-5.5`, `o3`
   - Type-once-carry handoff: `gpt-4o-mini`, `gpt-4.1-mini`, `qwen2.5:32b-instruct`, `qwen3:30b-a3b-instruct-2507-q4_K_M`

3. Complete-context framework RAG Canonical100
   - LangChain: `gpt-4o-mini`, `claude-haiku-4-5`
   - Haystack: `gpt-4o-mini`, `claude-haiku-4-5`
   - LlamaIndex: `gpt-4o-mini`, `claude-haiku-4-5`

## P2. Agentic And External Expansion

1. Agentic retrieval Canonical100
   - Haystack agentic: `gpt-4o-mini`, conditions `void,reachable`
   - LlamaIndex agentic: `gpt-4o-mini`, conditions `void,reachable`
   - gpt-researcher: `gpt-4o-mini`, conditions `void,reachable`

2. External seeded benchmark panels to 100
   - Generate/validate `expertqa100`, `alce100`, `ragtruth100`
   - Run seeded LaunderBench panels with `gpt-4o-mini`, `claude-opus-4-8`
   - Run type-once-carry handoff with `gpt-4o-mini`

3. External native-control panels to 100
   - Prepare native cases for ExpertQA, ALCE, RAGTruth
   - Run plain and citation-wrapper native controls with `gpt-4o-mini`, `claude-opus-4-8`

## P3. Smaller Open-Weight Controls

1. Canonical100 local models
   - `qwen2.5:7b-instruct`
   - `llama3.1:8b`
   - `mistral:7b`
   - `gemma2:9b`

2. Reporting
   - Rebuild Canonical100 summary
   - Rebuild open-weight summary
   - Rebuild all-results overview

