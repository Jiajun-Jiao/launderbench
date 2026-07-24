"""Experiment 2, phase 2: run the LLM controllers on the LLM-AUTHORED worlds (loaded from
outputs/eco_worlds.json). Local & free. Checks the main results — structured controller wins
gold-free; prompt baselines leak/false-witness — reproduce on non-templated authored data."""
import json
import os
import sys

sys.path.insert(0, "/path/to/launderbench")
from eco_worlds import build_queries, validate_and_build
from llm_controller import (
    OpenAICompatibleChatClient,
    StructuredSourceCalibratedController,
    make_llm_controllers,
)
from evaluator import aggregate_results, evaluate_all, format_summary_table

PATH = os.environ.get("ECO_PATH", "outputs/eco_worlds.json")
GEN_MODEL = os.environ.get("GEN_MODEL", "qwen2.5:32b-instruct")
GEN_BASE = os.environ.get("GEN_BASE", "http://localhost:11434/v1")


def _load(path):
    with open(path) as fh:
        txt = fh.read().strip()
    if txt.startswith("["):  # JSON array (eco_worlds.json)
        return json.loads(txt)
    return [json.loads(l) for l in txt.splitlines() if l.strip()]  # JSONL (eco_worlds_full.jsonl)


def main():
    limit = int(os.environ.get("LIMIT", "0"))  # 0 = all worlds
    raws = _load(PATH)
    worlds, queries = [], []
    for i, raw in enumerate(raws):
        if limit and len(worlds) >= limit:
            break
        w, msg = validate_and_build(raw, f"eco_{i:03d}")
        if w is None:
            print(f"skip world {i}: {msg}")
            continue
        worlds.append(w)
        queries.extend(build_queries(w))
    print(f"loaded {len(worlds)} authored worlds, {len(queries)} queries from {PATH}")
    if not worlds:
        return
    gen = OpenAICompatibleChatClient(model=GEN_MODEL, base_url=GEN_BASE, temperature=0.0,
                                     use_structured_output=True)
    ctrls = make_llm_controllers(gen, policies=["local_state", "careful", "source_calibrated"]) + [
        StructuredSourceCalibratedController(client=gen, name="llm_structured")]
    print(f"\n=== LLM controllers ({GEN_MODEL}) on {len(worlds)} LLM-AUTHORED worlds ===")
    print(format_summary_table(aggregate_results(evaluate_all(worlds, queries, ctrls))))


if __name__ == "__main__":
    main()
