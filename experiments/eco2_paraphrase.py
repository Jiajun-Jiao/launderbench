"""Experiment 2 (reliable path): robustness to de-templating.
Keep the programmatic gold (valid atoms / source types), but LLM-paraphrase each query into
fresh natural language, and check the structured win + metrics are stable vs the 4 fixed
templates. Free on local Ollama."""
import dataclasses
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, "/path/to/launderbench")
from query_generator import generate_dataset
from llm_controller import (
    OpenAICompatibleChatClient,
    StructuredSourceCalibratedController,
    make_llm_controllers,
)
from evaluator import aggregate_results, evaluate_all

PARA_SCHEMA = {"type": "object", "additionalProperties": False,
               "properties": {"question": {"type": "string"}}, "required": ["question"]}
_PARA_DEV = (
    "Rephrase the player's question to an NPC in a fresh, natural, conversational way. Keep the "
    "EXACT same meaning, the same subject/target, and the same focus (who/what/where, or yes-no). "
    "Do NOT answer it; do not add new facts. Return JSON {\"question\": ...}."
)
SEEDS = [7, 11, 17]
WORLDS, QPW = 8, 5
MODEL = os.environ.get("MODEL", "qwen2.5:32b-instruct")
BASE = "http://localhost:11434/v1"


def paraphrase(client, q):
    try:
        out = client.complete_json(
            [{"role": "developer", "content": _PARA_DEV}, {"role": "user", "content": f"Question: {q}"}],
            response_schema=PARA_SCHEMA)
        return out.get("question") or q
    except Exception:
        return q


def main():
    gen = OpenAICompatibleChatClient(model=MODEL, base_url=BASE, temperature=0.0, use_structured_output=True)
    par = OpenAICompatibleChatClient(model=MODEL, base_url=BASE, temperature=0.8, use_structured_output=True)
    rows = defaultdict(list)
    samples = []
    for seed in SEEDS:
        worlds, queries = generate_dataset(WORLDS, QPW, seed)
        pq = []
        for q in queries:
            nq = paraphrase(par, q.question)
            pq.append(dataclasses.replace(q, question=nq))
            if len(samples) < 6 and nq != q.question:
                samples.append((q.question, nq))
        for qver, qs in (("templated", queries), ("paraphrased", pq)):
            ctrls = make_llm_controllers(gen, policies=["source_calibrated"]) + [
                StructuredSourceCalibratedController(client=gen, name="llm_structured")]
            agg = aggregate_results(evaluate_all(worlds, qs, ctrls))
            for ctrl, md in agg.items():
                for m, v in md.items():
                    if v is not None:
                        rows[(qver, ctrl.replace("llm_", ""), m)].append(v)
        print(f"seed {seed} done", flush=True)

    print("\nparaphrase samples (T=templated, P=paraphrased):")
    for a, b in samples:
        print(f"  T: {a}\n  P: {b}")

    def mean(qver, cond, m):
        v = rows.get((qver, cond, m), [])
        return statistics.mean(v) if v else float("nan")

    print(f"\n=== templated vs LLM-paraphrased questions ({MODEL}, 3 seeds, mean) ===")
    for cond in ["source_calibrated", "structured"]:
        print(f"[{cond}]")
        for m in ["epistemic_consistency", "omniscience_leakage", "source_attribution_accuracy",
                  "warranted_answer_accuracy"]:
            t, p = mean("templated", cond, m), mean("paraphrased", cond, m)
            print(f"  {m:30s} templated={t:.2f}  paraphrased={p:.2f}  Δ={p-t:+.2f}")


if __name__ == "__main__":
    main()
