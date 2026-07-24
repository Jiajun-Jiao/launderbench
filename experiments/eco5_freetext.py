"""Free-text re-scoring (addresses the reviewer-consensus 'tautology' critique): the automatic
metrics read the structured fields a controller fills (mode/answer_value/claimed_source_type), so
'zero leakage by construction' partly grades the method's own slots. Here we instead judge each
controller's NATURAL-LANGUAGE reply (resp.text) with an INDEPENDENT LLM judge (gpt-4.1-mini),
and compare struct-field metrics vs free-text-judge metrics per controller. If the structured
controller still wins on prose, the win is real, not an artifact of the representation.
Controllers = local qwen; judge = gpt-4.1-mini. Key from .env, never persisted."""
import json
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, "/path/to/launderbench")
from query_generator import generate_dataset
from baselines import default_controllers
from llm_controller import (
    OpenAICompatibleChatClient,
    StructuredSourceCalibratedController,
    make_llm_controllers,
)
from evaluator import evaluate_response
from eco3_metric_validity import _JUDGE_DEV, JUDGE_SCHEMA, render_knowledge

SEEDS = [int(x) for x in os.environ.get("SEEDS", "7,11").split(",")]
WORLDS = int(os.environ.get("WORLDS", "8"))
QPW = int(os.environ.get("QPW", "5"))
GEN_MODEL = os.environ.get("GEN_MODEL", "qwen2.5:32b-instruct")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4.1-mini")
ORDER = ["global_state", "rag_npc", "llm_local_state", "llm_careful", "llm_source_calibrated", "llm_structured"]


def main():
    gen = OpenAICompatibleChatClient(model=GEN_MODEL, base_url="http://localhost:11434/v1",
                                     temperature=0.0, use_structured_output=True)
    judge = OpenAICompatibleChatClient(model=JUDGE_MODEL, base_url="https://api.openai.com/v1",
                                       api_key=os.environ["OPENAI_API_KEY"], temperature=0.0,
                                       use_structured_output=True)
    det = {c.name: c for c in default_controllers()}
    controllers = [det["global_state"], det["rag_npc"]] + \
        make_llm_controllers(gen, ["local_state", "careful", "source_calibrated"]) + \
        [StructuredSourceCalibratedController(client=gen, name="llm_structured")]
    rows = defaultdict(lambda: defaultdict(list))
    for seed in SEEDS:
        worlds, queries = generate_dataset(WORLDS, QPW, seed)
        by = {w.world_id: w for w in worlds}
        for q in queries:
            w = by[q.world_id]
            fact = w.facts[q.proposition_id]
            ctx_base = {"question": q.question, "the_true_answer": fact.true_value,
                        "what_this_npc_actually_knows": render_knowledge(w, q)}
            for ctrl in controllers:
                resp = ctrl.respond(w, q)
                m = evaluate_response(w, q, resp).metrics
                rows[ctrl.name]["EC_struct"].append(m["epistemic_consistency"])
                rows[ctrl.name]["OLR_struct"].append(m["omniscience_leakage"])
                rows[ctrl.name]["FWR_struct"].append(m["false_witness"])
                rows[ctrl.name]["RAF_struct"].append(m["rumor_as_fact"])
                try:
                    j = judge.complete_json(
                        [{"role": "developer", "content": _JUDGE_DEV},
                         {"role": "user", "content": json.dumps({**ctx_base, "npc_reply": resp.text}, ensure_ascii=True)}],
                        response_schema=JUDGE_SCHEMA)
                    rows[ctrl.name]["EC_text"].append(1.0 if j["only_says_what_npc_could_know"] else 0.0)
                    rows[ctrl.name]["OLR_text"].append(1.0 if j["asserts_true_answer_without_basis"] else 0.0)
                    rows[ctrl.name]["RAF_text"].append(1.0 if j["states_rumor_as_established_fact"] else 0.0)
                except Exception as exc:  # noqa: BLE001
                    print(f"  judge err: {str(exc)[:60]}", flush=True)
        print(f"seed {seed} done", flush=True)

    cost = judge.input_tokens / 1e6 * 0.40 + judge.output_tokens / 1e6 * 1.60
    print(f"\njudge calls={judge.calls} cost=${cost:.3f}")

    def mean(c, m):
        v = rows[c].get(m, [])
        return statistics.mean(v) if v else float("nan")

    print(f"\n=== struct-field metric vs FREE-TEXT judge, per controller ({GEN_MODEL}; judge {JUDGE_MODEL}) ===")
    print(f"  {'controller':22s}{'EC_st':>7}{'EC_tx':>7} | {'OLR_st':>7}{'OLR_tx':>7} | {'RAF_st':>7}{'RAF_tx':>7}")
    summary = {}
    for c in ORDER:
        if c not in rows:
            continue
        summary[c] = {k: round(mean(c, k), 3) for k in
                      ["EC_struct", "EC_text", "OLR_struct", "OLR_text", "FWR_struct", "RAF_struct", "RAF_text"]}
        print(f"  {c:22s}{mean(c,'EC_struct'):>7.2f}{mean(c,'EC_text'):>7.2f} | "
              f"{mean(c,'OLR_struct'):>7.2f}{mean(c,'OLR_text'):>7.2f} | "
              f"{mean(c,'RAF_struct'):>7.2f}{mean(c,'RAF_text'):>7.2f}")
    json.dump(summary, open("outputs/freetext_results.json", "w"), indent=2)
    print("\nwrote outputs/freetext_results.json")
    print("Reading: if llm_structured keeps low OLR_tx / high EC_tx on PROSE, the win is not a struct artifact.")


if __name__ == "__main__":
    main()
