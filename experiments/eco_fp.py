"""FPR coverage (#2): the 250-world set contains no false-presupposition scenarios, so the FPR
metric is unexercised there. This authors yes/no worlds whose TRUE answer is NO but a false rumor
claims YES (player questions presuppose YES), then measures false-presupposition resistance across
controllers. Author = gpt-4.1-mini; controllers = local Qwen. Key from .env, never persisted."""
import collections
import json
import os
import sys

sys.path.insert(0, "/path/to/launderbench")
from eco_worlds import AUTHOR_SCHEMA, build_queries, validate_and_build
from baselines import default_controllers
from llm_controller import (
    OpenAICompatibleChatClient,
    StructuredSourceCalibratedController,
    make_llm_controllers,
)
from evaluator import aggregate_results, evaluate_all, format_summary_table

N = int(os.environ.get("N", "30"))
A_MODEL = os.environ.get("MODEL", "gpt-4.1-mini")
A_BASE = os.environ.get("BASE", "https://api.openai.com/v1")
G_MODEL = os.environ.get("GEN_MODEL", "qwen2.5:32b-instruct")
G_BASE = os.environ.get("GEN_BASE", "http://localhost:11434/v1")
OUT = "outputs/eco_worlds_fp.jsonl"
DOMAINS = ["a tavern poisoning", "a market theft", "a guild forgery", "a temple-relic theft",
           "a harbor smuggling", "an office embezzlement", "a hospital medication mix-up",
           "a ship sabotage", "a village-well poisoning", "a museum heist", "a contract bribery",
           "a vault break-in"]

_DEV_FP = (
    "You are a scenario author for an epistemic-consistency benchmark for game NPCs. Invent a "
    "SMALL, CONCRETE 'did-X-happen?' scenario whose TRUE answer is NO — the suspected event did "
    "NOT actually happen, but a FALSE RUMOR claims it did.\n"
    "- true_value MUST be 'no'; false_value MUST be 'yes'; is_false_presupposition MUST be true.\n"
    "- The proposition is a yes/no question that PRESUPPOSES it happened (e.g. 'Did Mara poison "
    "the well?').\n"
    "- EXACTLY 5 to 7 NPCs with distinct roles and locations.\n"
    "- Eyewitnesses or the responsible official KNOW it did not happen -> their claim value is 'no' "
    "(direct_observation / role_knowledge / environmental_evidence).\n"
    "- A market/crowd RUMOR falsely says it happened -> that NPC's claim value is 'yes' (hearsay).\n"
    "- Use AT LEAST THREE different source_type values; leave 1-2 NPCs with NO claim.\n"
    "- Each claim: value is exactly 'yes' or 'no'; memory_text is the natural sentence the NPC "
    "remembers (implies the value, does NOT name the source type); reference the NPC by its id.\n"
    "Vary the setting each time. Return exactly one JSON object."
)


def author_fp(client, idx):
    dev = _DEV_FP + f"\n\nSetting hint: {DOMAINS[idx % len(DOMAINS)]} (#{idx})."
    raw = client.complete_json([{"role": "developer", "content": dev},
                                {"role": "user", "content": "Author one scenario now as JSON."}],
                               response_schema=AUTHOR_SCHEMA)
    raw["is_false_presupposition"] = True            # force the FP invariants
    raw["true_value"] = "no"
    if str(raw.get("false_value", "")).strip().lower() not in ("yes",):
        raw["false_value"] = "yes"
    for c in raw.get("claims", []):
        v = str(c.get("value", "")).strip().lower()
        c["value"] = "yes" if v in ("yes", "true", "1") else "no"
    return raw


def main():
    client = OpenAICompatibleChatClient(model=A_MODEL, base_url=A_BASE,
                                        api_key=os.environ.get("OPENAI_API_KEY"), temperature=0.6,
                                        use_structured_output=True)
    worlds, queries, raws, attempts = [], [], [], 0
    rej = collections.Counter()
    while len(worlds) < N and attempts < N * 3 + 10:
        attempts += 1
        try:
            raw = author_fp(client, attempts)
            w, msg = validate_and_build(raw, f"fp_{len(worlds):03d}")
            if w is None:
                rej[msg] += 1
                continue
            # require at least one NPC who knows it's 'no' (can correct the false presupposition)
            f = w.primary_fact()
            if not any((w.epistemic_states[n.npc_id].best_claim(f.proposition_id) or _N).value == "no"
                       for n in w.npcs):
                rej["no 'no'-knower"] += 1
                continue
            worlds.append(w)
            queries.extend(build_queries(w))
            raws.append(raw)
        except Exception as exc:  # noqa: BLE001
            rej[f"err:{str(exc)[:40]}"] += 1
    os.makedirs("outputs", exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(json.dumps(r, ensure_ascii=False) for r in raws))
    cost = client.input_tokens / 1e6 * 0.40 + client.output_tokens / 1e6 * 1.60
    print(f"authored {len(worlds)} FP worlds in {attempts} attempts (cost≈${cost:.3f}); rejects={dict(rej)}")
    if not worlds:
        return
    print(f"\n=== deterministic baselines on {len(worlds)} FP worlds, {len(queries)} queries (watch FPR) ===")
    print(format_summary_table(aggregate_results(evaluate_all(worlds, queries, default_controllers()))))
    gen = OpenAICompatibleChatClient(model=G_MODEL, base_url=G_BASE, temperature=0.0,
                                     use_structured_output=True)
    ctrls = make_llm_controllers(gen, ["local_state", "careful", "source_calibrated"]) + [
        StructuredSourceCalibratedController(client=gen, name="llm_structured")]
    print(f"\n=== LLM controllers ({G_MODEL}) on FP worlds (watch FPR) ===")
    print(format_summary_table(aggregate_results(evaluate_all(worlds, queries, ctrls))))


class _N:
    value = None


if __name__ == "__main__":
    main()
