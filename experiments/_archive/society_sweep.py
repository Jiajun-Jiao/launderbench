"""Cross-model society sweep -> the formal curves for agent_society_direction.md.
Runs the society-sim (graph + multi-round propagation + injected false rumor) across MODELS x SEEDS
with a FIXED independent judge, and aggregates:
  - OLR / PF by HOP-DEPTH (the decay curve), naive vs disciplined, per agent-model;
  - BCR (structural containment), naive vs disciplined, over seeds (model-independent).
Agent model varies; the judge is held constant (the measurement instrument). Local models free; gpt-*/
claude-* via API; reasoning (o1) routed to OpenAI. Env: N(40) ROUNDS(12) SHORTCUTS(6) SEEDS(7,11)
MODELS(...) JUDGE_MODEL(gpt-4.1-mini).
"""
import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import society_sim as ss
from llm_controller import AnthropicChatClient, OpenAICompatibleChatClient

N = int(os.environ.get("N", "40"))
ROUNDS = int(os.environ.get("ROUNDS", "12"))
SHORTCUTS = int(os.environ.get("SHORTCUTS", "6"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "7,11").split(",") if s.strip()]
DEFAULT_MODELS = "qwen2.5:7b-instruct,llama3.1:8b,gpt-4.1-mini,gpt-4.1,claude-sonnet-4-6"
MODELS = [m.strip() for m in os.environ.get("MODELS", DEFAULT_MODELS).split(",") if m.strip()]
OLLAMA = os.environ.get("OLLAMA_BASE", "http://localhost:11434/v1")
PRICE = {"gpt-4.1": (2.0, 8.0), "gpt-4.1-mini": (0.40, 1.60), "gpt-4.1-nano": (0.10, 0.40),
         "claude-sonnet-4-6": (3.0, 15.0), "claude-haiku-4-5": (0.80, 4.0)}


def client_for(model):
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        return OpenAICompatibleChatClient(model=model, base_url="https://api.openai.com/v1",
                                          api_key=os.environ.get("OPENAI_API_KEY"), use_structured_output=True)
    if model.startswith("claude-"):
        return AnthropicChatClient(model=model, api_key=os.environ.get("ANTHROPIC_API_KEY", ""), max_tokens=1024)
    return OpenAICompatibleChatClient(model=model, base_url=OLLAMA, use_structured_output=True)


def main():
    true_v, false_v, question = ss.FACT
    judge_model = os.environ.get("JUDGE_MODEL", "gpt-4.1-mini")
    judge = OpenAICompatibleChatClient(model=judge_model, base_url=os.environ.get("JUDGE_BASE", "https://api.openai.com/v1"),
                                       api_key=os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY"),
                                       use_structured_output=True)

    # --- structural BCR (model-independent): one per seed ---
    beliefs = {}
    bcr_naive, bcr_disc = [], []
    for seed in SEEDS:
        adj = ss.build_graph(N, seed, SHORTCUTS)
        rumor_origin = N // 2
        b = ss.propagate(adj, 0, rumor_origin, true_v, false_v, ROUNDS)
        beliefs[seed] = b
        bcr_naive.append(ss.bcr(b, N, "naive"))
        bcr_disc.append(ss.bcr(b, N, "disciplined"))
    avg = lambda xs: round(sum(xs) / len(xs), 3) if xs else 0.0  # noqa: E731
    print(f"society: N={N} rounds={ROUNDS} shortcuts={SHORTCUTS} seeds={SEEDS} models={MODELS}", flush=True)
    print(f"structural BCR (firm false belief): naive={avg(bcr_naive)}  disciplined={avg(bcr_disc)}", flush=True)

    results = {"config": {"N": N, "ROUNDS": ROUNDS, "SHORTCUTS": SHORTCUTS, "SEEDS": SEEDS,
                          "judge": judge_model},
               "BCR": {"naive": avg(bcr_naive), "disciplined": avg(bcr_disc)}, "models": {}}

    for model in MODELS:
        try:
            agent_c = client_for(model)
            buck = {arm: defaultdict(lambda: {"n": 0, "olr": 0, "pf": 0, "fwr": 0})
                    for arm in ("naive", "disciplined")}
            for seed in SEEDS:
                belief = beliefs[seed]
                for arm in ("naive", "disciplined"):
                    for i, b in belief.items():
                        name = ss.NAMES[i % len(ss.NAMES)]
                        txt = ss.utterance(agent_c, arm, name, b, question, 0.7 if arm == "naive" else 0.0)
                        sc = ss.score_utterance(judge, txt, true_v, question, b["depth"], b["is_false"])
                        dk = min(b["depth"], 5)
                        buck[arm][dk]["n"] += 1
                        for k in ("olr", "pf", "fwr"):
                            buck[arm][dk][k] += sc[k]
            ci, co = PRICE.get(model, (0.0, 0.0))
            cost = agent_c.input_tokens / 1e6 * ci + agent_c.output_tokens / 1e6 * co
            mres = {}
            for arm in ("naive", "disciplined"):
                tot = sum(v["n"] for v in buck[arm].values()) or 1
                mres[arm] = {
                    "PF": round(sum(v["pf"] for v in buck[arm].values()) / tot, 3),
                    "OLR": round(sum(v["olr"] for v in buck[arm].values()) / tot, 3),
                    "FWR": round(sum(v["fwr"] for v in buck[arm].values()) / tot, 3),
                    "OLR_by_depth": {str(d): round(buck[arm][d]["olr"] / buck[arm][d]["n"], 3)
                                     for d in sorted(buck[arm]) if buck[arm][d]["n"]},
                    "n_by_depth": {str(d): buck[arm][d]["n"] for d in sorted(buck[arm])},
                }
            results["models"][model] = {**mres, "cost": round(cost, 3)}
            print(f"[done] {model:22s} OLR naive={mres['naive']['OLR']} disc={mres['disciplined']['OLR']} | "
                  f"PF naive={mres['naive']['PF']} disc={mres['disciplined']['PF']} | ${round(cost,3)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {model}: {str(exc)[:150]}", flush=True)
            results["models"][model] = {"error": str(exc)[:200]}

    # ---- formal curves ----
    depths = sorted({int(d) for m in results["models"].values() if "naive" in m
                     for d in m["naive"]["OLR_by_depth"]})
    print("\n================ OLR DECAY CURVE (by hop-depth) ================")
    print(f"{'model':22s}{'arm':12s}" + "".join(f"d{d:>5}" for d in depths))
    for model in MODELS:
        m = results["models"].get(model, {})
        if "error" in m:
            continue
        for arm in ("naive", "disciplined"):
            row = "".join(f"{m[arm]['OLR_by_depth'].get(str(d), float('nan')):>6.2f}" for d in depths)
            print(f"{model:22s}{arm:12s}{row}")
    print("\n================ PF / OLR summary (naive vs disciplined) ================")
    print(f"{'model':22s}{'PF_naive':>10}{'PF_disc':>9}{'OLR_naive':>11}{'OLR_disc':>10}")
    for model in MODELS:
        m = results["models"].get(model, {})
        if "error" in m:
            print(f"{model:22s}  ERROR"); continue
        print(f"{model:22s}{m['naive']['PF']:>10}{m['disciplined']['PF']:>9}{m['naive']['OLR']:>11}{m['disciplined']['OLR']:>10}")
    print(f"\nstructural BCR (firm false belief): naive={results['BCR']['naive']} disciplined={results['BCR']['disciplined']}")

    path = os.environ.get("OUT", os.path.join(_ROOT, "outputs", "society", "society_sweep.json"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(results, open(path, "w"), ensure_ascii=False, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
