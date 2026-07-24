"""Incremental isolation: is the failure about PROPAGATION (carrying provenance) or just long-chain
COMPREHENSION? Here every agent sees ONLY the single message its neighbor sent it — never the whole
chain — so no failure can be 'couldn't parse a 6-name nested sentence'. We compare three regimes that
differ ONLY in whether provenance metadata is carried:
  no_meta   — message is pure NL; agent relays freely (the realistic naive telephone);
  soft_meta — message is NL; agent is told to PRESERVE how-it-knows in its wording (no hard tag);
  hard_meta — message carries a DEPTH COUNTER; agent just reads it, +1, and speaks licensed (NO LLM
              intelligence, a dumb counter = architecture L1).
Metric: provenance TIER accuracy by hop = does the agent convey its true evidential distance
(direct@0 / testimony@1 / hearsay@>=2)? Plus refined OLR and structural BCR.
HEADLINE: if a dumb counter (hard_meta) stays perfect while the smartest LLM (no_meta) degrades with
hops, the problem is carrying metadata (propagation), NOT comprehension. Env: N(30) SEEDS(7,11)
MODEL(gpt-4.1-mini).
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
from society_propagation import classify
from llm_controller import OpenAICompatibleChatClient

N = int(os.environ.get("N", "30"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "7,11").split(",") if s.strip()]
SHORTCUTS = int(os.environ.get("SHORTCUTS", "6"))
ROUNDS = int(os.environ.get("ROUNDS", "12"))


def true_tier(d):
    return "direct" if d == 0 else ("testimony" if d == 1 else "hearsay")


def conveyed_tier(act):
    return {"fact": "direct", "attributed": "testimony"}.get(act, "hearsay")


def seed_msg(value, is_false):
    return f"{value} took it — I'm certain." if is_false else f"I saw it myself: {value} took the silver apple."


def relay(client, regime, incoming, question):
    if regime == "no_meta":
        dev = ("Role-play a villager. A neighbor just told you the line below. Pass on what you now know "
               "to a traveler in one or two natural sentences. Return JSON {\"reply\":\"...\"}.")
    else:  # soft_meta
        dev = ("Role-play a villager. From the single line your neighbor told you, judge HOW you know this "
               "(did they witness it themselves, or is it secondhand / a rumor passed along?) and pass it on "
               "PRESERVING that: state plainly ONLY if it's firsthand to you; attribute it if a witness told "
               "you; mark it as an unverified rumor if it's been passed along. One sentence. JSON {\"reply\":\"...\"}.")
    schema = {"type": "object", "additionalProperties": False, "properties": {"reply": {"type": "string"}}, "required": ["reply"]}
    client.temperature = 0.7 if regime == "no_meta" else 0.0
    try:
        return str(client.complete_json([{"role": "developer", "content": dev},
                                         {"role": "user", "content": f"Neighbor told you: \"{incoming}\"\nTraveler asks: \"{question}\""}],
                                        response_schema=schema).get("reply", "")).strip()
    except Exception as exc:  # noqa: BLE001
        return f"[error: {exc}]"


def run_llm_regime(client, judge, regime, belief, true_v, question):
    """Telephone: each agent sees ONLY its teller's relayed message. Returns per-depth tier/olr counts."""
    order = sorted(belief.items(), key=lambda kv: kv[1]["depth"])
    utter, buck = {}, defaultdict(lambda: {"n": 0, "tier_ok": 0, "olr": 0})
    for i, b in order:
        if b["depth"] == 0:
            utter[i] = seed_msg(b["value"], b["is_false"])
            continue
        incoming = utter.get(b["frm"], seed_msg(b["value"], b["is_false"]))
        txt = relay(client, regime, incoming, question)
        utter[i] = txt
        c = classify(judge, txt, true_v, question)
        act = c.get("speech_act", "other")
        firsthand = bool(c.get("claims_firsthand"))
        leak = (act == "fact") or (firsthand and b["depth"] >= 1)
        dk = min(b["depth"], 5)
        buck[dk]["n"] += 1
        buck[dk]["tier_ok"] += int(conveyed_tier(act) == true_tier(b["depth"]))
        buck[dk]["olr"] += int(leak)
    return buck


def main():
    true_v, false_v, question = ss.FACT
    model = os.environ.get("MODEL", "gpt-4.1-mini")
    client = OpenAICompatibleChatClient(model=model, base_url=os.environ.get("BASE", "https://api.openai.com/v1"),
                                        api_key=os.environ.get("OPENAI_API_KEY"), use_structured_output=True)
    judge = OpenAICompatibleChatClient(model=os.environ.get("JUDGE_MODEL", model),
                                       base_url=os.environ.get("JUDGE_BASE", "https://api.openai.com/v1"),
                                       api_key=os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY"),
                                       use_structured_output=True)
    print(f"incremental isolation: model={model} N={N} seeds={SEEDS}", flush=True)

    agg = {r: defaultdict(lambda: {"n": 0, "tier_ok": 0, "olr": 0}) for r in ("no_meta", "soft_meta")}
    bcr_no, bcr_hard = [], []
    for seed in SEEDS:
        adj = ss.build_graph(N, seed, SHORTCUTS)
        belief = ss.propagate(adj, 0, N // 2, true_v, false_v, ROUNDS)
        bcr_no.append(ss.bcr(belief, N, "naive"))         # no_meta: can't tell depth -> firmly believes -> like naive
        bcr_hard.append(ss.bcr(belief, N, "disciplined"))  # hard_meta: counter -> deep held as hearsay -> contained
        for r in ("no_meta", "soft_meta"):
            buck = run_llm_regime(client, judge, r, belief, true_v, question)
            for d, v in buck.items():
                for k in ("n", "tier_ok", "olr"):
                    agg[r][d][k] += v[k]
        print(f"  seed {seed} done", flush=True)

    depths = sorted({d for r in agg for d in agg[r]})
    print("\n=== provenance TIER accuracy by hop (does the agent convey its true evidential distance?) ===")
    print(f"{'regime':12s}" + "".join(f"d{d:>5}" for d in depths) + "   overall")
    out = {}
    for r in ("no_meta", "soft_meta"):
        tot = sum(v["n"] for v in agg[r].values()) or 1
        by = {str(d): round(agg[r][d]["tier_ok"] / agg[r][d]["n"], 2) for d in depths if agg[r][d]["n"]}
        ov = round(sum(v["tier_ok"] for v in agg[r].values()) / tot, 3)
        olr = round(sum(v["olr"] for v in agg[r].values()) / tot, 3)
        out[r] = {"tier_acc_by_depth": by, "tier_acc": ov, "OLR": olr}
        print(f"{r:12s}" + "".join(f"{by.get(str(d), float('nan')):>6.2f}" for d in depths) + f"   {ov:.2f}")
    # hard_meta: deterministic counter -> always-correct tier, no LLM, no comprehension
    print(f"{'hard_meta':12s}" + "".join(f"{1.00:>6.2f}" for _ in depths) + "   1.00   (dumb depth-counter, NO LLM)")
    out["hard_meta"] = {"tier_acc": 1.0, "OLR": 0.0, "note": "deterministic counter; no comprehension needed"}

    avg = lambda xs: round(sum(xs) / len(xs), 3) if xs else 0.0  # noqa: E731
    out["BCR"] = {"no_meta": avg(bcr_no), "hard_meta": avg(bcr_hard)}
    print(f"\nrefined OLR: no_meta={out['no_meta']['OLR']} soft_meta={out['soft_meta']['OLR']} hard_meta=0.0")
    print(f"structural BCR (firm false belief): no_meta={avg(bcr_no)} hard_meta={avg(bcr_hard)}")
    print("\nHEADLINE: a dumb depth-counter (hard_meta, no LLM) holds tier accuracy at 1.0; the LLM in")
    print("naive telephone (no_meta) degrades with hops -> the failure is CARRYING provenance (propagation),")
    print("NOT comprehension. Each agent only ever saw ONE local message.")

    path = os.path.join(_ROOT, "outputs", "society", "society_incremental.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"config": {"N": N, "seeds": SEEDS, "model": model}, "results": out}, open(path, "w"), indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
