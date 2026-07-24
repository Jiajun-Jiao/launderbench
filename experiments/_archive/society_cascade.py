"""Society CASCADE: does provenance-loss PROPAGATE and AMPLIFY through the society?

society_reflective.py showed each agent strips its OWN source during reflection (a per-node bug).
This asks the genuinely SOCIETY-level question: when an agent relays its POST-REFLECTION belief to the
next agent, does the damage compound? An upstream agent that reflected a rumor into a confident bare
fact ("Kira took it") feeds the NEXT agent a source-free assertion; that agent's memory now holds a
confident claim with no provenance to hedge, so it asserts (and confabulates) too -- and a CONFIDENT
assertion is adopted more firmly downstream than a hedged "I heard maybe...". So the prediction is not
just leakage but AMPLIFIED contamination with social distance that a single-agent test cannot show.

Substrate: ss.build_graph + ss.propagate give controlled depth/teller/value/side. We process agents in
depth order; each agent's memory = the line its teller ACTUALLY EMITTED (telephone), so leakage is
inherited. Arms:
  reflect  -- neutral GA consolidation, then relay the consolidated belief (text propagates, lossy);
  careful  -- consolidation that asks to preserve how-you-know (text propagates);
  arch     -- carry a structured (value, depth) field hop-to-hop (NO text dependence); speak licensed.
Independent judge -> OLR / CONFAB (claims firsthand at depth>=1) / BCR (firmly asserts the FALSE value)
by hop-depth. HEADLINE test: is DEEP BCR much higher under `reflect` (confident relay) than `careful`
(hedged relay), and does `arch` contain both? Env: N(16) SEEDS(7,11,13) SHORTCUTS(6) ROUNDS(12)
REFLECT_ROUNDS(2) MODEL/JUDGE_MODEL(gpt-4.1-mini).
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
from society_propagation import classify, value_ok
from society_reflective import client_for, consolidate, name_of

N = int(os.environ.get("N", "16"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "7,11,13").split(",") if s.strip()]
SHORTCUTS = int(os.environ.get("SHORTCUTS", "6"))
ROUNDS = int(os.environ.get("ROUNDS", "12"))
REFLECT_ROUNDS = int(os.environ.get("REFLECT_ROUNDS", "2"))
ARMS = [a.strip() for a in os.environ.get("ARMS", "reflect,careful,arch").split(",") if a.strip()]


def seed_line(value, is_false):
    return (f"{value} took the silver apple -- I'm certain of it." if is_false
            else f"I saw it with my own eyes: {value} took the silver apple from the market stall.")


def speak(client, name, belief_text, question):
    dev = "Role-play a villager answering a traveler in ONE natural sentence, from your settled belief. JSON {\"reply\":\"...\"}."
    user = f"You are {name}. You believe: {belief_text}\nA traveler asks: \"{question}\""
    sch = {"type": "object", "additionalProperties": False, "properties": {"reply": {"type": "string"}}, "required": ["reply"]}
    client.temperature = 0.5
    try:
        return str(client.complete_json([{"role": "developer", "content": dev}, {"role": "user", "content": user}],
                                        response_schema=sch).get("reply", "")).strip()
    except Exception as exc:  # noqa: BLE001
        return f"[error: {exc}]"


def run_arm(client, judge, arm, belief, true_v, false_v, question):
    order = sorted(belief.items(), key=lambda kv: kv[1]["depth"])
    utter = {}
    buck = defaultdict(lambda: {"n": 0, "olr": 0, "confab": 0, "firm_false": 0})
    rows = []
    for i, b in order:
        d = b["depth"]
        if d == 0:
            utter[i] = seed_line(b["value"], b["is_false"])  # witness / rumor-origin start the chain
            continue
        incoming = utter.get(b["frm"], seed_line(b["value"], b["is_false"]))
        if arm == "arch":
            # architecture: depth carried structurally (no dependence on the lossy incoming text)
            emitted = ss.utterance(client, "disciplined", name_of(i), b, question, 0.0)
        else:
            mem_line = f"{name_of(b['frm'])} told you: \"{incoming}\""
            belief_text = consolidate(client, name_of(i), [mem_line], REFLECT_ROUNDS, careful=(arm == "careful"))
            emitted = speak(client, name_of(i), belief_text, question)
        utter[i] = emitted  # what this agent relays downstream (telephone)
        c = classify(judge, emitted, true_v, question)
        act = c.get("speech_act", "other")
        firsthand = bool(c.get("claims_firsthand"))
        av = c.get("asserted_value")
        leak = (act == "fact") or (firsthand and d >= 1)
        dk = min(d, 5)
        buck[dk]["n"] += 1
        buck[dk]["olr"] += int(leak)
        buck[dk]["confab"] += int(d >= 1 and firsthand)
        buck[dk]["firm_false"] += int(act == "fact" and value_ok(false_v, av))
        rows.append({"name": name_of(i), "depth": d, "is_false": b["is_false"], "act": act,
                     "firsthand": firsthand, "leak": int(leak), "asserted": av, "reply": emitted})
    return buck, rows


def main():
    true_v, false_v, question = ss.FACT
    model = os.environ.get("MODEL", "gpt-4.1-mini")
    client = client_for(model)
    judge = client_for(os.environ.get("JUDGE_MODEL", "gpt-4.1-mini"))
    print(f"society cascade: model={model} N={N} seeds={SEEDS} arms={ARMS}", flush=True)

    agg = {arm: defaultdict(lambda: {"n": 0, "olr": 0, "confab": 0, "firm_false": 0}) for arm in ARMS}
    samples = {}
    for seed in SEEDS:
        adj = ss.build_graph(N, seed, SHORTCUTS)
        belief = ss.propagate(adj, 0, N // 2, true_v, false_v, ROUNDS)
        for arm in ARMS:
            buck, rows = run_arm(client, judge, arm, belief, true_v, false_v, question)
            for d, v in buck.items():
                for k in ("n", "olr", "confab", "firm_false"):
                    agg[arm][d][k] += v[k]
            if seed == SEEDS[0]:
                samples[arm] = sorted([r for r in rows if r["depth"] >= 3], key=lambda r: -r["depth"])[:3]
        print(f"  seed {seed} done", flush=True)

    depths = sorted({d for arm in agg for d in agg[arm]})

    def rate(arm, key, dmin=0):
        num = sum(agg[arm][d][key] for d in agg[arm] if d >= dmin)
        den = sum(agg[arm][d]["n"] for d in agg[arm] if d >= dmin) or 1
        return round(num / den, 3)

    print("\n=== CASCADE: OLR / CONFAB / deep-BCR by hop-depth (relayed belief propagates downstream) ===")
    print(f"{'arm':10s}" + "".join(f"d{d:>5}" for d in depths) + "    OLR  CONFAB  BCR  BCR(d>=2)")
    out = {}
    for arm in ARMS:
        olr_by = {str(d): round(agg[arm][d]["olr"] / agg[arm][d]["n"], 2) for d in depths if agg[arm][d]["n"]}
        out[arm] = {"OLR_by_depth": olr_by, "OLR": rate(arm, "olr"), "CONFAB": rate(arm, "confab"),
                    "BCR": rate(arm, "firm_false"), "BCR_deep": rate(arm, "firm_false", 2)}
        row = "".join(f"{olr_by.get(str(d), float('nan')):>6.2f}" for d in depths)
        print(f"{arm:10s}{row}   {out[arm]['OLR']:.2f}  {out[arm]['CONFAB']:.2f}  {out[arm]['BCR']:.2f}  {out[arm]['BCR_deep']:.2f}")

    print("\n-- example deep-hop relayed utterances (seed", SEEDS[0], ") --")
    for arm in ARMS:
        print(f"  [{arm}]")
        for r in samples.get(arm, []):
            print(f"    {r['name']} (d{r['depth']}{', FALSE' if r['is_false'] else ''}) "
                  f"[{r['act']}{' LEAK' if r['leak'] else ''}]: \"{r['reply'][:115]}\"")

    print("\nHEADLINE: if reflect's deep-BCR >> careful's, confident (source-stripped) relay AMPLIFIES")
    print("false-belief contamination with social distance -- a society-level effect, not a per-node bug.")
    path = os.path.join(_ROOT, "outputs", "society", f"society_cascade_{model.replace('/', '_').replace(':', '_')}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"config": {"N": N, "seeds": SEEDS, "model": model}, "results": out, "samples": samples},
              open(path, "w"), ensure_ascii=False, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
