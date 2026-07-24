"""What actually breaks deep-provenance tracing: DEPTH, or COMPETING CHAINS?

Clean single chains are traced perfectly by claude-opus/o3 (provenance_trace.py: 1.00 to depth 6). Yet on
the very-hard worlds the same models collapse to ~0 on hearsay. The vhard worlds differ in one big way:
MULTIPLE conflicting chains spreading different (true vs false) values through ~13 interleaved agents. We
isolate that variable: K independent CLEAN chains, each witness->...->depth-D, each carrying a DIFFERENT
value, their sentences interleaved/shuffled into one passage; the model classifies every agent's
source_type (d0 direct, d1 testimony, d>=2 hearsay -- within its own chain). All gold is unambiguous (every
chain has a real eyewitness; no inference origins). If accuracy is ~1.0 at K=1 and collapses as K grows,
the binding difficulty is tracking COMPETING provenance chains, not depth -- models lose which witness
grounds which agent once narratives compete. Env: K(1,2,3) DEPTH(5) REPS(4) MODEL(gpt-4.1).
"""
import json
import os
import random
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import society_sim as ss
from society_reflective import client_for

KS = [int(k) for k in os.environ.get("K", "1,2,3").split(",") if k.strip()]
DEPTH = int(os.environ.get("DEPTH", "5"))
REPS = int(os.environ.get("REPS", "4"))
SEED = int(os.environ.get("SEED", "20260629"))
TYPES = ["direct_observation", "testimony", "hearsay"]
VALUES = ["Kira", "Tomas", "Doran", "Selka", "Bex"]  # distinct culprit per chain
NAMEPOOL = ["Mara", "Iris", "Joren", "Pol", "Wen", "Edda", "Nico", "Lysa", "Sira", "Rafe", "Dax",
            "Yara", "Cole", "Fenn", "Bran", "Garet", "Helka", "Milo", "Rufus", "Ari", "Nessa", "Orla",
            "Pell", "Quill", "Tova", "Ulla", "Vix", "Wynn", "Zane", "Asha", "Bode", "Cyra", "Dorn",
            "Esk", "Frey", "Gale", "Hale", "Imre", "Juno", "Kael"]  # disjoint from VALUES (culprits)


def gold_type(d):
    return "direct_observation" if d == 0 else ("testimony" if d == 1 else "hearsay")


def build(k, rng):
    """k chains, each DEPTH long, distinct agents + distinct value. Returns (agents, gold, lines)."""
    names = rng.sample(NAMEPOOL, k * (DEPTH + 1))
    gold, lines, agents = {}, [], []
    for ci in range(k):
        chain = names[ci * (DEPTH + 1):(ci + 1) * (DEPTH + 1)]
        val = VALUES[ci % len(VALUES)]
        agents += chain
        for d, nm in enumerate(chain):
            gold[nm] = gold_type(d)
        if os.environ.get("NATURALISTIC") == "1":
            wt = rng.choice([
                f"{chain[0]}, who was right there at the stall, watched {val} snatch the silver apple.",
                f"It was {chain[0]} who actually witnessed {val} take the silver apple, plain as day.",
                f"{chain[0]} happened to see {val} slip the silver apple away during the bustle."])
            lines.append(wt)
            verbs = ["confided to", "relayed it to", "whispered to", "passed word to", "mentioned it to",
                     "let slip to", "told", "shared the story with"]
            for d in range(1, DEPTH + 1):
                v = rng.choice(verbs)
                lines.append(rng.choice([
                    f"{chain[d-1]} {v} {chain[d]} that {val} was the one who took it.",
                    f"{chain[d]} got wind of it when {chain[d-1]} {v} them.",
                    f"Word reached {chain[d]} that {val} did it, by way of {chain[d-1]}."]))
        else:
            lines.append(f"{chain[0]} saw with their own eyes that {val} took the silver apple from the market stall.")
            for d in range(1, DEPTH + 1):
                lines.append(f"{chain[d]} came to know that {val} was the culprit after {chain[d-1]} passed the news on to them.")
    rng.shuffle(lines)
    return agents, gold, lines


def ask(client, agents, lines):
    dev = ("You classify how each person came to know who took the silver apple. source_type is one of: "
           "direct_observation = they personally witnessed it; testimony = they were told by the person who "
           "personally witnessed it (a first-hand source); hearsay = the news reached them through someone "
           "who did NOT personally witness it. Note there may be SEVERAL separate accounts naming different "
           "culprits; judge each person by their own chain back to whoever actually saw something. "
           "Return JSON {\"classifications\":[{\"agent\":\"<name>\",\"source_type\":\"<type>\"}]}.")
    user = "Information (unordered):\n" + "\n".join(f"- {l}" for l in lines) + f"\n\nClassify each: {', '.join(agents)}."
    sch = {"type": "object", "additionalProperties": False, "properties": {"classifications": {"type": "array",
           "items": {"type": "object", "additionalProperties": False, "properties": {
               "agent": {"type": "string"}, "source_type": {"type": "string", "enum": TYPES}},
               "required": ["agent", "source_type"]}}}, "required": ["classifications"]}
    try:
        out = client.complete_json([{"role": "developer", "content": dev}, {"role": "user", "content": user}], response_schema=sch)
        return {str(c.get("agent", "")).strip(): c.get("source_type") for c in out.get("classifications", [])}
    except Exception as exc:  # noqa: BLE001
        print("  [err]", exc, flush=True); return {}


def main():
    model = os.environ.get("MODEL", "gpt-4.1")
    client = client_for(model)
    if hasattr(client, "temperature"):
        client.temperature = 0.0
    print(f"provenance complexity: model={model} K={KS} depth={DEPTH} reps={REPS}", flush=True)
    rng = random.Random(SEED)

    # acc[k][depth] = [correct, n]; hconf[k] = confusion on gold-hearsay
    acc = {k: defaultdict(lambda: [0, 0]) for k in KS}
    hconf = {k: defaultdict(int) for k in KS}
    for k in KS:
        for r in range(REPS):
            agents, gold, lines = build(k, rng)
            pred = ask(client, agents, lines)
            # recompute per-agent depth from gold-type is ambiguous; track by gold type bucket instead
            # depth recovery: within each chain, position. We stored gold as type; bucket by type-as-depth proxy.
            for nm in agents:
                g = gold[nm]; p = pred.get(nm)
                acc[k][g][0] += int(p == g); acc[k][g][1] += 1
                if g == "hearsay":
                    hconf[k][p or "missing"] += 1
        print(f"  K={k} done", flush=True)

    print("\n=== source_type accuracy by gold type, as #competing chains grows ===")
    print(f"{'K':>3}  {'direct':>8}{'testimony':>10}{'hearsay':>9}   overall   hearsay->testimony")
    out = {}
    for k in KS:
        cells = {}
        for t in TYPES:
            c, n = acc[k][t]
            cells[t] = round(c / n, 2) if n else None
        tot_c = sum(acc[k][t][0] for t in TYPES); tot_n = sum(acc[k][t][1] for t in TYPES) or 1
        h_tot = sum(hconf[k].values()) or 1
        h2t = round(hconf[k].get("testimony", 0) / h_tot, 2)
        out[str(k)] = {"by_type": cells, "overall": round(tot_c / tot_n, 3), "hearsay_to_testimony": h2t,
                       "hearsay_confusion": dict(hconf[k])}
        d = lambda v: f"{v:.2f}" if v is not None else " -- "  # noqa: E731
        print(f"{k:>3}  {d(cells['direct_observation']):>8}{d(cells['testimony']):>10}{d(cells['hearsay']):>9}   "
              f"{out[str(k)]['overall']:.2f}      {h2t:.2f}")

    print("\nVERDICT: if hearsay accuracy falls (and hearsay->testimony rises) as K grows while a single")
    print("chain was ~1.0, the binding difficulty is tracking COMPETING provenance chains, not depth.")
    path = os.path.join(_ROOT, "outputs", "society", f"provenance_complexity_{model.replace('/', '_').replace(':', '_')}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"config": {"model": model, "K": KS, "depth": DEPTH, "reps": REPS}, "results": out}, open(path, "w"), indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
