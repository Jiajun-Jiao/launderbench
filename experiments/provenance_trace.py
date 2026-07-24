"""WHY do all frontier models collapse on deep provenance? Mechanism diagnostic.

The very-hard breakdown shows source-type accuracy is ~1.0 for direct/testimony but ~0.06 for HEARSAY,
i.e. the collapse is specifically a hearsay→testimony confusion at hop-depth >=2. Hypothesis: models
classify provenance from the LOCAL/most-recent relation ("someone told me" -> testimony) and do not
RECURSE down the chain to check whether that someone was themselves a witness; so multi-hop hearsay is
mislabeled as first-hand testimony. value survives because it is COPIED (no reasoning); depth dies
because it requires RELATIONAL COMPOSITION.

We test this on clean, controlled linear chains A0(witness)->A1->...->AD with gold:
  A0 direct_observation, A1 testimony (told by the witness), A2.. hearsay (told by a non-witness).
Two conditions isolate reading vs computation:
  prose   -- the chain is described in shuffled prose; the model must reconstruct who-told-whom (the .00
             setting in miniature);
  edges   -- the model is GIVEN the explicit chain (A0 saw it; A0->A1->...->AD); only classification left.
If edges >> prose, the failure is RELATIONAL READING (reconstructing the propagation graph from text),
not depth computation -> you must CARRY the graph, not ask the model to recover it. We also log the
confusion: of gold-hearsay agents, how many are called 'testimony' (the predicted collapse). Env:
DEPTHS(1,2,3,4,5,6) NCHAINS(4) MODEL(gpt-4.1) [o3 needs OPENAI_API_KEY=$OPENAI_API_KEY_2].
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

DEPTHS = [int(d) for d in os.environ.get("DEPTHS", "1,2,3,4,5,6").split(",") if d.strip()]
NCHAINS = int(os.environ.get("NCHAINS", "4"))
CONDS = [c.strip() for c in os.environ.get("CONDS", "prose,edges").split(",") if c.strip()]
SEED = int(os.environ.get("SEED", "20260629"))
TYPES = ["direct_observation", "testimony", "hearsay"]


def gold_type(depth):
    return "direct_observation" if depth == 0 else ("testimony" if depth == 1 else "hearsay")


def make_chain(rng):
    names = rng.sample(ss.NAMES, max(DEPTHS) + 1)
    return names  # names[k] is the agent at depth k


def prose_lines(chain, depth):
    """Shuffled sentences describing the chain up to `depth`. Only A0 is said to have witnessed it."""
    lines = [f"{chain[0]} saw with their own eyes that Kira took the silver apple from the market stall."]
    for k in range(1, depth + 1):
        lines.append(f"{chain[k]} came to know about the stolen apple after {chain[k-1]} passed the news on to them.")
    return lines


def edges_text(chain, depth):
    seq = " -> ".join(chain[:depth + 1])
    return (f"{chain[0]} is the only eyewitness (saw it directly). The news then passed person-to-person "
            f"along this chain, each telling the next: {seq}.")


def ask(client, cond, chain, depth):
    agents = chain[:depth + 1]
    body = ("\n".join(f"- {l}" for l in random.Random(hash((cond, depth)) & 0xffff).sample(prose_lines(chain, depth), depth + 1))
            if cond == "prose" else edges_text(chain, depth))
    dev = ("You classify how each person came to know that Kira took the silver apple. source_type is one "
           "of: direct_observation = they personally witnessed it; testimony = they were told by the person "
           "who personally witnessed it (a first-hand source); hearsay = the news reached them through "
           "someone who did NOT personally witness it (i.e. through one or more intermediaries). "
           "Return JSON {\"classifications\":[{\"agent\":\"<name>\",\"source_type\":\"<type>\"}]}.")
    user = f"Information (unordered):\n{body}\n\nClassify each of these people: {', '.join(agents)}."
    sch = {"type": "object", "additionalProperties": False, "properties": {"classifications": {"type": "array",
           "items": {"type": "object", "additionalProperties": False, "properties": {
               "agent": {"type": "string"}, "source_type": {"type": "string", "enum": TYPES}},
               "required": ["agent", "source_type"]}}}, "required": ["classifications"]}
    try:
        out = client.complete_json([{"role": "developer", "content": dev}, {"role": "user", "content": user}], response_schema=sch)
        return {str(c.get("agent", "")).strip(): c.get("source_type") for c in out.get("classifications", [])}
    except Exception as exc:  # noqa: BLE001
        print("  [error]", exc, flush=True)
        return {}


def main():
    model = os.environ.get("MODEL", "gpt-4.1")
    client = client_for(model)
    if hasattr(client, "temperature"):
        client.temperature = 0.0
    print(f"provenance trace: model={model} depths={DEPTHS} nchains={NCHAINS} conds={CONDS}", flush=True)
    rng = random.Random(SEED)
    chains = [make_chain(rng) for _ in range(NCHAINS)]

    # acc[cond][depth] = (correct, n); confusion[cond] counts gold-hearsay predicted as each type
    acc = {c: defaultdict(lambda: [0, 0]) for c in CONDS}
    confusion = {c: defaultdict(int) for c in CONDS}
    for cond in CONDS:
        for ci, chain in enumerate(chains):
            for depth in DEPTHS:
                pred = ask(client, cond, chain, depth)
                for k in range(depth + 1):
                    g = gold_type(k)
                    p = pred.get(chain[k])
                    acc[cond][k][0] += int(p == g)
                    acc[cond][k][1] += 1
                    if g == "hearsay":
                        confusion[cond][p or "missing"] += 1
            print(f"  {cond} chain {ci} done", flush=True)

    alldepths = sorted({k for c in acc for k in acc[c]})
    print("\n=== source_type accuracy by hop-depth (gold: d0=direct, d1=testimony, d>=2=hearsay) ===")
    print(f"{'cond':8s}" + "".join(f"d{d:>5}" for d in alldepths) + "   overall")
    out = {}
    for cond in CONDS:
        tot_c = sum(acc[cond][k][0] for k in acc[cond]); tot_n = sum(acc[cond][k][1] for k in acc[cond]) or 1
        by = {str(d): (round(acc[cond][d][0] / acc[cond][d][1], 2) if acc[cond][d][1] else None) for d in alldepths}
        out[cond] = {"by_depth": by, "overall": round(tot_c / tot_n, 3),
                     "hearsay_confusion": dict(confusion[cond])}
        row = "".join(f"{(by[str(d)] if by[str(d)] is not None else float('nan')):>6.2f}" for d in alldepths)
        print(f"{cond:8s}{row}   {out[cond]['overall']:.2f}")
    print("\n=== of gold-HEARSAY agents (depth>=2), what did the model call them? ===")
    for cond in CONDS:
        tot = sum(confusion[cond].values()) or 1
        frac = {k: round(v / tot, 2) for k, v in sorted(confusion[cond].items(), key=lambda x: -x[1])}
        print(f"  {cond:8s} {frac}")
    print("\nVERDICT: edges>>prose => failure is RELATIONAL READING (reconstructing who-told-whom from text),")
    print("not depth computation. hearsay->testimony confusion => models classify by the LOCAL relation and")
    print("do not recurse down the chain. Either way: provenance must be CARRIED, not recovered from prose.")
    path = os.path.join(_ROOT, "outputs", "society", f"provenance_trace_{model.replace('/', '_').replace(':', '_')}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"config": {"model": model, "depths": DEPTHS, "nchains": NCHAINS}, "results": out}, open(path, "w"), indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
