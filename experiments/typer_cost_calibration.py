"""typer_cost_calibration.py — measured token accounting for the type-once-carry protocol.

Reviewer ask: the typer reads every retrieved page, so what does the protocol actually save over
just letting the frontier model answer? This script MEASURES (API-reported usage, not estimates)
per-call input/output tokens for the three legs on K balanced canonical scenarios:

  A frontier_direct : typer-model answers straight from the pages (the do-nothing baseline)
  B typer_typing    : typer-model reads pages once, emits briefing + structured provenance field
  C carrier_consume : cheap carrier answers from the typed briefing + field (per downstream use)

Per-query cost of the protocol at k downstream consumptions per typed bundle = B/k + C, vs A per
query direct. The companion analyzer (typer_cost_report.py) joins these means with the exact usage
recorded by the full gpt-5-mini-typer run and a price table.

Env: TYPER(claude-opus-4-8) CARRIER(gpt-4o-mini) K(10) SEED(20260703)
SCENARIOS(outputs/society/citation_scenarios_100_v2.jsonl).
Output: outputs/society/typer_cost_calibration.json
"""
import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from society_reflective import client_for
from research_citation_multi import load_scenarios, ask
from provenance_pipeline import pick_scenarios, make_briefing_carry, answer_from_briefing

TYPER = os.environ.get("TYPER", "claude-opus-4-8")
CARRIER = os.environ.get("CARRIER", "gpt-4o-mini")
K = int(os.environ.get("K", "10"))
SEED = int(os.environ.get("SEED", "20260703"))


def measured(client, fn):
    i0, o0 = client.input_tokens, client.output_tokens
    result = fn()
    return result, {"in": client.input_tokens - i0, "out": client.output_tokens - o0}


def stats(deltas):
    ins = [d["in"] for d in deltas]
    outs = [d["out"] for d in deltas]
    return {"n": len(deltas), "in_mean": round(sum(ins) / len(ins), 1), "in_min": min(ins), "in_max": max(ins),
            "out_mean": round(sum(outs) / len(outs), 1), "out_min": min(outs), "out_max": max(outs),
            "per_call": deltas}


def main():
    scenarios = pick_scenarios(load_scenarios(), K)
    rng = random.Random(SEED)
    tclient = client_for(TYPER)
    cclient = client_for(CARRIER)
    legs = {"frontier_direct": [], "typer_typing": [], "carrier_consume": []}
    print(f"cost calibration: typer={TYPER} carrier={CARRIER} K={len(scenarios)}", flush=True)
    for sc in scenarios:
        _, d = measured(tclient, lambda: ask(tclient, sc, "trace", rng))
        legs["frontier_direct"].append(d)
        (brief, prov), d = measured(tclient, lambda: make_briefing_carry(tclient, sc, rng))
        legs["typer_typing"].append(d)
        _, d = measured(cclient, lambda: answer_from_briefing(cclient, sc, brief, prov))
        legs["carrier_consume"].append(d)
        print(f"  {sc['id']}: direct {legs['frontier_direct'][-1]} typing {legs['typer_typing'][-1]} "
              f"consume {legs['carrier_consume'][-1]}", flush=True)

    out = {"config": {"typer": TYPER, "carrier": CARRIER, "k": len(scenarios), "seed": SEED,
                      "scenarios": os.environ.get("SCENARIOS", "default")},
           "legs": {name: stats(deltas) for name, deltas in legs.items()}}
    path = os.path.join(_ROOT, "outputs", "society", "typer_cost_calibration.json")
    json.dump(out, open(path, "w"), indent=2)
    for name in legs:
        s = out["legs"][name]
        print(f"{name:16s} in {s['in_mean']:>7.1f} [{s['in_min']}-{s['in_max']}]   "
              f"out {s['out_mean']:>6.1f} [{s['out_min']}-{s['out_max']}]")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
