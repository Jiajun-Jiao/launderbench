"""provenance_handoff.py — the ARCHITECTURE money-shot: transfer provenance discipline from a capable
TYPER to cheap CARRIERS via structured state.

Pilot finding (provenance_pipeline.py, n=10): a cheap model launders on direct reading (0.90), worse
after compression (1.00); carrying its OWN structured provenance field cuts it to 0.30, and the residual
is ENTIRELY explained by the cheap model mis-typing the field (all residual failures had the field
mislabeled 'independent_primary_sources'). So laundering decomposes into TYPING (capability-gated) and
CARRYING (architectural). Deployment claim: type the provenance ONCE with a capable model, carry it as a
structured field, and cheap downstream consumers stop laundering — provenance rides in structured state,
not in re-inferred prose.

Per (carrier model, scenario) three arms answer the SAME question; only the handoff differs:
  self_downstream : carrier summarizes the pages (prose only), then answers from its OWN prose briefing
  self_carry      : carrier summarizes WITH its own structured field, then answers from briefing + field
  typed_handoff   : carrier answers from a STRONG TYPER's (default Opus) briefing + field  <- the fix

Judge = established_fact => laundered. Prediction: self_downstream high, self_carry mid (~typing errors),
typed_handoff ~0. We also log, per scenario, the TYPER field vs each carrier's own field, to show the
residual is typing.

Env: TYPER(claude-opus-4-8) CARRIERS(gpt-4o-mini,gpt-4.1-mini,gpt-4o,gpt-4.1) LIMIT(30)
COND(trace|solid) JUDGE_MODEL(gpt-4.1-mini) SEED(20260630) OUT(optional). Frontier/OpenAI: run with
OPENAI_API_KEY=$OPENAI_API_KEY_2.
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

from society_reflective import client_for
from research_citation_multi import load_scenarios, judge, wilson
from provenance_pipeline import pick_scenarios, make_briefing_plain, make_briefing_carry, answer_from_briefing

TYPER = os.environ.get("TYPER", "claude-opus-4-8")
CARRIERS = [m.strip() for m in os.environ.get("CARRIERS", "gpt-4o-mini,gpt-4.1-mini,gpt-4o,gpt-4.1").split(",") if m.strip()]
LIMIT = int(os.environ.get("LIMIT", "30"))
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4.1-mini")
SEED = int(os.environ.get("SEED", "20260630"))
COND = os.environ.get("COND", "trace").strip().lower()
if COND not in ("trace", "solid"):
    raise ValueError("COND must be trace or solid")
ALL_ARMS = ("self_downstream", "self_carry", "typed_handoff")
# ARMS env: comma-subset of the three arms (default all). E.g. ARMS=typed_handoff re-runs only the
# typed wing under a different TYPER without re-buying the self arms.
ARMS = tuple(a.strip() for a in os.environ.get("ARMS", ",".join(ALL_ARMS)).split(",") if a.strip())
assert all(a in ALL_ARMS for a in ARMS), f"ARMS must be a subset of {ALL_ARMS}"


def _usage(client):
    return {"calls": getattr(client, "calls", None), "input_tokens": getattr(client, "input_tokens", None),
            "output_tokens": getattr(client, "output_tokens", None)}


def output_path():
    explicit = os.environ.get("OUT", "").strip()
    if explicit:
        return explicit if os.path.isabs(explicit) else os.path.join(_ROOT, explicit)
    tag = os.environ.get("SCENARIO_TAG", "").strip()
    if not tag:
        sc_path = os.environ.get("SCENARIOS", "")
        if sc_path:
            base = os.path.basename(sc_path)
            canonical = {"citation_scenarios.jsonl", "citation_scenarios_v2.jsonl"}
            if base not in canonical:
                tag = os.path.splitext(base)[0].replace("citation_scenarios_", "")
                if tag.endswith("_v2"):
                    tag = tag[:-3]
    cond_suffix = "" if COND == "trace" else f"_{COND}"
    name = "prov_handoff" + (f"_{tag}" if tag else "") + cond_suffix + ".json"
    return os.path.join(_ROOT, "outputs", "society", name)


def field_is_correct(prov, cond):
    source_type = prov.get("source_type")
    if cond == "solid":
        return source_type == "independent_primary_sources"
    return source_type in ("single_unverified_origin", "circular_reporting")


def build_typed_handoffs(scenarios, rng):
    """Strong typer produces (briefing, provenance) once per scenario."""
    tclient = client_for(TYPER)
    handoffs = {}
    correct = 0
    for sc in scenarios:
        brief, prov = make_briefing_carry(tclient, sc, rng, COND)
        handoffs[sc["id"]] = {"briefing": brief, "provenance": prov}
        if field_is_correct(prov, COND):
            correct += 1
    print(f"typer={TYPER}: typed {len(scenarios)} scenarios; "
          f"field matches {COND} gold in {correct}/{len(scenarios)}", flush=True)
    return handoffs, correct, _usage(tclient)


def run_carrier(model, scenarios, handoffs, judge_c, rng):
    client = client_for(model)
    agg = {arm: defaultdict(int) for arm in ARMS}
    own_field_correct = 0
    rows = []
    for sc in scenarios:
        v = {}
        answers = {}
        carrier_briefings = {}
        own_prov = {}
        if "self_downstream" in ARMS:
            brief_plain = make_briefing_plain(client, sc, rng, COND)
            carrier_briefings["self_downstream"] = brief_plain
            answers["self_downstream"] = answer_from_briefing(client, sc, brief_plain)
            v["self_downstream"] = judge(judge_c, sc, answers["self_downstream"])
        if "self_carry" in ARMS:
            brief_carry, own_prov = make_briefing_carry(client, sc, rng, COND)
            carrier_briefings["self_carry"] = brief_carry
            answers["self_carry"] = answer_from_briefing(client, sc, brief_carry, own_prov)
            v["self_carry"] = judge(judge_c, sc, answers["self_carry"])
        if "typed_handoff" in ARMS:
            ho = handoffs[sc["id"]]
            answers["typed_handoff"] = answer_from_briefing(
                client, sc, ho["briefing"], ho["provenance"]
            )
            v["typed_handoff"] = judge(judge_c, sc, answers["typed_handoff"])
        for arm in ARMS:
            agg[arm][v[arm]] += 1
        own_ok = field_is_correct(own_prov, COND)
        own_field_correct += int(own_ok)
        row = {
            "id": sc["id"],
            "topology": sc["topology"],
            "value": sc["value"],
            "condition": COND,
            "verdicts": v,
            # Preserve the complete generated artifacts so every judge verdict can be re-audited
            # without another paid carrier call.
            "answers": answers,
            "carrier_briefings": carrier_briefings,
        }
        if "self_carry" in ARMS:
            row.update({"own_field": own_prov.get("source_type"), "own_field_ok": own_ok})
        rows.append(row)
        print(f"  [{model}] {sc['id']} {sc['topology']:18s} "
              + " ".join(f"{arm.split('_')[-1]}={v[arm][:5]:5s}" for arm in ARMS), flush=True)

    n = len(scenarios)
    summary = {}
    own_note = (f"; own field matches gold {own_field_correct}/{n}"
                if "self_carry" in ARMS else "")
    print(f"\n=== carrier={model}: ESTABLISHED rate, condition={COND}, n={n}{own_note} ===")
    print(f"{'arm':16s}{'ESTAB.':>9}{'  95% CI':>14}{'hedged':>8}{'traced':>8}")
    for arm in ARMS:
        k = agg[arm]["established_fact"]
        lo, hi = wilson(k, n)
        summary[arm] = {
            "established_rate": round(k / n, 2),
            "established_n": f"{k}/{n}",
            # Backward-compatible aliases used by the trace-only table scripts.
            "launder": round(k / n, 2),
            "launder_n": f"{k}/{n}",
            "ci95": [lo, hi],
            "dist": dict(agg[arm]),
        }
        print(f"{arm:16s}{k / n:>9.2f}{f'[{lo},{hi}]':>14}{agg[arm]['hedged'] / n:>8.2f}"
              f"{agg[arm]['traced_weak_origin'] / n:>8.2f}")
    print("  >> " + " -> ".join(f"{arm} {summary[arm]['launder']:.2f}" for arm in ARMS), flush=True)
    out = {"summary": summary, "rows": rows, "usage": _usage(client)}
    if "self_carry" in ARMS:
        out["own_field_correct"] = f"{own_field_correct}/{n}"
    return out


def main():
    available = load_scenarios()
    if COND == "solid":
        available = [sc for sc in available if sc.get("solid_pages")]
        if not available:
            raise SystemExit("COND=solid requested but the scenario file has no solid_pages")
    scenarios = pick_scenarios(available, LIMIT)
    judge_c = client_for(JUDGE_MODEL)
    rng = random.Random(SEED)
    print(f"provenance HANDOFF: condition={COND} typer={TYPER} carriers={CARRIERS} "
          f"arms={ARMS} n={len(scenarios)}", flush=True)
    handoffs, typer_correct, typer_usage = ({}, None, None)
    if "typed_handoff" in ARMS:
        handoffs, typer_correct, typer_usage = build_typed_handoffs(scenarios, rng)
    out = {"output_schema_version": 2,
           "config": {"typer": TYPER, "carriers": CARRIERS, "arms": list(ARMS), "condition": COND,
                      "n": len(scenarios),
                      "seed": SEED, "scenarios": os.environ.get("SCENARIOS", "default")},
           "carrier_answers_saved": True,
           "typer_fields": {sid: h["provenance"] for sid, h in handoffs.items()},
           "typer_briefings": {sid: h["briefing"] for sid, h in handoffs.items()},
           "typer_field_correct": (f"{typer_correct}/{len(scenarios)}" if typer_correct is not None else None),
           "typer_usage": typer_usage, "carriers": {}}
    path = output_path()
    for model in CARRIERS:
        try:
            out["carriers"][model] = run_carrier(model, scenarios, handoffs, judge_c, rng)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{model}] FAILED: {exc}", flush=True)
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(out, open(path, "w"), indent=2)
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
