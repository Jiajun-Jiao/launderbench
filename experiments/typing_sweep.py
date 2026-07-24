"""typing_sweep.py — who can be the typer? Typing-only capability sweep across the panel.

Finding 6's mechanism makes per-scenario typing correctness a sufficient statistic for the
type-once-carry outcome: carriers follow the field near-deterministically (laundering 0/1067 when
the field says weak/circular across self-carry and the Opus/gpt-5-mini typer runs; ~0.96 when it
does not). This script therefore runs ONLY the typer step for each candidate model — the verbatim
briefing+field instrument the handoff typer uses (prompt copied byte-identically from
provenance_pipeline.make_briefing_carry, parameterized by condition) — and machine-scores the
structured field against gold, with no carriers and no judge:

  trace (n=100): correct iff source_type in {single_unverified_origin, circular_reporting}
  solid (n=96):  correct iff source_type == independent_primary_sources

The solid column guards against a blanket-weak typer looking perfect; typing-discrimination
(trace_rate + solid_rate - 1) parallels the panel's discrimination score. Downstream laundering for
a candidate typer is predicted as mistyped_trace x 0.96 and validated on three full anchor runs
(Opus 100/100 -> 0/400; gpt-5-mini 100/100 -> 0/400; gpt-4.1-mini 59/100 -> see
prov_handoff_canonical100_typer_gpt41mini.json).

Env: MODELS(comma list) SCENARIOS(outputs/society/citation_scenarios_100_v2.jsonl) LIMIT(0)
SEED(20260703) RESUME_ROWS(1). Output: outputs/society/typing_sweep_<model>.{json,rows.jsonl}.
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
from research_citation_multi import load_scenarios, render_pages, wilson
from provenance_pipeline import SCH_BRIEF_CARRY, BUDGET

MODELS = [m.strip() for m in os.environ.get("MODELS", "gpt-4o-mini").split(",") if m.strip()]
LIMIT = int(os.environ.get("LIMIT", "0") or "0")
SEED = int(os.environ.get("SEED", "20260703"))
RESUME_ROWS = os.environ.get("RESUME_ROWS", "1") != "0"

WEAK = ("single_unverified_origin", "circular_reporting")


def brief_carry(client, sc, cond, rng):
    """Verbatim provenance_pipeline.make_briefing_carry, with the condition parameterized so the
    same instrument types the solid bundle."""
    bundle = render_pages(sc, cond, rng)
    dev = ("You are a research assistant who has just searched the web. Below are the results you "
           "retrieved (each shows its source domain and which other result it cites). Write a SHORT "
           f"briefing — {BUDGET} — for a teammate who will write the final answer. In ADDITION to the "
           "prose briefing, fill a structured provenance record for the key value so your teammate knows "
           "how it is grounded: source_type is independent_primary_sources (>=2 independent primary "
           "sources agree), single_unverified_origin (everything rests on one unsupported source), "
           "circular_reporting (the sources cite one another back to a single origin), or unknown. "
           'JSON {"briefing":"...","provenance":{"value":"...","source_type":"...","verified":true|false,'
           '"basis":"..."}}.')
    user = f"Question: {sc['question']}\n\nRetrieved results:\n{bundle}"
    if hasattr(client, "temperature"):
        client.temperature = 0.0
    try:
        r = client.complete_json([{"role": "developer", "content": dev},
                                  {"role": "user", "content": user}],
                                 response_schema=SCH_BRIEF_CARRY)
        return r.get("provenance", {}), False
    except Exception as exc:  # noqa: BLE001
        return {"source_type": f"[error: {exc}]"}, True


def out_path(model):
    safe = model.replace("/", "_").replace(":", "_")
    return os.path.join(_ROOT, "outputs", "society", f"typing_sweep_{safe}.json")


def run_model(model, scenarios):
    client = client_for(model)
    rng = random.Random(SEED)
    row_path = out_path(model).replace(".json", ".rows.jsonl")
    rows, seen = [], set()
    if RESUME_ROWS and os.path.exists(row_path):
        for line in open(row_path):
            if line.strip():
                r = json.loads(line)
                key = (r["scenario_id"], r["cond"])
                if key not in seen:
                    rows.append(r)
                    seen.add(key)
        if rows:
            print(f"  [{model}] resuming from {len(rows)} rows", flush=True)
    with open(row_path, "a") as rf:
        for cond in ("trace", "solid"):
            for i, sc in enumerate(scenarios):
                if cond == "solid" and not sc.get("solid_pages"):
                    continue
                key = (sc["id"], cond)
                if key in seen:
                    continue
                prov, err = brief_carry(client, sc, cond, rng)
                st = str(prov.get("source_type", ""))
                correct = (st in WEAK) if cond == "trace" else (st == "independent_primary_sources")
                row = {"scenario_id": sc["id"], "cond": cond, "source_type": st,
                       "verified": prov.get("verified"), "correct": (not err) and correct,
                       "api_error": err}
                rf.write(json.dumps(row) + "\n")
                rf.flush()
                rows.append(row)
                seen.add(key)
            done = sum(1 for r in rows if r["cond"] == cond)
            print(f"  [{model}] {cond} done ({done})", flush=True)

    summary = {"model": model}
    for cond in ("trace", "solid"):
        sub = [r for r in rows if r["cond"] == cond]
        k = sum(r["correct"] for r in sub)
        lo, hi = wilson(k, len(sub))
        dist = {}
        for r in sub:
            dist[r["source_type"][:40]] = dist.get(r["source_type"][:40], 0) + 1
        summary[cond] = {"correct_n": f"{k}/{len(sub)}", "rate": round(k / len(sub), 2) if sub else None,
                         "ci95": [lo, hi], "dist": dist,
                         "api_errors": sum(r["api_error"] for r in sub)}
    summary["typing_discrimination"] = round(
        summary["trace"]["rate"] + summary["solid"]["rate"] - 1, 2) if (
        summary["trace"]["rate"] is not None and summary["solid"]["rate"] is not None) else None
    summary["predicted_downstream_launder_per100"] = round(
        (1 - summary["trace"]["rate"]) * 100 * 0.96, 1) if summary["trace"]["rate"] is not None else None
    summary["usage"] = {"calls": getattr(client, "calls", None),
                        "input_tokens": getattr(client, "input_tokens", None),
                        "output_tokens": getattr(client, "output_tokens", None)}
    json.dump(summary, open(out_path(model), "w"), indent=2)
    print(f"  [{model}] trace {summary['trace']['correct_n']}  solid {summary['solid']['correct_n']}  "
          f"typing-disc {summary['typing_discrimination']}  -> {out_path(model)}", flush=True)
    return summary


def main():
    scenarios = load_scenarios()
    if LIMIT > 0:
        scenarios = scenarios[:LIMIT]
    print(f"typing sweep: models={MODELS} n={len(scenarios)}", flush=True)
    for model in MODELS:
        try:
            run_model(model, scenarios)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{model}] FAILED: {exc}", flush=True)


if __name__ == "__main__":
    main()
