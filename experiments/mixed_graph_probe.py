"""mixed_graph_probe.py — dissociate graph reading from terminal counting (reviewer R1 / the
terminal-node concern, App E).

Three bundle conditions per scenario, all composed programmatically from already-gated material
(no new generation, no new gates):

  trace : the canonical 5-page bundle — every chain resolves to the one weak origin.
          gold independent primaries = 0 (the origin is a terminal but establishes nothing);
          terminal pages = 1.
  mixed : the trace bundle minus one in-degree-0 echo, plus ONE independent primary lifted from the
          scenario's validated solid bundle (renamed P6).
          gold = 1; terminals = 2.
  solid : the canonical solid bundle (three primary records + two citers).
          gold = 3; terminals = 3.

The probe asks a COUNTING question with machine-checkable gold (no judge): how many pages are
independent primary sources for the value, and which. A model counting terminal (no-cite) pages
answers 1/2/3 across the conditions; a model actually reading lineage and source quality answers
0/1/3. The two heuristics disagree on trace and mixed and agree only on solid, the calibration
anchor.

Env: MODELS(gpt-4o-mini) LIMIT(30) SEED(20260703)
SCENARIOS(outputs/society/citation_scenarios_100_v2.jsonl) RESUME_ROWS(1).
Output: outputs/society/mixed_graph_probe_<model>.{json,rows.jsonl}.
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
from research_citation_multi import load_scenarios, wilson
from provenance_pipeline import pick_scenarios

MODELS = [m.strip() for m in os.environ.get("MODELS", "gpt-4o-mini").split(",") if m.strip()]
LIMIT = int(os.environ.get("LIMIT", "30"))
SEED = int(os.environ.get("SEED", "20260703"))
RESUME_ROWS = os.environ.get("RESUME_ROWS", "1") != "0"

DEV = ("You are auditing retrieved search results. Below are retrieved results for a claim; each "
       "shows its source domain and which other result it cites. Count how many of these pages are "
       "INDEPENDENT PRIMARY SOURCES for the claimed value: a page that itself establishes the value "
       "first-hand (an original record, measurement, register, or eyewitness document). A page that "
       "repeats or cites another page's figure is not primary, and a page that merely asserts the "
       "value without evidence is not primary. Pages that trace back to the same underlying source "
       "are one lineage, and only that lineage's primary establisher (if any) is counted. The count "
       'can be 0. JSON {"count": 0, "pages": ["P3"], "basis": "..."} where pages lists the ids of '
       "exactly the pages you counted.")
SCH = {"type": "object", "additionalProperties": False,
       "properties": {"count": {"type": "integer"}, "pages": {"type": "array", "items": {"type": "string"}},
                      "basis": {"type": "string"}},
       "required": ["count", "pages", "basis"]}


def build_bundles(sc):
    """Return {cond: (pages, gold_count, gold_page_ids, n_terminals)} for one scenario."""
    trace = [dict(p) for p in sc["pages"]]
    solid = [dict(p) for p in sc.get("solid_pages") or []]
    out = {}
    t_terms = [p["id"] for p in trace if not p.get("cites")]
    out["trace"] = (trace, 0, [], t_terms)
    if solid:
        prim_ids = [p["id"] for p in solid if not p.get("cites")]
        out["solid"] = (solid, len(prim_ids), prim_ids, prim_ids)
        # mixed: drop one in-degree-0 non-origin echo, inject the first solid primary as P6
        cited = {p.get("cites") for p in trace}
        droppable = [p["id"] for p in trace if p["id"] != sc["origin_id"] and p["id"] not in cited]
        assert droppable, f"{sc['id']}: no in-degree-0 echo to drop"
        keep = [p for p in trace if p["id"] != droppable[0]]
        primary = dict(next(p for p in solid if not p.get("cites")))
        primary["id"] = "P6"
        mixed = keep + [primary]
        out["mixed"] = (mixed, 1, ["P6"], [p["id"] for p in mixed if not p.get("cites")])
    return out


def render(pages, rng):
    pages = list(pages)
    rng.shuffle(pages)
    return "\n".join(
        f"- [{p['id']}] {p['domain']}: \"{p['text']}\""
        + (f" (cites {p['cites']})" if p.get("cites") else " (cites nothing — original source)")
        for p in pages)


def out_path(model):
    safe = model.replace("/", "_").replace(":", "_")
    return os.path.join(_ROOT, "outputs", "society", f"mixed_graph_probe_{safe}.json")


def run_model(model, scenarios):
    client = client_for(model)
    if hasattr(client, "temperature"):
        client.temperature = 0.0
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
        for sc in scenarios:
            bundles = build_bundles(sc)
            for cond in ("trace", "mixed", "solid"):
                if cond not in bundles:
                    continue
                key = (sc["id"], cond)
                if key in seen:
                    continue
                pages, gold_n, gold_ids, term_ids = bundles[cond]
                n_term = len(term_ids)
                user = f"Claim: {sc['claim_sentence']}\n\nRetrieved results:\n{render(pages, rng)}"
                try:
                    r = client.complete_json([{"role": "developer", "content": DEV},
                                              {"role": "user", "content": user}], response_schema=SCH)
                    cnt = int(r.get("count", -1))
                    picked = [str(x).strip() for x in (r.get("pages") or [])]
                    basis = str(r.get("basis", ""))[:300]
                    err = False
                except Exception as exc:  # noqa: BLE001
                    cnt, picked, basis, err = -1, [], f"[error: {exc}]", True
                row = {"scenario_id": sc["id"], "cond": cond, "topology": sc["topology"],
                       "gold_count": gold_n, "gold_pages": gold_ids, "terminal_pages": term_ids,
                       "count": cnt, "pages": picked, "basis": basis,
                       "count_correct": (not err) and cnt == gold_n,
                       "pages_correct": (not err) and set(picked) == set(gold_ids),
                       "terminal_count_answer": (not err) and gold_n != n_term
                                                and cnt == n_term and set(picked) == set(term_ids),
                       "api_error": err}
                rf.write(json.dumps(row) + "\n")
                rf.flush()
                rows.append(row)
                seen.add(key)
            print(f"  [{model}] {sc['id']} " + " ".join(
                f"{c}={next((r['count'] for r in rows if r['scenario_id']==sc['id'] and r['cond']==c), '?')}"
                for c in ("trace", "mixed", "solid")), flush=True)

    summary = {"model": model}
    for cond in ("trace", "mixed", "solid"):
        sub = [r for r in rows if r["cond"] == cond]
        if not sub:
            continue
        k = sum(r["count_correct"] for r in sub)
        lo, hi = wilson(k, len(sub))
        dist = {}
        for r in sub:
            dist[str(r["count"])] = dist.get(str(r["count"]), 0) + 1
        summary[cond] = {"count_correct": f"{k}/{len(sub)}", "ci95": [lo, hi],
                         "pages_correct": f"{sum(r['pages_correct'] for r in sub)}/{len(sub)}",
                         "terminal_count_answers": sum(r["terminal_count_answer"] for r in sub),
                         "count_dist": dist, "api_errors": sum(r["api_error"] for r in sub)}
    summary["usage"] = {"calls": getattr(client, "calls", None),
                        "input_tokens": getattr(client, "input_tokens", None),
                        "output_tokens": getattr(client, "output_tokens", None)}
    json.dump(summary, open(out_path(model), "w"), indent=2)
    print(f"  [{model}] " + "  ".join(
        f"{c}: {summary[c]['count_correct']} (term-cnt {summary[c]['terminal_count_answers']})"
        for c in ("trace", "mixed", "solid") if c in summary), flush=True)
    return summary


def main():
    scenarios = pick_scenarios([s for s in load_scenarios() if s.get("solid_pages")], LIMIT)
    print(f"mixed-graph counting probe: models={MODELS} n={len(scenarios)} x3 conds", flush=True)
    for model in MODELS:
        try:
            run_model(model, scenarios)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{model}] FAILED: {exc}", flush=True)


if __name__ == "__main__":
    main()
