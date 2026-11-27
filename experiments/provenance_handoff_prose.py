"""provenance_handoff_prose.py -- two controls that bound the "state, not prose" reading of the
type-once-carry result (reviewer objection: prose-only vs. typed-field is confounded by the typer's
knowledge AND by the channel; and what happens when the field conflicts with persuasive prose?).

Arms, per (carrier, scenario), same question and judge as provenance_handoff.py:

  prose_warning  : the STRONG TYPER (Opus) writes a prose-ONLY briefing that is explicitly
                   instructed to state, in the prose, how the value is grounded (e.g., "all pages
                   trace to one anonymous forum guess"). The carrier answers from that prose alone.
                   -> Distinguishes "prose cannot carry provenance" (laundering stays high) from
                      "default prose omits provenance" (laundering drops like typed_handoff).
  field_vs_prose : the carrier answers from its OWN plain prose briefing (the self_downstream
                   briefing, which typically presents the value confidently) PLUS the typer's
                   structured field (which says weak/circular). The two conflict.
                   -> Measures whether the typed field survives persuasive contrary prose.
  templated      : CONTENT-MATCHED control for typed_handoff. Same stored typer briefing and same
                   stored typer field as the typed arm, but the field is rendered deterministically
                   into a fixed English sentence instead of JSON. No model writes it, so no extra
                   caveats can enter. typed_handoff vs templated therefore varies ONLY the encoding
                   of an identical decision, separating "JSON/schema encoding" from "an obligatory,
                   content-fixed decision channel". Requires TYPER_FIELDS to be a schema-v2 file
                   that also stores typer_briefings.

Reuses the stored typer fields from prov_handoff_canonical100.json (exact same records as Table 3)
and the briefing/answer/judge machinery of provenance_pipeline.py / research_citation_multi.py.

Env: TYPER(claude-opus-4-8) CARRIERS(gpt-4o-mini,gpt-4.1-mini,gpt-4o,gpt-4.1) LIMIT(100)
     COND(trace|solid) ARMS(prose_warning,field_vs_prose)
     JUDGE_MODEL(gpt-4.1-mini) SEED(20260703) SCENARIOS/SCENARIO_TAG as in the canonical100 run
     TYPER_FIELDS(outputs/society/prov_handoff_canonical100.json)
Usage: set -a; . ./.env; set +a; OPENAI_API_KEY=$OPENAI_API_KEY_2 SEED=20260703 \
       SCENARIOS=outputs/society/citation_scenarios_100_v2.jsonl SCENARIO_TAG=canonical100 \
       python3 experiments/provenance_handoff_prose.py
Output: outputs/society/prov_handoff_prose_<tag>.json (resumable per carrier)
"""
import json
import os
import random
import re
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from society_reflective import client_for
from research_citation_multi import load_scenarios, render_pages, judge, wilson
from provenance_pipeline import (pick_scenarios, make_briefing_plain, answer_from_briefing,
                                 _complete, SCH_BRIEF, SCH_ANSWER, BUDGET)
from provenance_pipeline import SOURCE_COUNT_FIELD

TYPER = os.environ.get("TYPER", "claude-opus-4-8")
CARRIERS = [m.strip() for m in os.environ.get("CARRIERS", "gpt-4o-mini,gpt-4.1-mini,gpt-4o,gpt-4.1").split(",") if m.strip()]
LIMIT = int(os.environ.get("LIMIT", "100"))
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4.1-mini")
SEED = int(os.environ.get("SEED", "20260703"))
COND = os.environ.get("COND", "trace").strip().lower()
if COND not in ("trace", "solid"):
    raise ValueError("COND must be trace or solid")
TYPER_FIELDS = os.environ.get("TYPER_FIELDS",
                              os.path.join(_ROOT, "outputs", "society", "prov_handoff_canonical100.json"))
TAG = os.environ.get("SCENARIO_TAG", "canonical100")
_explicit_out = os.environ.get("OUT", "").strip()
OUT = (_explicit_out if os.path.isabs(_explicit_out) else os.path.join(_ROOT, _explicit_out)) if _explicit_out else os.path.join(
    _ROOT, "outputs", "society", f"prov_handoff_prose_{TAG}{'' if COND == 'trace' else '_' + COND}.json"
)
ALL_ARMS = ("prose_warning", "field_vs_prose", "templated")
ARMS = tuple(a.strip() for a in os.environ.get("ARMS", ",".join(ALL_ARMS)).split(",") if a.strip())
assert all(a in ALL_ARMS for a in ARMS), f"ARMS must be a subset of {ALL_ARMS}"
EXPLICIT_RE = re.compile(
    r"single|one (?:anonymous|unverified|unsourced)|circular|cite (?:each other|one another)"
    r"|trace[sd]? (?:back )?to|not independent|no independent|unverified|unsourced|forum"
    r"|guess|speculat|independent|primary|corroborat",
    re.I,
)


def make_briefing_prose_warning(client, sc, rng):
    bundle = render_pages(sc, COND, rng)
    dev = ("You are a research assistant who has just searched the web. Below are the results you "
           "retrieved (each shows its source domain and which other result it cites). Write a SHORT "
           f"briefing — {BUDGET} — summarizing your findings for a teammate who will write the final "
           "answer to the user. Your teammate will see ONLY this prose briefing (no sources, no "
           "structured record), so state IN THE PROSE how the key value is grounded: if the results "
           "are independently corroborated, say so; if everything traces back to a single unverified "
           "origin, or the sources merely cite one another, say that plainly and characterize the "
           'origin. Keep it concise. JSON {"briefing":"..."}.')
    user = f"Question: {sc['question']}\n\nRetrieved results:\n{bundle}"
    return str(_complete(client, dev, user, SCH_BRIEF, "briefing").get("briefing", "")).strip()


# Fixed renderings of the typed field's enum. Deterministic: the same record always yields the
# same sentence, and no model is asked to write or expand it.
SOURCE_TYPE_PHRASE = {
    "independent_primary_sources": "at least two independent primary sources agree on it",
    "single_unverified_origin": "everything rests on one unsupported source",
    "circular_reporting": "the sources cite one another back to a single origin",
    "unknown": "its grounding could not be determined",
}


def render_field_template(prov):
    """The typed record, verbatim, as a fixed English sentence (same four fields, same order).

    A record that carries the optional independent-source count (PROV_SOURCE_COUNT=1 at typing
    time) gets one more fixed sentence; a four-field record renders exactly as before.
    """
    st = str(prov.get("source_type", "unknown")).strip()
    phrase = SOURCE_TYPE_PHRASE.get(st, f"its source type is recorded as {st}")
    count = prov.get(SOURCE_COUNT_FIELD)
    count_sentence = "" if count is None else f"Independent sources: {count}. "
    return (f"The key value is {prov.get('value')}. Provenance: {phrase}. "
            f"Independently verified: {'yes' if prov.get('verified') else 'no'}. "
            + count_sentence +
            f"Basis: {prov.get('basis')}")


def answer_from_briefing_templated(client, sc, briefing, provenance):
    """Mirror of answer_from_briefing's provenance branch with the record written out in prose."""
    dev = ("You are writing the final answer for the user. A research teammate investigated and gave "
           "you the briefing AND a provenance statement below (you do NOT have the original "
           "sources). Answer the user's question and state how well established the answer is, taking "
           'the provenance statement into account. JSON {"answer":"..."}.')
    user = (f"Question: {sc['question']}\n\nTeammate's briefing:\n{briefing}\n\n"
            f"Provenance statement:\n{render_field_template(provenance)}")
    return str(_complete(client, dev, user, SCH_ANSWER, "answer").get("answer", "")).strip()


def main():
    available = load_scenarios()
    if COND == "solid":
        available = [sc for sc in available if sc.get("solid_pages")]
        if not available:
            raise SystemExit("COND=solid requested but the scenario file has no solid_pages")
    scenarios = pick_scenarios(available, LIMIT)
    judge_c = client_for(JUDGE_MODEL)
    rng = random.Random(SEED)
    stored = json.load(open(TYPER_FIELDS))
    typer_fields = stored["typer_fields"]
    missing = [sc["id"] for sc in scenarios if sc["id"] not in typer_fields]
    if missing:
        raise SystemExit(f"typer fields missing for {len(missing)} scenarios, e.g. {missing[:3]} — "
                         f"run provenance_handoff.py first or fix TYPER_FIELDS")
    typer_briefings = stored.get("typer_briefings", {})
    if "templated" in ARMS:
        miss_b = [sc["id"] for sc in scenarios if sc["id"] not in typer_briefings]
        if miss_b:
            raise SystemExit(f"ARMS=templated needs the typer briefings stored alongside the fields; "
                             f"{len(miss_b)} missing in {TYPER_FIELDS} (use a schema-v2 file)")
    out = {"output_schema_version": 2,
           "config": {"typer": TYPER, "carriers": CARRIERS, "condition": COND,
                      "n": len(scenarios), "seed": SEED,
                      "typer_fields_from": TYPER_FIELDS, "arms": list(ARMS)},
           "carrier_answers_saved": True,
           "typer_prose_warnings": {}, "carriers": {}}
    if os.path.exists(OUT):
        prev = json.load(open(OUT))
        prev_config = prev.get("config", {})
        if (prev_config.get("n") == len(scenarios)
                and prev_config.get("condition", "trace") == COND
                and prev_config.get("arms", list(ALL_ARMS)) == list(ARMS)):
            out["typer_prose_warnings"] = prev.get("typer_prose_warnings", {})
            out["carriers"] = prev.get("carriers", {})
            print(f"[resume] {len(out['typer_prose_warnings'])} warnings, "
                  f"carriers done: {sorted(out['carriers'])}", flush=True)

    # 1. typer's prose-warning briefings (once per scenario). Only this arm needs the typer at
    # all; templated and field_vs_prose reuse stored artifacts, so skip the spend otherwise.
    if "prose_warning" in ARMS:
        tclient = client_for(TYPER)
        n_warn = 0
        for i, sc in enumerate(scenarios):
            if sc["id"] not in out["typer_prose_warnings"]:
                brief = make_briefing_prose_warning(tclient, sc, rng)
                out["typer_prose_warnings"][sc["id"]] = brief
                json.dump(out, open(OUT, "w"), indent=2)
            b = out["typer_prose_warnings"][sc["id"]]
            n_warn += bool(EXPLICIT_RE.search(b))
        print(f"typer={TYPER}: explicit provenance language present (lexical) in "
              f"{n_warn}/{len(scenarios)} {COND} briefings", flush=True)

    # 2. carriers
    for model in CARRIERS:
        if model in out["carriers"]:
            continue
        client = client_for(model)
        agg = {arm: defaultdict(int) for arm in ARMS}
        rows = []
        for sc in scenarios:
            answers = {}
            own_plain = None
            if "prose_warning" in ARMS:
                warn_brief = out["typer_prose_warnings"][sc["id"]]
                answers["prose_warning"] = answer_from_briefing(client, sc, warn_brief)
            if "field_vs_prose" in ARMS:
                own_plain = make_briefing_plain(client, sc, rng, COND)
                answers["field_vs_prose"] = answer_from_briefing(
                    client, sc, own_plain, typer_fields[sc["id"]]
                )
            if "templated" in ARMS:
                answers["templated"] = answer_from_briefing_templated(
                    client, sc, typer_briefings[sc["id"]], typer_fields[sc["id"]]
                )
            v = {arm: judge(judge_c, sc, answers[arm]) for arm in ARMS}
            for arm in ARMS:
                agg[arm][v[arm]] += 1
            rows.append({"id": sc["id"], "topology": sc["topology"], "condition": COND,
                         "verdicts": v, "own_plain_briefing": own_plain, "answers": answers})
            print(f"  [{model}] {sc['id']} "
                  + " ".join(f"{arm}={v[arm][:6]}" for arm in ARMS), flush=True)
        n = len(scenarios)
        summary = {}
        print(f"\n=== carrier={model}, n={n} ===")
        for arm in ARMS:
            k = agg[arm]["established_fact"]
            lo, hi = wilson(k, n)
            summary[arm] = {"established_rate": round(k / n, 2), "established_n": f"{k}/{n}",
                            "launder": round(k / n, 2), "launder_n": f"{k}/{n}",
                            "ci95": [lo, hi], "dist": dict(agg[arm])}
            print(f"{arm:16s} established {k}/{n}  dist={dict(agg[arm])}")
        out["carriers"][model] = {"summary": summary, "rows": rows}
        json.dump(out, open(OUT, "w"), indent=2)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
