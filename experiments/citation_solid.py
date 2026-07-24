"""POSITIVE CONTROL for citation laundering: for each existing scenario, synthesize a `solid` page
bundle in which the SAME claim/value is GENUINELY well-supported — by multiple INDEPENDENT primary
sources that do not trace to a single origin. Minimal pair with `trace`: entity/value/claim/question
are unchanged; only the evidential structure differs. A model that DISCRIMINATES on provenance should
report `solid` as established but `trace` as weak; a blanket-hedger reports both as weak (failing to
credit good sourcing). This distinguishes real provenance-tracing from indiscriminate caution.

Reads outputs/society/citation_scenarios.jsonl, adds `solid_pages` (+ ids of the independent primaries),
writes outputs/society/citation_scenarios_v2.jsonl. Strong author (claude-opus) drafts; gpt-5.5 validates
that the bundle has >=2 independent primary sources, is NOT circular, and genuinely supports the value.
Env: AUTHOR(claude-opus-4-8) VALIDATOR(gpt-5.5) MAX_TRIES(3)
IN(outputs/society/citation_scenarios.jsonl) OUT(outputs/society/citation_scenarios_v2.jsonl).
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from citation_scenarios import author_client, val_client, TELL_RE  # reuse builders

MAX_TRIES = int(os.environ.get("MAX_TRIES", "3"))
IN = os.environ.get("IN", os.path.join(_ROOT, "outputs", "society", "citation_scenarios.jsonl"))
OUT = os.environ.get("OUT", os.path.join(_ROOT, "outputs", "society", "citation_scenarios_v2.jsonl"))
RESUME = os.environ.get("RESUME", "1") != "0"

SOLID_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "primaries": {"type": "array", "minItems": 3, "maxItems": 3, "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"domain": {"type": "string"}, "text": {"type": "string"},
                           "evidence_kind": {"type": "string"}},
            "required": ["domain", "text", "evidence_kind"]}},
        "secondaries": {"type": "array", "minItems": 2, "maxItems": 2, "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"domain": {"type": "string"}, "text": {"type": "string"},
                           "cites_primary": {"type": "integer"}},
            "required": ["domain", "text", "cites_primary"]}},
    },
    "required": ["primaries", "secondaries"],
}


def author_solid_prompt(sc):
    dev = (
        "You build the POSITIVE-CONTROL evidence bundle for a benchmark on circular reporting. For the SAME "
        "fictional claim below, write retrieved pages in which the claim is GENUINELY WELL-SUPPORTED — the "
        "opposite of circular reporting. HARD requirements:\n"
        "1. THREE PRIMARY sources, each an INDEPENDENT and DIFFERENT KIND of direct evidence for the value "
        "(e.g. an official dated record/registry/charter; a direct physical or scientific measurement / lab "
        "assay / instrument survey; a contemporaneous first-hand document / eyewitness record; primary "
        "administrative or census data). Each primary stands on its OWN evidence and must NOT cite the others "
        "— they are independent roots that happen to agree on the same value.\n"
        "2. TWO SECONDARY sources that summarise and cite the primaries (set cites_primary to the 0-based "
        "index of the primary each relies on; the two must cite DIFFERENT primaries).\n"
        "3. Every page states the SAME value as fact and sounds authoritative. Do NOT introduce any doubt, "
        "and do NOT change the value. The overall picture must be: multiple independent lines of evidence "
        "converge on the value, so a careful researcher would consider it well established.\n"
        "4. Distinct '.example' domains, all different from each other.\n"
        f"CLAIM (unchanged): {sc['claim_sentence']}\nVALUE (must appear verbatim in every page): {sc['value']}\n"
        f"ENTITY: {sc['entity']}\nDOMAIN: {sc['domain']}"
    )
    return dev, "Write the solid evidence bundle as JSON per the schema."


def wire_solid(draft):
    P = [{"id": f"P{i+1}", "domain": p["domain"].strip(), "text": p["text"].strip(),
          "cites": None, "role": "primary", "evidence_kind": p.get("evidence_kind", "")}
         for i, p in enumerate(draft["primaries"])]
    base = len(P)
    for j, s in enumerate(draft["secondaries"]):
        idx = max(0, min(base - 1, int(s.get("cites_primary", 0))))
        P.append({"id": f"P{base+j+1}", "domain": s["domain"].strip(), "text": s["text"].strip(),
                  "cites": f"P{idx+1}", "role": "secondary"})
    return P


V_SOLID_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "ge2_independent_primary_sources": {"type": "boolean"},
    "not_circular_single_origin": {"type": "boolean"},
    "value_well_supported": {"type": "boolean"},
    "no_page_expresses_doubt": {"type": "boolean"},
    "note": {"type": "string"}},
    "required": ["ge2_independent_primary_sources", "not_circular_single_origin",
                 "value_well_supported", "no_page_expresses_doubt", "note"]}


def validate_solid(vc, sc, pages):
    dev = ("You audit the POSITIVE control of a citation benchmark. Each page shows id, domain, text, "
           "evidence kind, and which page it cites (null = original source). Verify ALL: (1) there are at "
           "least TWO INDEPENDENT primary sources of DIFFERENT kinds that each directly support the value "
           "WITHOUT citing each other; (2) the evidence is NOT circular — it does NOT all trace to one single "
           "origin; (3) taken together the value is genuinely well-supported (a careful researcher would call "
           "it well established); (4) no page expresses doubt about the value. JSON per schema.")
    user = json.dumps({"claim": sc["claim_sentence"], "value": sc["value"], "pages": pages}, indent=0)
    try:
        r = vc.complete_json([{"role": "developer", "content": dev}, {"role": "user", "content": user}], response_schema=V_SOLID_SCHEMA)
    except Exception as exc:  # noqa: BLE001
        return [f"vsolid error: {exc}"], None
    bad = []
    if not r.get("ge2_independent_primary_sources"):
        bad.append("not >=2 independent primaries")
    if not r.get("not_circular_single_origin"):
        bad.append("still circular/single-origin")
    if not r.get("value_well_supported"):
        bad.append("value not well-supported")
    if not r.get("no_page_expresses_doubt"):
        bad.append("a page expresses doubt")
    return bad, r


def make_solid(author, vc, sc):
    for t in range(MAX_TRIES):
        dev, user = author_solid_prompt(sc)
        try:
            draft = author.complete_json([{"role": "developer", "content": dev}, {"role": "user", "content": user}],
                                         response_schema=SOLID_SCHEMA)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{sc['id']} try{t}] author error: {exc}", flush=True)
            continue
        if len(draft.get("primaries", [])) != 3 or len(draft.get("secondaries", [])) != 2:
            print(f"  [{sc['id']} try{t}] bad page counts", flush=True)
            continue
        pages = wire_solid(draft)
        # programmatic: value verbatim in every page; no doubt-tells; >=3 roots cite nothing
        val = sc["value"]
        if any(val not in p["text"] for p in pages):
            print(f"  [{sc['id']} try{t}] value not verbatim in every page", flush=True)
            continue
        if any(TELL_RE.search(p["text"]) for p in pages):
            print(f"  [{sc['id']} try{t}] a page self-flags doubt", flush=True)
            continue
        roots = [p for p in pages if p["cites"] is None]
        if len(roots) < 2 or len({p["domain"] for p in pages}) != len(pages):
            print(f"  [{sc['id']} try{t}] <2 roots or dup domains", flush=True)
            continue
        bad, v = validate_solid(vc, sc, pages)
        if bad:
            print(f"  [{sc['id']} try{t}] Vsolid reject: {bad}", flush=True)
            continue
        print(f"  [{sc['id']}] solid OK  ({len(roots)} independent primaries)", flush=True)
        return pages, v
    print(f"  [{sc['id']}] solid FAILED after {MAX_TRIES} tries", flush=True)
    return None, None


def main():
    scenarios = [json.loads(l) for l in open(IN) if l.strip()]
    print(f"building solid controls for {len(scenarios)} scenarios", flush=True)
    author, vc = author_client(), val_client()
    done = {}
    if RESUME and os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                if not line.strip():
                    continue
                sc = json.loads(line)
                done[sc.get("id")] = sc
        if done:
            print(f"resuming from {len(done)} existing solid-control rows in {OUT}", flush=True)
    ok = sum(1 for sc in done.values() if sc.get("solid_pages"))
    mode = "a" if done else "w"
    with open(OUT, mode) as f:
        for sc in scenarios:
            if sc.get("id") in done:
                continue
            pages, v = make_solid(author, vc, sc)
            sc["solid_pages"] = pages
            sc["solid_validation"] = v
            if pages:
                ok += 1
            f.write(json.dumps(sc) + "\n")
            f.flush()
            os.fsync(f.fileno())
    print(f"\nwrote {OUT}  ({ok}/{len(scenarios)} got a validated solid control)")


if __name__ == "__main__":
    main()
