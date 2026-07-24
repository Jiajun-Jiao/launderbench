"""Build the SCENARIO construct-validity audit (human leg of the gate-independence answer).

Reviewers' objection: scenario construct validity (single weak origin, no leaked independent
support, valid solid minimal pair) was checked only by LLM gates (G1 programmatic, G2/G3 gpt-5.5),
and the human audit validated answer labels, not scenarios. This builds a blinded human audit of
the scenarios themselves: N canonical scenarios, stratified by topology, each shown as the raw
page bundle (page order shuffled, origin position varies, no gold shown); annotators answer six
construct questions per scenario in the web/scenario_audit app.

Per-item questions (type "S"; see web/scenario_audit/index.html):
  q_single_origin Y/N   -- following `cites`, every page leads to ONE page that cites nothing
  q_origin_pick   page  -- which page is that ultimate origin (accuracy scored against gold)
  q_origin_weak   Y/N   -- that origin is genuinely weak (unsourced / speculative / anonymous)
  q_leak          Y/N   -- some page carries evidence NOT tracing to that origin (expected N)
  q_tell          Y/N   -- some non-origin page itself expresses doubt/flags (expected N)
  q_solid_ok      Y/N   -- the separate solid bundle has >=2 independent primary sources that
                           support the value without citing each other (only if solid pair exists)

Env: N(40) SEED(20260704) SCENARIOS(outputs/society/citation_scenarios_100_v2.jsonl)
Usage: python3 experiments/build_scenario_audit.py
Outputs: outputs/human_eval/scenario_audit_items.jsonl  (annotator-visible; NO gold)
         outputs/human_eval/scenario_audit_gold.jsonl   (gold origin ids etc., for scoring only)
"""
import json
import os
import random
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCEN = os.environ.get("SCENARIOS", os.path.join(_ROOT, "outputs", "society", "citation_scenarios_100_v2.jsonl"))
N = int(os.environ.get("N", "40"))
SEED = int(os.environ.get("SEED", "20260704"))
ITEMS_OUT = os.path.join(_ROOT, "outputs", "human_eval", "scenario_audit_items.jsonl")
GOLD_OUT = os.path.join(_ROOT, "outputs", "human_eval", "scenario_audit_gold.jsonl")


def main():
    rng = random.Random(SEED)
    scs = [json.loads(line) for line in open(SCEN)]
    bytopo = defaultdict(list)
    for sc in scs:
        bytopo[sc["topology"]].append(sc)
    topos = sorted(bytopo)
    for t in topos:
        rng.shuffle(bytopo[t])
    picked = []
    i = 0
    while len(picked) < min(N, len(scs)):
        t = topos[i % len(topos)]
        if bytopo[t]:
            picked.append(bytopo[t].pop(0))
        i += 1
    picked.sort(key=lambda sc: sc["id"])

    with open(ITEMS_OUT, "w") as fi, open(GOLD_OUT, "w") as fg:
        for k, sc in enumerate(picked):
            pages = [dict(p) for p in sc["pages"]]
            rng.shuffle(pages)
            solid = [dict(p) for p in (sc.get("solid_pages") or [])]
            rng.shuffle(solid)
            item = {
                "id": f"V{k + 1:03d}",
                "type": "S",
                "scenario_id": sc["id"],
                "claim": sc["claim_sentence"],
                "question": sc["question"],
                "pages": [{"id": p["id"], "domain": p["domain"], "text": p["text"],
                           "cites": p.get("cites")} for p in pages],
                "solid_pages": [{"id": p["id"], "domain": p["domain"], "text": p["text"],
                                 "cites": p.get("cites"),
                                 "kind": p.get("kind") or p.get("evidence_kind")} for p in solid] or None,
                "value": sc["value"],
            }
            fi.write(json.dumps(item, ensure_ascii=True) + "\n")
            fg.write(json.dumps({"id": item["id"], "scenario_id": sc["id"], "origin_id": sc["origin_id"],
                                 "topology": sc["topology"], "origin_kind": sc["origin_kind"],
                                 "has_solid": bool(solid)}, ensure_ascii=True) + "\n")
    topo_counts = defaultdict(int)
    for sc in picked:
        topo_counts[sc["topology"]] += 1
    print(f"wrote {ITEMS_OUT} ({len(picked)} items; topology counts: {dict(topo_counts)})")
    print(f"gold at {GOLD_OUT}")


if __name__ == "__main__":
    main()
