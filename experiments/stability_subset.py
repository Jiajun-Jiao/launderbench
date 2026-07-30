#!/usr/bin/env python3
"""Build the seeded stratified 30-scenario subset for the near-zero stability probe.

Draws 10 scenarios per topology (seed 20260729) from the 96 canonical scenarios that
have a matched solid control, so every subset item supports both probe conditions.
Output: outputs/society/citation_scenarios_100_v2_stab30.jsonl (sorted by id).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "society" / "citation_scenarios_100_v2.jsonl"
OUT = ROOT / "outputs" / "society" / "citation_scenarios_100_v2_stab30.jsonl"
SEED = 20260729
PER_TOPOLOGY = 10


def main() -> None:
    rows = [json.loads(l) for l in open(SRC) if l.strip()]
    pool = [r for r in rows if r.get("solid_pages")]
    random.seed(SEED)
    sel = []
    for topo in ("linear", "star", "fake_corroboration"):
        cands = sorted((r for r in pool if r["topology"] == topo), key=lambda r: r["id"])
        sel += random.sample(cands, PER_TOPOLOGY)
    sel.sort(key=lambda r: r["id"])
    with open(OUT, "w") as f:
        for r in sel:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {OUT.name}: {len(sel)} scenarios ({sel[0]['id']}..{sel[-1]['id']})")


if __name__ == "__main__":
    main()
