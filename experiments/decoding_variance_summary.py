#!/usr/bin/env python3
"""Summarize decoding-variance spot-check (TEMP=0.7, REPS=3, trace).

Reads research_citation_multi_var_t07_<model>.rows.jsonl, computes the
laundering rate (fraction of trace answers scored established_fact) per
independent sampled rep, and reports mean +/- SD across reps against the
greedy (TEMP=0) headline rate. Shows that the headline is not an artifact of
deterministic decoding.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs" / "society"

# greedy headline laundering rate on canonical100 trace (Table 2)
GREEDY = {"gpt-4o-mini": 0.96, "gpt-4.1-mini": 0.74}


def main() -> None:
    for model in ("gpt-4o-mini", "gpt-4.1-mini"):
        path = OUTDIR / f"research_citation_multi_var_t07_{model}.rows.jsonl"
        if not path.exists():
            print(f"{model}: no rows yet ({path.name})")
            continue
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        rows = [r for r in rows if r.get("cond") == "trace" and not r.get("api_error")]
        by_rep: dict[int, list[bool]] = defaultdict(list)
        for r in rows:
            by_rep[int(r.get("rep", 0))].append(r.get("presented") == "established_fact")
        rates = []
        for rep in sorted(by_rep):
            v = by_rep[rep]
            rates.append(sum(v) / len(v))
        if not rates:
            print(f"{model}: no scored trace rows")
            continue
        mean = statistics.mean(rates)
        sd = statistics.pstdev(rates) if len(rates) > 1 else 0.0
        per = ", ".join(f"{r:.2f}" for r in rates)
        n_per = len(by_rep[sorted(by_rep)[0]])
        print(
            f"{model}: greedy={GREEDY.get(model, float('nan')):.2f} | "
            f"temp0.7 reps=[{per}] (n={n_per}/rep) | "
            f"mean={mean:.3f} SD={sd:.3f} range={max(rates)-min(rates):.3f}"
        )


if __name__ == "__main__":
    main()
