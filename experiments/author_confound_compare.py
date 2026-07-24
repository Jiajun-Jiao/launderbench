#!/usr/bin/env python3
"""Author-family confound check: compare laundering on Opus-authored (canonical100)
vs o3-authored scenarios for the confound-critical model set.

If the tier split (cheap OpenAI launders high; Claude models ~0, incl. the cheap
Claude Haiku) persists on scenarios authored by a frontier NON-Claude model (o3),
the reviewer's "Opus-authored scenarios favor Claude" confound is ruled out.

Reads research_citation_multi_<tag>_<model>.json summaries. Prints a side-by-side
table (trace laundering, discrimination) and a verdict line. LaTeX body at the end.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs" / "society"

MODELS = [
    ("gpt-4o-mini", "GPT-4o-mini", "cheap OpenAI"),
    ("gpt-4.1-mini", "GPT-4.1-mini", "cheap OpenAI"),
    ("claude-haiku-4-5", "Claude Haiku 4.5", "cheap Claude"),
    ("claude-sonnet-4-6", "Claude Sonnet 4.6", "frontier Claude"),
    ("claude-opus-4-8", "Claude Opus 4.8", "frontier Claude"),
]


def safe(m: str) -> str:
    return m.replace(":", "_").replace("/", "_")


def load(tag: str, model: str):
    p = OUTDIR / f"research_citation_multi_{tag}_{safe(model)}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def cell(d):
    if d is None:
        return None
    r = d["results"]
    trace = r["trace"]["launder"]
    solid = r["solid"]["launder"]
    discr = d.get("discrimination")
    if discr is None:
        discr = round(solid - trace, 2)  # definition: P(est|solid) - P(est|trace)
    return {
        "trace": trace,
        "trace_n": r["trace"]["launder_n"],
        "flagged": r["flagged"]["launder"],
        "solid": solid,
        "discr": discr,
    }


def main() -> None:
    print(f"{'model':<20}{'tier':<16}{'Opus-auth trace':>16}{'o3-auth trace':>16}{'Δ':>7}   discr(O/ o3)")
    print("-" * 92)
    rows = []
    for m, label, tier in MODELS:
        o = cell(load("canonical100", m))
        n = cell(load("o3author", m))
        rows.append((label, tier, o, n))
        if o is None or n is None:
            miss = "canonical100" if o is None else ""
            miss += " o3author" if n is None else ""
            print(f"{label:<20}{tier:<16}  [missing: {miss.strip()}]")
            continue
        dtxt = f"{n['trace']-o['trace']:+.2f}"
        print(f"{label:<20}{tier:<16}{o['trace_n']:>16}{n['trace_n']:>16}{dtxt:>7}   "
              f"{o['discr']:+.2f}/{n['discr']:+.2f}")

    print()
    complete = [(lab, tier, o, n) for lab, tier, o, n in rows if o and n]
    if len(complete) == len(MODELS):
        claude = [n["trace"] for lab, tier, o, n in complete if "Claude" in lab]
        cheap_oa = [n["trace"] for lab, tier, o, n in complete if tier == "cheap OpenAI"]
        print("VERDICT on o3-authored scenarios:")
        print(f"  Claude models trace-launder: {['%.2f'%x for x in claude]} (want ~0)")
        print(f"  cheap OpenAI trace-launder:  {['%.2f'%x for x in cheap_oa]} (want high)")
        ruled_out = max(claude) <= 0.10 and min(cheap_oa) >= 0.40
        print(f"  author-family confound {'RULED OUT' if ruled_out else 'NOT cleanly ruled out'}: "
              f"the tier split {'persists' if ruled_out else 'does not clearly persist'} "
              f"under a frontier non-Claude author.")

    print("\n% --- LaTeX body: Opus-authored vs o3-authored trace laundering ---")
    for lab, tier, o, n in rows:
        if o and n:
            print(f"{lab} & {o['trace_n']} & {n['trace_n']} \\\\")


if __name__ == "__main__":
    main()
