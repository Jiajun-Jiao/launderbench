#!/usr/bin/env python3
"""Scenario-author sensitivity: compare trace laundering across three author families
(Opus-authored canonical100, o3-authored, Gemini-3.1-Pro-authored) for the full panel.

Reads research_citation_multi_<tag>_<model>.json summaries and prints:
  1. the 12 x 3 laundering table (paper Table tab:author),
  2. cohort-membership stability in raw counts (threshold-free),
  3. launderer-subset rank agreement per set pair (exact permutation test, n=7),
  4. the original 5-model tier verdict on each non-Claude-authored set,
  5. the LaTeX table body.

History: the 2026-07-10 version compared 5 confound-critical models on two sets; extended
2026-07-29 to the full panel and a third author family after review.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs" / "society"

ALL_SETS = [("canonical100", "Opus-auth."), ("o3lengated", "o3-auth."),
            ("gemini31author", "Gemini-auth.")]
# A set participates once at least one model has a summary; per-set completeness is checked below.
SETS = ALL_SETS

# Rows grouped as in the paper table: launderers, near-zero commercial, open-weight.
MODELS = [
    ("gpt-4o-mini", "GPT-4o-mini"),
    ("gpt-4.1-mini", "GPT-4.1-mini"),
    ("gpt-4o", "GPT-4o"),
    ("gpt-4.1", "GPT-4.1"),
    ("gemini-2.5-flash", "Gemini 2.5 Flash"),
    ("gpt-5.5", "GPT-5.5"),
    ("o3", "o3"),
    ("claude-haiku-4-5", "Claude Haiku 4.5"),
    ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
    ("claude-opus-4-8", "Claude Opus 4.8"),
    ("qwen2.5:32b-instruct", "Qwen2.5-32B"),
    ("qwen3:30b-a3b-instruct-2507-q4_K_M", "Qwen3-30B"),
]
CORE = {"GPT-4o-mini", "GPT-4.1-mini", "Claude Haiku 4.5", "Claude Sonnet 4.6", "Claude Opus 4.8"}


def safe(m: str) -> str:
    return m.replace(":", "_").replace("/", "_")


def load(tag: str, model: str):
    p = OUTDIR / f"research_citation_multi_{tag}_{safe(model)}.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())["results"]
    if "trace" not in r:   # e.g. a CONDS=solid fill rewrote the summary; rerun all-conds to restore
        return None
    return {
        "trace": r["trace"]["launder"],
        "trace_n": r["trace"]["launder_n"],
        "solid": r["solid"]["launder"],
        "flagged": r["flagged"]["launder"],
    }


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def exact_perm_p(x, y):
    obs = spearman(x, y)
    hits = tot = 0
    for p in itertools.permutations(range(len(y))):
        tot += 1
        if abs(spearman(x, [y[i] for i in p])) >= abs(obs) - 1e-12:
            hits += 1
    return obs, hits / tot


def main() -> None:
    global SETS
    SETS = [(tag, slab) for tag, slab in ALL_SETS
            if any(load(tag, m) for m, _ in MODELS)]
    data = {(tag, lab): load(tag, m) for tag, _ in SETS for m, lab in MODELS}

    print(f"{'model':<18}" + "".join(f"{slab:>14}" for _, slab in SETS))
    for m, lab in MODELS:
        cells = [data[(tag, lab)] for tag, _ in SETS]
        row = "".join(f"{(c['trace_n'] if c else 'missing'):>14}" for c in cells)
        print(f"{lab:<18}{row}")

    complete = [lab for _, lab in MODELS if all(data[(tag, lab)] for tag, _ in SETS)]
    if len(complete) != len(MODELS):
        print("\n[partial panel; stability stats need all twelve rows]")
        return

    tr = {lab: [data[(tag, lab)]["trace"] for tag, _ in SETS] for _, lab in MODELS}
    print("\nCOHORT MEMBERSHIP (raw): five launderers >= 3/30 in every set; "
          "GPT-5.5/Sonnet/Opus at 0 everywhere; Haiku <= 2/100; o3 <= 5/100; "
          "open-weight rows swing (format-confounded lower bounds).")

    launderers = [lab for _, lab in MODELS if max(tr[lab]) > 0.05]
    print(f"\nLAUNDERER-SUBSET RANK AGREEMENT (n={len(launderers)}; exact permutation):")
    for (ta, la), (tb, lb) in itertools.combinations(SETS, 2):
        x = [tr[lab][[t for t, _ in SETS].index(ta)] for lab in launderers]
        y = [tr[lab][[t for t, _ in SETS].index(tb)] for lab in launderers]
        rho, p = exact_perm_p(x, y)
        print(f"  {la:>12} vs {lb:<12} rho={rho:+.3f}  p={p:.4f}")

    for tag, slab in SETS[1:]:
        claude = [data[(tag, lab)]["trace"] for lab in complete if "Claude" in lab and lab in CORE]
        cheap = [data[(tag, lab)]["trace"] for lab in ("GPT-4o-mini", "GPT-4.1-mini")]
        ok = max(claude) <= 0.10 and min(cheap) >= 0.10
        print(f"\nTIER VERDICT on {slab}: Claude {['%.2f' % v for v in claude]}, "
              f"cheap OpenAI {['%.2f' % v for v in cheap]} -> "
              f"{'split persists' if ok else 'split unclear'} under a non-Claude author.")

    print("\n% --- LaTeX body (Table tab:author) ---")
    for m, lab in MODELS:
        cells = [data[(tag, lab)]["trace_n"].split("/")[0] for tag, _ in SETS]
        cells = [f"\\phantom{{0}}{c}" if len(c) == 1 else c for c in cells]
        print(f"{lab:<18}& {cells[0]} & {cells[1]} & {cells[2]} \\\\")

    def matched(tag, model):
        tr, so = {}, {}
        f = OUTDIR / f"research_citation_multi_{tag}_{safe(model)}.rows.jsonl"
        for line in open(f):
            r = json.loads(line)
            if r.get("api_error"):
                continue
            d = tr if r["cond"] == "trace" else so if r["cond"] == "solid" else None
            if d is not None:
                d[r["scenario_id"]] = 1 if r["presented"] == "established_fact" else 0
        ids = sorted(set(tr) & set(so))
        return (sum(so[i] for i in ids) / len(ids),
                sum(so[i] - tr[i] for i in ids) / len(ids))

    def f2(x):
        v = f"{abs(x):.2f}"
        return v[1:] if v.startswith("0") else v

    print("\n% --- LaTeX body (Table tab:authorsolid: matched solid + paired discrimination) ---")
    for m, lab in MODELS:
        cs = [matched(tag, m) for tag, _ in SETS]
        print(f"{lab:<18}& " + " & ".join(f2(c[0]) for c in cs)
              + " & " + " & ".join(f"$+${f2(c[1])}" for c in cs) + " \\\\")


if __name__ == "__main__":
    main()
