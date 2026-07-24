#!/usr/bin/env python3
"""Failure-mode breakdown of laundered answers (canonical100, trace condition).

For every answer a judge scored as `established_fact` on a weak-origin `trace`
scenario, categorize the rationale the model gave by lexical markers. The
categories are ordered; each answer takes its first matching category.

  multiplicity      -- explicitly counts agreement across sources, the literal
                       "echo-count" signature ("multiple sources", "consistently
                       across sources", "corroborated", "independent sources")
  establishment     -- vague establishment with no source count ("well established",
                       "widely documented/accepted", "commonly cited")
  authority         -- treats a downstream record as authoritative
                       ("historical records", "archives", "encyclopedia")
  bare_assertion    -- states the value with no rationale at all

First-match order: multiplicity > establishment > authority > bare. Reports the
per-model distribution over laundered answers and the pooled cheap/mid-tier rate
at which laundering invokes an explicit multiplicity signal. LaTeX-ready body.
"""

from __future__ import annotations

import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs" / "society"

MODELS = [
    "gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "gpt-4.1",
    "qwen2.5_32b-instruct", "qwen3_30b-a3b-instruct-2507-q4_K_M",
]

MULTIPLICITY = re.compile(
    r"(multiple|several|various|numerous|many)\s+"
    r"(independent\s+)?(sources|accounts|references|records|documents|pages|citations)"
    r"|consisten\w*\s+(cited|reported|documented|referenced|mentioned)"
    r"|(cited|reported|documented|referenced|corroborated)\s+"
    r"(consistently\s+)?across"
    r"|across\s+(multiple|several|various|the|all)\s+(sources|accounts|pages)"
    r"|corroborat\w*|cross-?referenc\w*"
    r"|independent(ly)?\s+(sources|verif\w*|corroborat\w*|sourced)"
    r"|sources?\s+(agree|confirm|concur|corroborate)",
    re.I,
)
ESTABLISHMENT = re.compile(
    r"well[- ]?(established|documented|known|attested|supported)"
    r"|widely\s+(documented|accepted|recognized|known|reported|cited)"
    r"|broadly\s+(accepted|documented|supported)"
    r"|commonly\s+(cited|accepted|known|reported)"
    r"|generally\s+accepted",
    re.I,
)
AUTHORITY = re.compile(
    r"\b(historical (record|account)\w*|archiv\w*|encyclopedi\w*|"
    r"documentation|official record\w*|scholarly|primary source\w*|"
    r"heritage|register|records? (say|show|indicate))\b",
    re.I,
)


def categorize(ans: str) -> str:
    if MULTIPLICITY.search(ans):
        return "multiplicity"
    if ESTABLISHMENT.search(ans):
        return "establishment"
    if AUTHORITY.search(ans):
        return "authority"
    return "bare_assertion"


def safe(m: str) -> str:
    return m.replace(":", "_").replace("/", "_")


def load(model: str) -> list[dict]:
    for pat in (
        OUTDIR / f"research_citation_multi_canonical100_{safe(model)}.rows.jsonl",
        OUTDIR / f"research_citation_multi_canonical100_{model}.rows.jsonl",
    ):
        if pat.exists():
            return [json.loads(l) for l in pat.read_text().splitlines() if l.strip()]
    hits = glob.glob(str(OUTDIR / f"research_citation_multi_canonical100_*{safe(model)}*.rows.jsonl"))
    return [json.loads(l) for l in Path(hits[0]).read_text().splitlines() if l.strip()] if hits else []


def main() -> None:
    cats = ["multiplicity", "establishment", "authority", "bare_assertion"]
    per_model: dict[str, Counter] = {}
    pooled = Counter()
    pooled_launder = 0
    for m in MODELS:
        rows = [r for r in load(m) if r.get("cond") == "trace" and not r.get("api_error")]
        laundered = [r for r in rows if r.get("presented") == "established_fact"]
        c = Counter(categorize(r["answer"]) for r in laundered)
        per_model[m] = c
        if m in ("gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "gpt-4.1"):
            pooled += c
            pooled_launder += len(laundered)

    print(f"{'model':<40}{'N_laund':>8}  " + "  ".join(f"{c[:11]:>12}" for c in cats))
    for m in MODELS:
        c = per_model[m]
        tot = sum(c.values())
        if not tot:
            print(f"{m:<40}{0:>8}  (no laundered answers / no rows)")
            continue
        pcts = "  ".join(f"{c[cat]:>3} ({c[cat]/tot:>3.0%})" for cat in cats)
        print(f"{m:<40}{tot:>8}  {pcts}")

    print(f"\npooled OpenAI cheap/mid laundered answers: {pooled_launder}")
    for cat in cats:
        print(f"  {cat}: {pooled[cat]} ({pooled[cat]/pooled_launder:.0%})")
    consensus_share = pooled["multiplicity"] / pooled_launder
    print(f"\nEXPLICIT MULTIPLICITY/CONSENSUS signal in {consensus_share:.0%} of pooled laundered answers")

    print("\n% --- LaTeX table body (per-model laundered-answer failure modes) ---")
    for m in MODELS:
        c = per_model[m]
        tot = sum(c.values())
        if not tot:
            continue
        label = m.replace("_", "-").replace("qwen2.5-32b-instruct", "Qwen2.5-32B").replace(
            "qwen3-30b-a3b-instruct-2507-q4-K-M", "Qwen3-30B")
        cells = " & ".join(f"{c[cat]/tot:.0%}" for cat in cats)
        print(f"{label} & {tot} & {cells} \\\\")


if __name__ == "__main__":
    main()
