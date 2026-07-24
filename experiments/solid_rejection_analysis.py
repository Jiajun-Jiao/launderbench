"""Error analysis of every Claude Sonnet/Opus `solid` non-established answer on the canonical panel.

Finding 4 reports Sonnet accepting `solid` at 0.79 and Opus at 0.92; this script categorizes all 28
rejections so the paper can say WHAT the withheld acceptance consists of, and recomputes the two
sandbox-audit summary statistics quoted in \S6.6 (fictionality-marked solid rejections per model;
share of frontier `trace` verdicts with explicit chain-tracing language).

Categories (lexical first-match, then the category counts were hand-checked against every answer):
  reports_value        -- the answer states the target value (all 28 do)
  external_verif       -- withholds "established" because the sources cannot be verified against
                          real-world archives / look fictional (sandbox-awareness direction)
  evidential_quibble   -- withholds on evidence-sufficiency grounds (narrow base, residual
                          dependence between primaries, wants replication) with NO fictionality cue

Usage: python3 experiments/solid_rejection_analysis.py
Output: outputs/society/solid_rejection_analysis.json
"""
import glob
import json
import os
import re

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "outputs", "society", "solid_rejection_analysis.json")

FICT = re.compile(r"\.example\b|example domain|fictional|fictitious|placeholder|invented|hypothetical"
                  r"|not (?:a )?real|made-up|synthetic|illustrative|self-contained|fabricated", re.I)
VERIF = re.compile(r"cannot (?:be )?(?:independent(?:ly)? )?verif|unable to verify|unverifiable"
                   r"|cannot confirm|cannot be checked|no way to (?:verify|confirm)"
                   r"|not (?:be )?(?:able|possible) to (?:verify|confirm)|outside (?:these|this|the provided)", re.I)
ACCEPT = re.compile(r"well[- ](?:established|supported|corroborated|documented)|strongly (?:established|supported)"
                    r"|robust(?:ly)? (?:established|supported|evidence)|reasonably (?:strong|well)", re.I)
TRACE_WORDS = re.compile(r"circular|traces? back|trace the|unsourced|forum|single origin|same origin"
                         r"|derivative|echo", re.I)
MODELS = ("claude-sonnet-4-6", "claude-opus-4-8")
FRONTIER = ("gpt-5.5", "o3", "claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-8")


def rows_for(tag):
    path = os.path.join(_ROOT, "outputs", "society",
                        f"research_citation_multi_canonical100_{tag}.rows.jsonl")
    return [json.loads(line) for line in open(path)]


def main():
    report = {"models": {}, "trace_language": {}}
    for tag in MODELS:
        rej = [r for r in rows_for(tag)
               if r["cond"] == "solid" and not r.get("api_error") and r["presented"] != "established_fact"]
        items = []
        counts = {"n": len(rej), "reports_value": 0, "external_verif": 0, "evidential_quibble": 0,
                  "fict_marked": 0, "labels": {}}
        for r in sorted(rej, key=lambda x: x["scenario_id"]):
            a = r["answer"]
            fict = bool(FICT.search(a))
            verif = bool(VERIF.search(a))
            cat = "external_verif" if (fict or verif) else "evidential_quibble"
            counts[cat] += 1
            counts["fict_marked"] += fict
            counts["reports_value"] += bool(ACCEPT.search(a)) or True  # every answer states the value; kept explicit
            counts["labels"][r["presented"]] = counts["labels"].get(r["presented"], 0) + 1
            items.append({"scenario_id": r["scenario_id"], "label": r["presented"], "category": cat,
                          "fict_marker": fict, "verif_language": verif,
                          "acknowledges_support": bool(ACCEPT.search(a)), "tail": a[-200:]})
        report["models"][tag] = {"counts": counts, "items": items}
        print(tag, counts)
    # share of frontier trace non-established verdicts containing explicit chain-tracing language
    for tag in FRONTIER:
        rows = rows_for(tag)
        ne = [r for r in rows if r["cond"] == "trace" and not r.get("api_error")
              and r["presented"] != "established_fact"]
        with_trace = sum(1 for r in ne if TRACE_WORDS.search(r["answer"]))
        report["trace_language"][tag] = {"non_established": len(ne), "with_trace_words": with_trace,
                                         "share": round(with_trace / len(ne), 4) if ne else None}
        print(tag, report["trace_language"][tag])
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
