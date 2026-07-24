#!/usr/bin/env python3
"""Scenario diversity / templatedness statistics for the canonical100 set.

Quantifies that the 100 weak-origin scenarios (and their solid twins) are
lexically diverse rather than one template with names swapped: entity
uniqueness, pairwise content-word and bigram Jaccard between scenario
corpora, corpus-wide distinct n-gram ratios, and exact-sentence reuse
across scenarios. Writes outputs/society/scenario_diversity_summary.json
and prints a short report.
"""

from __future__ import annotations

import itertools
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "outputs" / "society" / "citation_scenarios_100_v2.jsonl"
OUT = ROOT / "outputs" / "society" / "scenario_diversity_summary.json"

STOPWORDS = set(
    """a an and are as at be but by for from has have in is it its of on or that the
    this to was were will with which who whose not no than then into over under after
    before between about against during through where when what how why all any each
    few more most other some such only own same so too very s t can just don should
    now he she they them his her their there here also been being do does did i we you
    your our us if would could may might must one two three""".split()
)

WORD_RE = re.compile(r"[a-z][a-z'\-]+")
SENT_RE = re.compile(r"[.!?]\s+")


def content_words(text: str) -> list[str]:
    return [w for w in WORD_RE.findall(text.lower()) if w not in STOPWORDS]


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def corpus_text(pages: list[dict]) -> str:
    return " ".join(p.get("text", "") for p in pages)


def pairwise_stats(bags: list[set]) -> dict:
    vals = [jaccard(a, b) for a, b in itertools.combinations(bags, 2)]
    return {
        "mean": round(statistics.mean(vals), 4),
        "median": round(statistics.median(vals), 4),
        "p95": round(sorted(vals)[int(0.95 * (len(vals) - 1))], 4),
        "max": round(max(vals), 4),
    }


def analyse(rows: list[dict], key: str) -> dict:
    texts = [corpus_text(r[key]) for r in rows]
    toks = [content_words(t) for t in texts]
    uni = [set(t) for t in toks]
    bi = [set(ngrams(t, 2)) for t in toks]

    all_toks = [w for t in toks for w in t]
    distinct = {
        f"distinct_{n}gram_ratio": round(
            len(set(ngrams(all_toks, n))) / max(1, len(ngrams(all_toks, n))), 4
        )
        for n in (1, 2, 3)
    }

    # exact-sentence reuse across scenarios (normalized whitespace/case)
    seen: dict[str, set[str]] = {}
    for r, t in zip(rows, texts):
        for s in SENT_RE.split(t):
            s = re.sub(r"\s+", " ", s).strip().lower()
            if len(s.split()) >= 6:
                seen.setdefault(s, set()).add(r["id"])
    reused = {s: sorted(ids) for s, ids in seen.items() if len(ids) > 1}

    return {
        "n_scenarios": len(rows),
        "pages_per_scenario": sorted({len(r[key]) for r in rows}),
        "mean_content_words_per_corpus": round(statistics.mean(len(t) for t in toks), 1),
        "pairwise_unigram_jaccard": pairwise_stats(uni),
        "pairwise_bigram_jaccard": pairwise_stats(bi),
        **distinct,
        "sentences_reused_across_scenarios": len(reused),
        "reused_examples": dict(itertools.islice(reused.items(), 5)),
    }


def main() -> None:
    rows = [json.loads(l) for l in SCENARIOS.read_text().splitlines() if l.strip()]

    entities = [r["entity"] for r in rows]
    first_tokens = [content_words(e)[0] if content_words(e) else e for e in entities]
    summary = {
        "scenario_file": str(SCENARIOS.relative_to(ROOT)),
        "n_scenarios": len(rows),
        "unique_entities": len(set(entities)),
        "unique_entity_head_names": len(set(first_tokens)),
        "unique_values": len(set(r["value"] for r in rows)),
        "unique_domains": len(set(r["domain"] for r in rows)),
        "unique_origin_kinds": len(set(r["origin_kind"] for r in rows)),
        "topology_counts": {
            t: sum(1 for r in rows if r["topology"] == t)
            for t in sorted(set(r["topology"] for r in rows))
        },
        "trace_pages": analyse(rows, "pages"),
        "solid_pages": analyse([r for r in rows if r.get("solid_pages")], "solid_pages"),
    }

    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2)[:4000])
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
