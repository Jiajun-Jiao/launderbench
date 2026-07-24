"""Verify the MODEL-VISIBLE content of seeded scenarios reuses NO text from the source datasets.

The seed adapter stores the original benchmark question (and short evidence-format snippets) in the
seed spec as a style exemplar for the generation prompt (benchmark_seed_adapter.py); the generator
is instructed to invent a new fictional entity/claim/question. Each scenario retains its seed under
the `external_seed` metadata key, which the evaluation harness never shows to models. This check
makes the Ethics-statement claim auditable: for every seeded scenario, compute the longest word
n-gram shared between the model-visible content (question + all trace/solid page text) and the
seed's source-dataset text (question + evidence_style snippets).

Usage: python3 experiments/seed_overlap_check.py
Output: outputs/society/seed_overlap_check.json (max shared n-gram per panel; expected 0 at n>=4)
"""
import json
import os
import re

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "outputs", "society", "seed_overlap_check.json")
PANELS = ("alce", "expertqa", "ragtruth")
MIN_N = 4


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()


def ngrams(ws, n):
    return {" ".join(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def longest_shared(a_words, b_words, lo=MIN_N, hi=12):
    for n in range(hi, lo - 1, -1):
        if ngrams(a_words, n) & ngrams(b_words, n):
            return n
    return 0


def main():
    report = {}
    for name in PANELS:
        scs = [json.loads(line) for line in open(
            os.path.join(_ROOT, "outputs", "society", f"citation_scenarios_{name}_v2.jsonl"))]
        q_max, p_max, matched = 0, 0, 0
        for sc in scs:
            seed = sc.get("external_seed")
            if not seed:
                continue
            matched += 1
            b = norm((seed.get("question") or "") + " " + (seed.get("evidence_style") or ""))
            q_max = max(q_max, longest_shared(norm(sc["question"]), b))
            pages = sc["pages"] + (sc.get("solid_pages") or [])
            p_max = max(p_max, longest_shared(norm(" ".join(p["text"] for p in pages)), b))
        report[name] = {"scenarios": len(scs), "with_seed": matched,
                        f"max_question_ngram_ge{MIN_N}": q_max,
                        f"max_pagetext_ngram_ge{MIN_N}": p_max}
        print(name, report[name])
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
