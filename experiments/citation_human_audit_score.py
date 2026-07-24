#!/usr/bin/env python3
"""Score the citation/provenance human audit.

Reads:
  - outputs/human_eval/study_key.jsonl
  - web/app/annotations/*.json

Reports human inter-annotator agreement and human-majority agreement with the hidden LLM labels.
"""
from __future__ import annotations

import glob
import itertools
import json
import math
import os
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = Path(os.environ.get("OUTDIR", ROOT / "outputs" / "human_eval"))
SERVER_DIR = Path(os.environ.get("SERVER_DIR", ROOT / "web" / "app" / "annotations"))
LABELS = ["established_fact", "hedged", "traced_weak_origin", "debunked", "absent"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize_label(v: object) -> str | None:
    v = str(v or "").strip()
    return v if v in LABELS else None


def binary(v: str | None) -> bool | None:
    if v is None:
        return None
    return v == "established_fact"


def cohen(pairs: list[tuple[object, object]]) -> tuple[float | None, float | None, int]:
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    n = len(pairs)
    if n == 0:
        return None, None, 0
    po = sum(1 for a, b in pairs if a == b) / n
    cats = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    k = 1.0 if pe >= 1.0 else (po - pe) / (1.0 - pe)
    return k, po, n


def fleiss(rows: list[Counter]) -> tuple[float | None, int]:
    rows = [r for r in rows if sum(r.values()) >= 2]
    if not rows:
        return None, 0
    max_raters = max(sum(r.values()) for r in rows)
    rows = [r for r in rows if sum(r.values()) == max_raters]
    n = max_raters
    if n < 2:
        return None, 0
    N = len(rows)
    cats = sorted({c for r in rows for c in r})
    p_cat = {c: sum(r[c] for r in rows) / (N * n) for c in cats}
    pe = sum(p * p for p in p_cat.values())
    p_i = [(sum(v * v for v in r.values()) - n) / (n * (n - 1)) for r in rows]
    pbar = sum(p_i) / N
    return (1.0 if pe >= 1.0 else (pbar - pe) / (1.0 - pe)), N


def majority(labels: list[str | None]) -> str | None:
    labels = [x for x in labels if x is not None]
    if not labels:
        return None
    counts = Counter(labels)
    top = counts.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None
    return top[0][0]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None, float | None]:
    if n == 0:
        return None, None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def fmt_k(k: float | None, agree: float | None, n: int) -> str:
    if k is None:
        return "n/a"
    return f"k={k:.3f}, agree={agree:.3f}, n={n}"


def load_annotations() -> dict[str, dict[str, str | None]]:
    excluded = {n.strip() for n in os.environ.get("EXCLUDE", "").split(",") if n.strip()}
    ann: dict[str, dict[str, str | None]] = {}
    for path in sorted(glob.glob(str(SERVER_DIR / "*.json"))):
        name = Path(path).stem
        if name in excluded:
            continue
        raw = json.loads(Path(path).read_text())
        ann[name] = {
            iid: normalize_label(rec.get("speech_act"))
            for iid, rec in raw.items()
            if not iid.startswith("_") and isinstance(rec, dict)
        }
    return ann


def main() -> None:
    key = {r["item_id"]: r for r in read_jsonl(OUTDIR / "study_key.jsonl")}
    ann = load_annotations()
    if not ann:
        print(f"no server annotations found in {SERVER_DIR}")
        print(f"ready to score {len(key)} audit items once annotators submit labels")
        return

    names = sorted(ann)
    item_ids = list(key)
    print(f"annotators: {', '.join(names)}")
    print(f"items in key: {len(item_ids)}")
    for name in names:
        filled = sum(1 for iid in item_ids if ann[name].get(iid) is not None)
        print(f"  {name}: {filled}/{len(item_ids)} labeled")

    for a, b in itertools.combinations(names, 2):
        pairs = [(ann[a].get(iid), ann[b].get(iid)) for iid in item_ids]
        k, agree, n = cohen(pairs)
        print(f"pairwise exact {a} vs {b}: {fmt_k(k, agree, n)}")

    rows = []
    for iid in item_ids:
        rows.append(Counter(ann[n].get(iid) for n in names if ann[n].get(iid) is not None))
    fk, fn = fleiss(rows)
    print(f"Fleiss exact label: k={fk:.3f}, n={fn}" if fk is not None else "Fleiss exact label: n/a")

    maj = {
        iid: majority([ann[n].get(iid) for n in names])
        for iid in item_ids
    }
    maj = {iid: lab for iid, lab in maj.items() if lab is not None}
    print(f"human majority labels (exact 5-way): {len(maj)}/{len(item_ids)} items")

    # The binary laundering claim (established_fact vs not) is tested at binary
    # granularity, so its human reference is a majority over the BINARY label, not the
    # exact 5-way label. An item where annotators unanimously read "not established" but
    # split on the fine-grained non-established label (hedged vs traced_weak_origin) has a
    # clear binary answer and must count; exact-5-way majority (kept above) would drop it.
    bmaj = {}
    for iid in item_ids:
        vals = [b for b in (binary(ann[n].get(iid)) for n in names) if b is not None]
        if not vals:
            continue
        top = Counter(vals).most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            continue
        bmaj[iid] = top[0][0]
    print(f"human majority labels (binary): {len(bmaj)}/{len(item_ids)} items")

    for field, label in (("base_label", "vs gpt-4.1-mini judge"), ("strong_label", "vs gpt-5.5 judge")):
        pairs = [(maj[iid], key[iid].get(field)) for iid in maj if key[iid].get(field)]
        k, agree, n = cohen(pairs)
        print(f"exact {label}: {fmt_k(k, agree, n)}")
        bpairs = [(bmaj[iid], binary(key[iid].get(field))) for iid in bmaj if key[iid].get(field)]
        bk, bagree, bn = cohen(bpairs)
        print(f"binary established {label}: {fmt_k(bk, bagree, bn)}")

    for source in sorted({k.get("source") for k in key.values()}):
        ids = [iid for iid, k in key.items() if k.get("source") == source and iid in bmaj]
        est = sum(1 for iid in ids if bmaj[iid])
        p, lo, hi = wilson(est, len(ids))
        ci = "n/a" if p is None else f"{p:.1%} [{lo:.1%},{hi:.1%}]"
        print(f"human majority established rate, {source}: {est}/{len(ids)} = {ci}")


if __name__ == "__main__":
    main()
