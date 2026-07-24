"""Human-study scorer (A+B+C) — reads the typed study + server-collected annotations and reports the
validity numbers that fill enpc_paper_v1.md §7.5. Supersedes eco4_human_score.py (A-only). See
human_study_protocol.md §6.

Reads outputs/human_eval/study_items.jsonl + study_key.jsonl + web/app/annotations/*.json.

Reports:
  A (judge validity): per property — human IAA (Fleiss + avg pairwise Cohen, bootstrap CI),
     human-majority vs automatic metric (Cohen κ + CI), human-majority vs LLM-judge (Cohen κ + CI).
  Attention/anti-AI audit: per-annotator honeypot accuracy + timing/paste/blur anomalies.
  B (gold audit): gold error-rate (humans rejecting the benchmark's label) + Wilson CI.
  C (believability): preference for the licensed reply + Wilson CI; naturalness Δ(licensed − leaking).
Decision rule (pre-registered): judge "validated" for a property if its κ CI overlaps the human IAA CI.
"""
import glob
import itertools
import json
import math
import os
import random
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
OUTDIR = os.environ.get("OUTDIR", os.path.join(_ROOT, "outputs", "human_eval"))
SERVER_DIR = os.environ.get("SERVER_DIR", os.path.join(_ROOT, "web", "app", "annotations"))
HONEY_MIN = int(os.environ.get("HONEY_MIN", "13"))  # of 15; below -> exclude annotator

PROPS_A = [
    ("consistent_YN", "ec", "only_says_what_npc_could_know", "EC (consistent)"),
    ("leaks_truth_YN", "olr", "asserts_true_answer_without_basis", "OLR (leaks truth)"),
    ("rumor_as_fact_YN", "raf", "states_rumor_as_established_fact", "RAF (rumor as fact)"),
]


def to_bool(s):
    s = (s or "").strip().lower()
    if s in ("y", "yes", "1", "true", "t"):
        return True
    if s in ("n", "no", "0", "false", "f"):
        return False
    return None


def cohen(pairs):
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    pa = sum(1 for a, _ in pairs if a) / n
    pb = sum(1 for _, b in pairs if b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe >= 1 else (po - pe) / (1 - pe)


def fleiss(rows):
    rows = [r for r in rows if sum(r) >= 2]
    if not rows:
        return None
    maxn = max(sum(x) for x in rows)
    rows = [r for r in rows if sum(r) == maxn]
    n = maxn
    N = len(rows)
    p_cat = [sum(r[c] for r in rows) / (N * n) for c in (0, 1)]
    pe = sum(p * p for p in p_cat)
    Pi = [(sum(x * x for x in r) - n) / (n * (n - 1)) for r in rows]
    pbar = sum(Pi) / N
    return 1.0 if pe >= 1 else (pbar - pe) / (1 - pe)


def boot(items, stat, B=2000):
    """percentile bootstrap CI of stat(resample) over a list of per-item units."""
    items = [x for x in items if x is not None]
    base = stat(items)
    if base is None or len(items) < 3:
        return base, None, None
    rng = random.Random(0)
    n = len(items)
    s = []
    for _ in range(B):
        samp = [items[rng.randrange(n)] for _ in range(n)]
        v = stat(samp)
        if v is not None:
            s.append(v)
    if not s:
        return base, None, None
    s.sort()
    return base, s[int(0.025 * len(s))], s[int(0.975 * len(s))]


def wilson(k, n, z=1.96):
    if n == 0:
        return None, None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def fmt_ci(t):
    v, lo, hi = t
    if v is None:
        return "n/a"
    if lo is None:
        return f"{v:.2f}"
    return f"{v:.2f} [{lo:.2f},{hi:.2f}]"


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    items = {r["id"]: r for r in load_jsonl(os.path.join(OUTDIR, "study_items.jsonl"))}
    key = {r["item_id"]: r for r in load_jsonl(os.path.join(OUTDIR, "study_key.jsonl"))}
    files = sorted(glob.glob(os.path.join(SERVER_DIR, "*.json")))
    if not files:
        print(f"no annotations in {SERVER_DIR} yet. (items={len(items)}, key={len(key)})")
        return
    ann = {}
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        d = json.load(open(f))
        ann[name] = {iid: rec for iid, rec in d.items() if not iid.startswith("_")}
    names = list(ann)
    print(f"annotators: {names}")
    print(f"items: {len(items)}  (A={sum(1 for v in items.values() if v['type']=='A')}, "
          f"B={sum(1 for v in items.values() if v['type']=='B')}, "
          f"C={sum(1 for v in items.values() if v['type']=='C')})")

    # ---------- attention / anti-AI audit ----------
    print("\n=== attention & anti-AI audit ===")
    honey_ids = [i for i, k in key.items() if k.get("honeypot")]
    good_annotators = []
    for n in names:
        correct = tot = 0
        for hid in honey_ids:
            rec = ann[n].get(hid)
            hc = key[hid].get("honey_correct") or {}
            if not rec or not hc:
                continue
            tot += 1
            ok = all((to_bool(rec.get(col)) == to_bool(val)) for col, val in hc.items())
            correct += 1 if ok else 0
        dwell = [rec.get("telemetry", {}).get("dwell_ms") for rec in ann[n].values()
                 if isinstance(rec, dict) and rec.get("telemetry")]
        dwell = sorted(d for d in dwell if isinstance(d, (int, float)))
        med = dwell[len(dwell) // 2] / 1000.0 if dwell else None
        paste = sum(rec.get("telemetry", {}).get("paste", 0) for rec in ann[n].values() if isinstance(rec, dict))
        blur = sum(rec.get("telemetry", {}).get("blur", 0) for rec in ann[n].values() if isinstance(rec, dict))
        flag = "" if (tot == 0 or correct >= HONEY_MIN) else "  <-- EXCLUDE (honeypot fails)"
        keep = tot == 0 or correct >= HONEY_MIN
        if keep:
            good_annotators.append(n)
        med_s = f"{med:.0f}s" if med is not None else "n/a"
        print(f"  {n:12s} honeypots {correct}/{tot}  median_dwell {med_s}  pastes {paste}  blurs {blur}{flag}")
    if len(good_annotators) < len(names):
        print(f"  -> scoring with {good_annotators} (excluded {set(names)-set(good_annotators)})")
    names = good_annotators or names

    # ---------- A: judge validity ----------
    a_ids = [i for i, v in items.items() if v["type"] == "A" and not key.get(i, {}).get("honeypot")]
    have_judge = any(key.get(i, {}).get("llm_judge") for i in a_ids)
    print(f"\n=== A: judge validity (n A-items={len(a_ids)}, judge={'yes' if have_judge else 'NONE — run builder with RUN_JUDGE=1'}) ===")
    print(f"{'property':22s}{'humanIAA(κ)':>20}{'maj-vs-auto':>20}{'maj-vs-LLMjudge':>22}")
    for col, akey, jkey, label in PROPS_A:
        # majority per item
        maj, rows3 = {}, []
        for i in a_ids:
            vals = [to_bool(ann[n].get(i, {}).get(col)) for n in names]
            vals = [v for v in vals if v is not None]
            if len(vals) == len(names) and len(names) >= 2:
                rows3.append((sum(1 for v in vals if v), sum(1 for v in vals if not v)))
            if not vals:
                continue
            t = sum(1 for v in vals if v)
            if t * 2 != len(vals):
                maj[i] = t * 2 > len(vals)
        iaa = boot(rows3, fleiss) if rows3 else (None, None, None)
        auto_units = [(maj[i], bool(key[i]["auto"][akey])) for i in maj
                      if key.get(i, {}).get("auto", {}).get(akey) is not None]
        ka = boot(auto_units, cohen)
        if have_judge:
            jud_units = [(maj[i], bool(key[i]["llm_judge"][jkey])) for i in maj
                         if key.get(i, {}).get("llm_judge") and jkey in (key[i]["llm_judge"] or {})]
            kl = boot(jud_units, cohen)
        else:
            kl = (None, None, None)
        print(f"{label:22s}{fmt_ci(iaa):>20}{fmt_ci(ka):>20}{fmt_ci(kl):>22}")
    print("decision: judge validated for a property if its κ CI overlaps the human-IAA κ CI.")

    # ---------- B: gold audit ----------
    b_ids = [i for i, v in items.items() if v["type"] == "B"]
    print(f"\n=== B: gold audit (n B-items with labels) ===")
    for col, lab in (("licensing_ok_YN", "licensing rule wrong"), ("value_ok_YN", "true value wrong")):
        rej = tot = 0
        for i in b_ids:
            vals = [to_bool(ann[n].get(i, {}).get(col)) for n in names]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            tot += 1
            maj_ok = sum(1 for v in vals if v) * 2 > len(vals)
            if not maj_ok:
                rej += 1
        p, lo, hi = wilson(rej, tot)
        s = f"{p:.1%} [{lo:.1%},{hi:.1%}]" if p is not None else "n/a"
        print(f"  {lab:24s} error-rate {rej}/{tot} = {s}")

    # ---------- C: believability ----------
    c_ids = [i for i, v in items.items() if v["type"] == "C"]
    pref = tot = 0
    nat_lic, nat_leak = [], []
    for i in c_ids:
        lic = key.get(i, {}).get("licensed_side")
        choices = [ann[n].get(i, {}).get("more_believable") for n in names]
        choices = [c for c in choices if c in ("A", "B")]
        if choices and lic:
            a = sum(1 for c in choices if c == "A")
            if a * 2 != len(choices):
                tot += 1
                pref += 1 if (("A" if a * 2 > len(choices) else "B") == lic) else 0
        for n in names:
            rec = ann[n].get(i, {})
            for side, store in (("A", "naturalness_A"), ("B", "naturalness_B")):
                try:
                    val = float(rec.get(store))
                except (TypeError, ValueError):
                    continue
                (nat_lic if side == lic else nat_leak).append(val)
    p, lo, hi = wilson(pref, tot)
    print(f"\n=== C: believability (n pairs with majority) ===")
    s = f"{p:.1%} [{lo:.1%},{hi:.1%}]" if p is not None else "n/a"
    print(f"  prefer the LICENSED reply: {pref}/{tot} = {s}  (success if CI lower bound > 50%)")
    if nat_lic and nat_leak:
        print(f"  naturalness: licensed {sum(nat_lic)/len(nat_lic):.2f} vs leaking "
              f"{sum(nat_leak)/len(nat_leak):.2f}  (Δ {sum(nat_lic)/len(nat_lic)-sum(nat_leak)/len(nat_leak):+.2f})")
    print("\nκ guide (Landis–Koch): .41–.60 moderate, .61–.80 substantial, .81+ almost perfect")


if __name__ == "__main__":
    main()
