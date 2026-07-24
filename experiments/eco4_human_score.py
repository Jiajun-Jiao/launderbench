"""#1 Human-validation kit — STEP 2: score after annotation.
Reads the filled annotator CSVs (outputs/human_eval/annot_*.csv) + the hidden key
(key.jsonl), and reports, per epistemic property:
  - human inter-annotator agreement (average pairwise Cohen kappa + Fleiss kappa),
  - human-majority vs the automatic metric (% agree + Cohen kappa),
  - human-majority vs the gpt-4.1-mini judge (% agree + Cohen kappa).
This upgrades the §11 metric-validity claim from 'LLM-judge' to 'human-validated', and shows
whether the LLM judge is a valid stand-in for humans."""
import csv
import glob
import itertools
import json
import os
import sys
from collections import defaultdict

OUTDIR = os.environ.get("OUTDIR", "/path/to/launderbench/outputs/human_eval")
# (csv column, auto key, llm_judge key, human-True meaning aligns with auto-True)
PROPS = [
    ("consistent_YN", "ec", "only_says_what_npc_could_know", "EC (consistent)"),
    ("leaks_truth_YN", "olr", "asserts_true_answer_without_basis", "OLR (leaks truth)"),
    ("rumor_as_fact_YN", "raf", "states_rumor_as_established_fact", "RAF (rumor as fact)"),
    ("rumor_as_fact_YN", "fwr", "states_rumor_as_established_fact", "FWR (false witness)"),
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
        return None, 0
    po = sum(1 for a, b in pairs if a == b) / n
    pa = sum(1 for a, _ in pairs if a) / n
    pb = sum(1 for _, b in pairs if b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (1.0 if pe >= 1 else (po - pe) / (1 - pe)), n


def fleiss(rows):
    # rows: list of (n_true, n_false) for items rated by the SAME number of raters
    rows = [r for r in rows if sum(r) == max(sum(x) for x in rows)] if rows else []
    if not rows:
        return None, 0
    n = sum(rows[0])
    if n < 2:
        return None, 0
    N = len(rows)
    p_cat = [sum(r[c] for r in rows) / (N * n) for c in (0, 1)]
    pe = sum(p * p for p in p_cat)
    Pi = [(sum(x * x for x in r) - n) / (n * (n - 1)) for r in rows]
    pbar = sum(Pi) / N
    return (1.0 if pe >= 1 else (pbar - pe) / (1 - pe)), N


def main():
    # annotators[name][item_id][col] = bool|None. Prefer server-collected JSON, else CSV exports.
    server_dir = os.environ.get(
        "SERVER_DIR", os.path.normpath(os.path.join(OUTDIR, "..", "..", "web", "app", "annotations")))
    annotators = {}
    server_files = sorted(glob.glob(os.path.join(server_dir, "*.json")))
    if server_files:
        print(f"reading server-collected annotations from {server_dir}")
        for f in server_files:
            d = defaultdict(dict)
            for iid, rec in json.load(open(f)).items():
                for col, *_ in PROPS:
                    d[iid][col] = to_bool(rec.get(col, ""))
            annotators[os.path.splitext(os.path.basename(f))[0]] = d
    else:
        files = sorted(glob.glob(f"{OUTDIR}/annot_*.csv"))
        if not files:
            print(f"no annotations found: neither {server_dir}/*.json nor {OUTDIR}/annot_*.csv")
            return
        for f in files:
            d = defaultdict(dict)
            with open(f, newline="") as fh:
                for row in csv.DictReader(fh):
                    iid = row.get("item_id")
                    if not iid:
                        continue
                    for col, *_ in PROPS:
                        d[iid][col] = to_bool(row.get(col, ""))
            annotators[os.path.basename(f)] = d
    names = list(annotators)
    key = {}
    with open(f"{OUTDIR}/key.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            key[r["item_id"]] = r
    item_ids = list(key)
    print(f"annotators: {names}")
    print(f"items in key: {len(item_ids)}; filled per annotator: "
          + ", ".join(f"{n}={sum(1 for i in item_ids if any(annotators[n].get(i,{}).get(c[0]) is not None for c in PROPS))}" for n in names))

    print(f"\n{'property':22s}{'humanIAA(κ)':>13}{'Fleiss':>8}{'maj-vs-auto':>13}{'maj-vs-LLM':>12}")
    for col, akey, jkey, label in PROPS:
        # human inter-annotator: average pairwise Cohen
        kappas = []
        for a, b in itertools.combinations(names, 2):
            pairs = [(annotators[a].get(i, {}).get(col), annotators[b].get(i, {}).get(col)) for i in item_ids]
            k, n = cohen(pairs)
            if k is not None:
                kappas.append(k)
        iaa = sum(kappas) / len(kappas) if kappas else None
        # Fleiss over items rated by all annotators
        frows = []
        for i in item_ids:
            vals = [annotators[n].get(i, {}).get(col) for n in names]
            vals = [v for v in vals if v is not None]
            if len(vals) == len(names) and len(names) >= 2:
                frows.append((sum(1 for v in vals if v), sum(1 for v in vals if not v)))
        fk, fn = fleiss(frows)
        # human majority per item
        maj = {}
        for i in item_ids:
            vals = [annotators[n].get(i, {}).get(col) for n in names]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            t = sum(1 for v in vals if v)
            if t * 2 == len(vals):
                continue  # tie
            maj[i] = t * 2 > len(vals)
        auto_pairs = [(maj[i], bool(key[i]["auto"][akey])) for i in maj]
        ka, na = cohen(auto_pairs)
        llm_pairs = [(maj[i], bool(key[i]["llm_judge"][jkey])) for i in maj
                     if key[i].get("llm_judge") and jkey in key[i]["llm_judge"]]
        kl, nl = cohen(llm_pairs)
        f_iaa = f"{iaa:.2f}" if iaa is not None else "n/a"
        f_fk = f"{fk:.2f}" if fk is not None else "n/a"
        f_ka = f"{ka:.2f}(n{na})" if ka is not None else "n/a"
        f_kl = f"{kl:.2f}(n{nl})" if kl is not None else "n/a"
        print(f"{label:22s}{f_iaa:>13}{f_fk:>8}{f_ka:>13}{f_kl:>12}")
    print("\nκ guide (Landis–Koch): .21–.40 fair, .41–.60 moderate, .61–.80 substantial, .81+ almost perfect")


if __name__ == "__main__":
    main()
