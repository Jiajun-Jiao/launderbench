"""Score the human SCENARIO construct-validity audit against gold and report agreement.

Reads web/scenario_audit/annotations/*.json + outputs/human_eval/scenario_audit_gold.jsonl and
reports, per question: per-annotator agreement with gold/expected direction, majority verdicts,
pairwise percent agreement and Cohen's kappa on double-annotated items, plus the headline numbers
for the paper: origin-identification accuracy and the construct pass rate.

Expected direction per gold: q_single_origin=Y, q_origin_pick=<gold origin_id>, q_origin_weak=Y,
q_leak=N, q_tell=N, q_solid_ok=Y (when a solid pair exists). A scenario "passes" an annotator if
all applicable answers match that direction; the paper reports pass rates and disagreements.

Usage: python3 experiments/scenario_audit_score.py
Output: outputs/human_eval/scenario_audit_score.txt (+ stdout)
"""
import glob
import json
import os
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANN_DIR = os.path.join(_ROOT, "web", "scenario_audit", "annotations")
GOLD = os.path.join(_ROOT, "outputs", "human_eval", "scenario_audit_gold.jsonl")
OUT = os.path.join(_ROOT, "outputs", "human_eval", "scenario_audit_score.txt")
OUT_JSON = os.path.join(_ROOT, "outputs", "human_eval", "scenario_audit_summary.json")
QUESTIONS = ("q_single_origin", "q_origin_weak", "q_leak", "q_tell", "q_solid_ok")
EXPECT = {"q_single_origin": "Y", "q_origin_weak": "Y", "q_leak": "N", "q_tell": "N", "q_solid_ok": "Y"}


def kappa(pairs):
    """Cohen's kappa for a list of (a, b) label pairs."""
    if not pairs:
        return None
    labels = sorted({x for p in pairs for x in p})
    n = len(pairs)
    po = sum(1 for a, b in pairs if a == b) / n
    pe = sum((sum(1 for a, _ in pairs if a == l) / n) * (sum(1 for _, b in pairs if b == l) / n)
             for l in labels)
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def fleiss_kappa(rows):
    """Fleiss' kappa for equal-width rows of categorical labels."""
    if not rows:
        return None
    m = len(rows[0])
    if m < 2 or any(len(row) != m for row in rows):
        return None
    labels = sorted({label for row in rows for label in row})
    p_bar = 0.0
    totals = {label: 0 for label in labels}
    for row in rows:
        counts = {label: row.count(label) for label in labels}
        p_bar += (sum(n * n for n in counts.values()) - m) / (m * (m - 1))
        for label, n in counts.items():
            totals[label] += n
    p_bar /= len(rows)
    p_e = sum((n / (len(rows) * m)) ** 2 for n in totals.values())
    return 1.0 if p_e == 1.0 else (p_bar - p_e) / (1 - p_e)


def majority(values):
    counts = {value: values.count(value) for value in set(values)}
    winner, n = max(counts.items(), key=lambda item: item[1])
    return winner if list(counts.values()).count(n) == 1 else None


def complete_items(data, gold):
    complete = {}
    for iid, rec in data.items():
        g = gold.get(iid)
        if not g:
            continue
        applicable = [q for q in QUESTIONS if not (q == "q_solid_ok" and not g["has_solid"])]
        answered = all(rec.get(q) in ("Y", "N") for q in applicable) and bool(rec.get("q_origin_pick"))
        if answered:
            complete[iid] = rec
    return complete


def main():
    gold = {json.loads(l)["id"]: json.loads(l) for l in open(GOLD)}
    lines = []

    def w(s=""):
        print(s)
        lines.append(s)

    anns = {}
    for path in sorted(glob.glob(os.path.join(ANN_DIR, "*.json"))):
        name = os.path.basename(path)[:-5]
        data = json.load(open(path))
        anns[name] = {k: v for k, v in data.items() if not k.startswith("_")}
    if not anns:
        raise SystemExit(f"no annotations in {ANN_DIR}")
    w(f"annotators: { {n: len(a) for n, a in anns.items()} } of {len(gold)} items")

    # per annotator vs expected/gold
    for name, data in anns.items():
        stats = defaultdict(lambda: [0, 0])
        origin_ok = origin_n = 0
        passed = full = 0
        for iid, rec in data.items():
            g = gold.get(iid)
            if not g:
                continue
            applicable = [q for q in QUESTIONS if not (q == "q_solid_ok" and not g["has_solid"])]
            answered = all(rec.get(q) in ("Y", "N") for q in applicable) and bool(rec.get("q_origin_pick"))
            if not answered:
                continue
            full += 1
            ok = True
            for q in applicable:
                stats[q][1] += 1
                if rec.get(q) == EXPECT[q]:
                    stats[q][0] += 1
                else:
                    ok = False
            origin_n += 1
            if rec.get("q_origin_pick") == g["origin_id"]:
                origin_ok += 1
            else:
                ok = False
            passed += ok
        w(f"\n== {name}: {full} complete items ==")
        for q in QUESTIONS:
            k, n = stats[q]
            if n:
                w(f"  {q:16s} matches expected {k}/{n}")
        w(f"  origin_pick == gold origin: {origin_ok}/{origin_n}")
        w(f"  full construct pass: {passed}/{full}")

    # pairwise agreement on shared items
    names = sorted(anns)
    w("\n== pairwise agreement (shared complete items) ==")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = anns[names[i]], anns[names[j]]
            shared = [iid for iid in a if iid in b and iid in gold]
            for q in QUESTIONS + ("q_origin_pick",):
                pairs = [(a[iid].get(q), b[iid].get(q)) for iid in shared
                         if a[iid].get(q) and b[iid].get(q)]
                if len(pairs) >= 5:
                    agree = sum(1 for x, y in pairs if x == y)
                    w(f"  {names[i]}/{names[j]} {q:16s} {agree}/{len(pairs)} "
                      f"kappa={kappa(pairs):.3f}" if kappa(pairs) is not None else "")
    complete = {name: complete_items(data, gold) for name, data in anns.items()}
    primary_names = sorted(name for name, data in complete.items() if len(data) == len(gold))
    summary = {"n_items": len(gold), "annotator_complete_counts": {n: len(a) for n, a in complete.items()},
               "primary_annotators": primary_names}
    if len(primary_names) >= 3:
        w(f"\n== primary complete-annotator analysis ({', '.join(primary_names)}) ==")
        question_summary = {}
        dissent_audit_ids = set()
        for q in QUESTIONS:
            item_ids = [iid for iid, g in gold.items() if not (q == "q_solid_ok" and not g["has_solid"])]
            rows = [[complete[name][iid][q] for name in primary_names] for iid in item_ids]
            majority_ok = sum(majority(row) == EXPECT[q] for row in rows)
            unanimous = sum(len(set(row)) == 1 for row in rows)
            dissent = [iid for iid, row in zip(item_ids, rows) if len(set(row)) > 1]
            dissent_audit_ids.update(dissent)
            fk = fleiss_kappa(rows)
            w(f"  {q:16s} majority matches expected {majority_ok}/{len(rows)}; "
              f"unanimous {unanimous}/{len(rows)}; Fleiss kappa={fk:.3f}")
            if dissent:
                w("    one-rater disagreement: " + ", ".join(
                    f"{iid}({gold[iid]['scenario_id']})" for iid in dissent))
            question_summary[q] = {"majority_matches_expected": majority_ok, "n": len(rows),
                                   "unanimous": unanimous, "fleiss_kappa": round(fk, 6),
                                   "dissent_audit_ids": dissent,
                                   "dissent_scenario_ids": [gold[iid]["scenario_id"] for iid in dissent]}

        origin_rows = [[complete[name][iid]["q_origin_pick"] for name in primary_names] for iid in gold]
        origin_majority_ok = sum(majority(row) == gold[iid]["origin_id"]
                                 for iid, row in zip(gold, origin_rows))
        origin_unanimous = sum(len(set(row)) == 1 for row in origin_rows)
        origin_dissent = [iid for iid, row in zip(gold, origin_rows) if len(set(row)) > 1]
        dissent_audit_ids.update(origin_dissent)
        w(f"  {'q_origin_pick':16s} majority matches gold {origin_majority_ok}/{len(origin_rows)}; "
          f"unanimous {origin_unanimous}/{len(origin_rows)}; "
          f"Fleiss kappa={fleiss_kappa(origin_rows):.3f}")
        if origin_dissent:
            w("    one-rater disagreement: " + ", ".join(
                f"{iid}({gold[iid]['scenario_id']})" for iid in origin_dissent))

        majority_pass = 0
        for iid, g in gold.items():
            applicable = [q for q in QUESTIONS if not (q == "q_solid_ok" and not g["has_solid"])]
            q_ok = all(majority([complete[name][iid][q] for name in primary_names]) == EXPECT[q]
                       for q in applicable)
            origin_ok = majority([complete[name][iid]["q_origin_pick"] for name in primary_names]) == g["origin_id"]
            majority_pass += int(q_ok and origin_ok)
        w(f"  full construct majority pass: {majority_pass}/{len(gold)}")
        summary.update({"question_summary": question_summary,
                        "origin_majority_matches_gold": origin_majority_ok,
                        "origin_unanimous": origin_unanimous,
                        "origin_fleiss_kappa": round(fleiss_kappa(origin_rows), 6),
                        "origin_dissent_audit_ids": origin_dissent,
                        "full_construct_majority_pass": majority_pass,
                        "one_rater_dissent_audit_ids": sorted(dissent_audit_ids),
                        "one_rater_dissent_scenario_ids": sorted(gold[iid]["scenario_id"]
                                                                  for iid in dissent_audit_ids)})

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {OUT}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
