"""Analyze prov_pipeline_*.json: cross-model laundering table + the direct->downstream FLIPS
(scenarios a model got right reading the pages but laundered after its own compression), with the
briefing text that dropped the caveat. Usage: python3 experiments/prov_pipeline_analyze.py [model ...]"""
import glob
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOC = os.path.join(_ROOT, "outputs", "society")
ARMS = ("direct", "briefing", "downstream", "carry")
ORDER = {"claude-opus-4-8": 0, "gpt-5.5": 1, "o3": 2, "claude-sonnet-4-6": 3, "claude-haiku-4-5": 4,
         "gpt-4.1": 5, "gpt-4.1-mini": 6, "gpt-4o": 7, "gpt-4o-mini": 8}


def load():
    want = sys.argv[1:]
    files = sorted(glob.glob(os.path.join(SOC, "prov_pipeline_*.json")))
    out = {}
    for f in files:
        d = json.load(open(f))
        m = d["config"]["model"]
        if want and m not in want:
            continue
        out[m] = d
    return dict(sorted(out.items(), key=lambda kv: ORDER.get(kv[0], 99)))


def main():
    data = load()
    if not data:
        print("no prov_pipeline_*.json yet")
        return
    n = next(iter(data.values()))["config"]["n"]
    print(f"=== provenance-pipeline laundering (established_fact rate), n={n} ===")
    print(f"{'model':20s}{'direct':>8}{'briefing':>10}{'downstrm':>10}{'carry':>8}   self-laundering(down-direct)")
    for m, d in data.items():
        s = d["summary"]
        g = s["downstream"]["launder"] - s["direct"]["launder"]
        print(f"{m:20s}{s['direct']['launder']:>8.2f}{s['briefing']['launder']:>10.2f}"
              f"{s['downstream']['launder']:>10.2f}{s['carry']['launder']:>8.2f}   {g:+.2f}")

    for m, d in data.items():
        flips = [r for r in d["rows"]
                 if r["verdicts"]["direct"] != "established_fact"
                 and r["verdicts"]["downstream"] == "established_fact"]
        if not flips:
            continue
        print(f"\n--- {m}: {len(flips)} direct->downstream FLIP(s) (kept caveat on pages, laundered its own briefing) ---")
        for r in flips:
            print(f"  {r['id']} {r['topology']} value={r['value']} "
                  f"(direct={r['verdicts']['direct']}, briefing={r['verdicts']['briefing']}, carry={r['verdicts']['carry']})")
            print(f"    briefing(plain): {r['briefing_plain']}")
            print(f"    prov_field: {json.dumps(r['provenance_field'])}")


if __name__ == "__main__":
    main()
