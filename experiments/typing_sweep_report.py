"""typing_sweep_report.py — aggregate typing_sweep_<model>.json into the typer-eligibility table.

Prints panel-ordered rows (trace correct, solid correct, typing-discrimination, predicted
downstream laundering per 100 = mistyped_trace x 0.97) plus the three measured anchors.
Zero API calls.
"""
import glob
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

ORDER = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "gpt-4.1", "gemini-2.5-flash",
         "o4-mini", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5-mini", "o3", "gpt-5.5",
         "claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-8",
         "qwen2.5_7b-instruct", "qwen2.5_32b-instruct", "qwen3_30b-a3b-instruct-2507-q4_K_M",
         "llama3.1_8b", "mistral_7b", "gemma2_9b"]
ANCHORS = {"claude-opus-4-8": "0/400 measured", "gpt-5-mini": "0/400 measured",
           "gpt-4.1-mini": "159/400 measured"}


def main():
    data = {}
    for p in glob.glob(os.path.join(_ROOT, "outputs", "society", "typing_sweep_*.json")):
        d = json.load(open(p))
        if "model" not in d:  # skip this script's own aggregate output
            continue
        data[d["model"].replace("/", "_").replace(":", "_")] = d
    rows = [m for m in ORDER if m in data] + sorted(set(data) - set(ORDER))
    print(f"{'model':38s}{'trace':>9}{'solid':>9}{'t-disc':>8}{'pred launder/100':>18}  anchor")
    out = {}
    for m in rows:
        d = data[m]
        tr, so = d["trace"], d["solid"]
        pred = round((1 - tr["rate"]) * 100 * 0.97, 1) if tr["rate"] is not None else None
        anchor = ANCHORS.get(d["model"], "")
        print(f"{m:38s}{tr['correct_n']:>9}{so['correct_n']:>9}"
              f"{d['typing_discrimination'] if d['typing_discrimination'] is not None else '-':>8}"
              f"{pred if pred is not None else '-':>18}  {anchor}")
        out[d["model"]] = {"trace": tr["correct_n"], "solid": so["correct_n"],
                           "typing_discrimination": d["typing_discrimination"],
                           "predicted_downstream_launder_per100": pred,
                           "anchor": anchor or None,
                           "api_errors": tr["api_errors"] + so["api_errors"]}
    path = os.path.join(_ROOT, "outputs", "society", "typing_sweep_table.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
