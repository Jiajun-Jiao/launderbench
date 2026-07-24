"""Compare the extractor sweep on SHALLOW vs HARD worlds: does Δ(struct − graph) open up when the
information-flow gets complex? Tests the pre-registered prediction (hard ΔEC/Δsrc > shallow, esp. for
strong models). Reads the 4 sweep JSONs; robust to missing/partial. Run after the hard sweeps finish.
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def load(*names):
    d = {}
    for n in names:
        p = os.path.join(_ROOT, "outputs", n)
        if os.path.exists(p):
            try:
                d.update(json.load(open(p)))
            except Exception:  # noqa: BLE001
                pass
    return d


ORDER = ["mistral:7b", "gemma2:9b", "llama3.1:8b", "qwen2.5:7b-instruct",
         "qwen3:30b-a3b-instruct-2507-q4_K_M", "qwen2.5:32b-instruct", "gpt-4.1-nano",
         "claude-haiku-4-5", "gpt-4.1-mini", "claude-sonnet-4-6", "gpt-4.1"]
SHORT = {"qwen3:30b-a3b-instruct-2507-q4_K_M": "qwen3:30b-a3b", "qwen2.5:7b-instruct": "qwen2.5:7b",
         "qwen2.5:32b-instruct": "qwen2.5:32b"}


def deltas(r):
    if not r or "error" in r or "narrative_graph" not in r or "narrative_struct" not in r:
        return None
    g, s = r["narrative_graph"], r["narrative_struct"]
    ge, se = (g.get("EC") or 0), (s.get("EC") or 0)
    return {"dsrc": round(s["src"] - g["src"], 2), "dec": round(se - ge, 2),
            "g_ec": ge, "s_ec": se}


def main():
    shallow = load("extractor_sweep.json", "extractor_sweep_claude.json")
    hard = load("extractor_sweep_hard.json", "extractor_sweep_hard_local.json", "extractor_sweep_hard_claude.json")
    models = [m for m in ORDER if m in shallow or m in hard]
    for m in list(shallow) + list(hard):
        if m not in models:
            models.append(m)

    print("Δ = narrative_struct − narrative_graph (our method's net gain). Prediction: hard Δ > shallow Δ.\n")
    print(f"{'model':22s}| {'shallow Δsrc/ΔEC':>18} | {'HARD Δsrc/ΔEC':>16} | {'EC gap shift':>13} | opened?")
    print("-" * 88)
    shifts, opened = [], 0
    n_both = 0
    for m in models:
        sh, hd = deltas(shallow.get(m)), deltas(hard.get(m))
        name = SHORT.get(m, m)
        sh_s = f"{sh['dsrc']:+.2f}/{sh['dec']:+.2f}" if sh else "   —   "
        hd_s = f"{hd['dsrc']:+.2f}/{hd['dec']:+.2f}" if hd else "   —   "
        if sh and hd:
            n_both += 1
            ec_shift = hd["dec"] - sh["dec"]          # how much struct improved (vs graph) going easy->hard
            shifts.append(ec_shift)
            is_open = (ec_shift > 0.03) or (hd["dec"] > sh["dec"] and hd["dec"] >= 0)
            opened += int(is_open)
            flag = f"{ec_shift:+.2f}".rjust(13) + ("   ✓ opened" if is_open else "   ·")
        else:
            flag = "—".rjust(13)
        print(f"{name:22s}| {sh_s:>18} | {hd_s:>16} | {flag}")

    print("-" * 88)
    if shifts:
        avg = sum(shifts) / len(shifts)
        strong = ["gpt-4.1", "claude-sonnet-4-6", "gpt-4.1-mini", "claude-haiku-4-5"]
        strong_shifts = [hd["dec"] - sh["dec"] for m in strong
                         if (sh := deltas(shallow.get(m))) and (hd := deltas(hard.get(m)))]
        print(f"models compared on both: {n_both} | EC-gap opened (struct gained vs graph) on {opened}/{n_both}")
        print(f"avg ΔEC shift (hard − shallow) across models: {avg:+.3f}")
        if strong_shifts:
            print(f"avg ΔEC shift on STRONG models {strong}: {sum(strong_shifts)/len(strong_shifts):+.3f}")
        print("\nVERDICT GUIDE: prediction holds if the shift is broadly positive (esp. strong models) —")
        print("then the method's value lives in COMPLEX information-flow (real agent societies), and the")
        print("shallow-world null was a saturation artifact. If shifts are ~0/negative, accept no gain.")
    else:
        print("(hard results not in yet — rerun after extractor_sweep_hard*.json exist)")


if __name__ == "__main__":
    main()
