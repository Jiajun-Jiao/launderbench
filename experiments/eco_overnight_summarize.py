"""Summarize the unattended pipeline outputs (outputs/overnight/) into compact tables:
the provenance extractor curve, the FP/FPR result, and the scaled LLM eval."""
import json
import os

OD = "outputs/overnight"

print("=== Provenance extractor curve (controller fixed = qwen2.5:32b) ===")
print(f"  {'extractor':22s}{'src_acc':>8}{'presence':>9}{'val_acc':>8}{'EC_ext':>8}{'OLR_ext':>8}{'SAA_ext':>8}")
rows = []
for f in ["curve_7b.txt", "curve_32b.txt", "curve_nano.txt", "curve_gpt41.txt"]:
    p = os.path.join(OD, f)
    if not os.path.exists(p):
        print(f"  {f}: (missing)")
        continue
    lines = [l for l in open(p) if l.startswith("CURVE_JSON")]
    if not lines:
        print(f"  {f}: (no CURVE_JSON — still running or failed)")
        continue
    d = json.loads(lines[-1][len("CURVE_JSON "):])
    rows.append(d)
    print(f"  {d['extractor']:22s}{d['src_acc']:>8.2f}{d['presence']:>9.2f}{d['val_acc']:>8.2f}"
          f"{d['EC_ext']:>8.2f}{d['OLR_ext']:>8.2f}{d['SAA_ext']:>8.2f}")
if rows:
    print(f"\n  EC on GOLD state (reference) = {rows[0]['EC_gold']:.2f}; "
          f"gpt-4.1-mini extractor cost = ${rows[-1].get('extractor_openai_cost', 0):.3f}")
    print("  Read: as extraction source-type accuracy rises, downstream EC rises and OLR falls.")

for name, f in [("FP / FPR result", "fp.txt"), ("Scaled-100 LLM eval", "full250.txt")]:
    p = os.path.join(OD, f)
    print(f"\n=== {name} ({f}) ===")
    print("".join(open(p).readlines()[-16:]) if os.path.exists(p) else "  (not produced yet)")
