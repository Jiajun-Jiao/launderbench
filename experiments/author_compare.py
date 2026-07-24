"""Compare world AUTHORS: gpt-4.1-mini (current) vs gpt-4.1 (stronger). Same prompt/schema/validator
(from eco_worlds), K attempts each. Reports validity rate + richness (NPCs / distinct source types /
claims / distorted-rumor present) and dumps example worlds so the difference is visible.
Env: K(10), MODELS(gpt-4.1-mini,gpt-4.1). Needs OPENAI_API_KEY + egress."""
import json
import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eco_worlds import author_world, validate_and_build
from llm_controller import OpenAICompatibleChatClient

K = int(os.environ.get("K", "10"))
MODELS = [m.strip() for m in os.environ.get("MODELS", "gpt-4.1-mini,gpt-4.1").split(",") if m.strip()]
BASE = os.environ.get("BASE", "https://api.openai.com/v1")
KEY = os.environ["OPENAI_API_KEY"]
OUTDIR = os.path.join(_ROOT, "outputs", "author_compare")

PRICE = {"gpt-4.1": (2.0, 8.0), "gpt-4.1-mini": (0.40, 1.60), "gpt-4.1-nano": (0.10, 0.40)}


def src_types(w):
    f = w.primary_fact()
    s = set()
    for n in w.npcs:
        c = w.epistemic_states[n.npc_id].best_claim(f.proposition_id)
        if c:
            s.add(c.source_type)
    return s


def has_distorted_rumor(raw, w):
    """a hearsay claim whose value differs from the true value (the intended rumor distortion)."""
    tv = str(raw.get("true_value", "")).strip().lower()
    for c in raw.get("claims", []):
        if c.get("source_type") == "hearsay" and str(c.get("value", "")).strip().lower() != tv:
            return True
    return False


def example_str(raw):
    L = [f"  scenario: {raw.get('scenario_label')!r}",
         f"  proposition: {raw.get('proposition')!r}",
         f"  true={raw.get('true_value')!r}  false={raw.get('false_value')!r}  fp={raw.get('is_false_presupposition')}",
         f"  NPCs ({len(raw.get('npcs', []))}):"]
    for n in raw.get("npcs", []):
        L.append(f"    - {n['id']}: {n['name']} ({n['role']}) @ {n['location']}")
    for e in raw.get("events", []):
        L.append(f"  event: {e['description'][:90]!r}  observers={e.get('observers')}")
    L.append("  claims:")
    for c in raw.get("claims", []):
        L.append(f"    - {c['npc']}: {c['value']!r} [{c['source_type']}"
                 + (f" from {c['source_agent']}" if c.get("source_agent") else "") + f"]  “{c['memory_text'][:80]}”")
    return "\n".join(L)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    summary = {}
    for model in MODELS:
        client = OpenAICompatibleChatClient(model=model, base_url=BASE, api_key=KEY,
                                            temperature=0.5, use_structured_output=True)
        valid, raws_valid, reasons = 0, [], Counter()
        rich = {"npcs": [], "src": [], "claims": [], "distort": 0}
        for i in range(K):
            try:
                raw = author_world(client, i)
            except Exception as exc:  # noqa: BLE001
                reasons[f"author_err:{str(exc)[:40]}"] += 1
                continue
            w, msg = validate_and_build(raw, f"cmp_{i:03d}")
            if w is None:
                reasons[msg.split(":")[0]] += 1
                continue
            valid += 1
            raws_valid.append(raw)
            rich["npcs"].append(len(w.npcs))
            rich["src"].append(len(src_types(w)))
            rich["claims"].append(sum(1 for n in w.npcs if w.epistemic_states[n.npc_id].best_claim(w.primary_fact().proposition_id)))
            rich["distort"] += int(has_distorted_rumor(raw, w))
        ci, co = PRICE.get(model, (0, 0))
        cost = client.input_tokens / 1e6 * ci + client.output_tokens / 1e6 * co
        avg = lambda xs: round(sum(xs) / len(xs), 2) if xs else 0  # noqa: E731
        summary[model] = {
            "valid": valid, "attempts": K, "validity_rate": round(valid / K, 2),
            "avg_npcs": avg(rich["npcs"]), "avg_src_types": avg(rich["src"]),
            "avg_claims": avg(rich["claims"]), "with_distorted_rumor": rich["distort"],
            "reasons": dict(reasons), "cost_usd": round(cost, 3),
        }
        json.dump(raws_valid, open(os.path.join(OUTDIR, f"{model}_worlds.json"), "w"), ensure_ascii=False, indent=1)
        print(f"\n########## AUTHOR = {model} ##########")
        print(json.dumps(summary[model], ensure_ascii=False, indent=1))
        if raws_valid:
            print(f"\n--- EXAMPLE world by {model} ---")
            print(example_str(raws_valid[0]))

    print("\n\n================= SIDE BY SIDE =================")
    cols = ["validity_rate", "avg_npcs", "avg_src_types", "avg_claims", "with_distorted_rumor", "cost_usd"]
    print(f"{'metric':22s}" + "".join(f"{m:>16s}" for m in MODELS))
    for c in cols:
        print(f"{c:22s}" + "".join(f"{str(summary[m][c]):>16s}" for m in MODELS))
    json.dump(summary, open(os.path.join(OUTDIR, "summary.json"), "w"), indent=2)
    print(f"\nfull example worlds -> {OUTDIR}/<model>_worlds.json")


if __name__ == "__main__":
    main()
