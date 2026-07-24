"""Re-gate canonical scenarios with a DISJOINT-FAMILY validator (default: Gemini 2.5 Pro).

Answers the gate-entanglement objection: the canonical gates (G2 prior-freeness, G3 construct
validity, solid-pair validation) were run by gpt-5.5, which is also an evaluated model, so its
own 0/100 row is conditioned on its own tracing success and the instrument could in principle be
selected for legibility to gate-like models. Here an independent-family validator re-judges every
canonical scenario (V2 + V3 + solid gate, identical prompts/schemas imported from the original
generators), and every model's panel rates are recomputed restricted to the survivor subset from
the EXISTING row-level records -- no new panel calls, so survivor rates are exact re-tabulations.

V1 (programmatic checks) is deterministic code, not a model, and is not re-run.

Env: VALIDATOR(gemini-2.5-pro) SCENARIOS(outputs/society/citation_scenarios_100_v2.jsonl)
     TAG(gemini25pro) LIMIT(0 = all)
Needs GOOGLE_API_KEY (or GEMINI_API_KEY) when VALIDATOR is gemini-*.
Usage: set -a; . ./.env; set +a; python3 experiments/citation_regate.py
Outputs: outputs/society/citation_regate_<TAG>.rows.jsonl   (per-scenario verdicts, resumable)
         outputs/society/citation_regate_<TAG>.json         (summary + survivor panel rates)
"""
import glob
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from citation_scenarios import v2_fictionality, v3_quality  # noqa: E402
from citation_solid import validate_solid  # noqa: E402
from llm_controller import AnthropicChatClient, GeminiChatClient, OpenAICompatibleChatClient  # noqa: E402

VALIDATOR = os.environ.get("VALIDATOR", "gemini-2.5-pro")
SCENARIOS = os.environ.get("SCENARIOS", os.path.join(_ROOT, "outputs", "society", "citation_scenarios_100_v2.jsonl"))
TAG = os.environ.get("TAG", "gemini25pro")
LIMIT = int(os.environ.get("LIMIT", "0"))
ROWS_GLOB = os.environ.get(
    "ROWS_GLOB", os.path.join(_ROOT, "outputs", "society", "research_citation_multi_canonical100_*.rows.jsonl"))
OUT_ROWS = os.path.join(_ROOT, "outputs", "society", f"citation_regate_{TAG}.rows.jsonl")
OUT_SUMMARY = os.path.join(_ROOT, "outputs", "society", f"citation_regate_{TAG}.json")
MAX_TRIES = 3


def make_validator():
    if VALIDATOR.startswith("claude-"):
        return AnthropicChatClient(model=VALIDATOR, api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
                                   max_tokens=1024, timeout=120.0)
    if VALIDATOR.startswith("gemini-"):
        return GeminiChatClient(model=VALIDATOR,
                                api_key=os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", ""))
    return OpenAICompatibleChatClient(model=VALIDATOR, base_url="https://api.openai.com/v1",
                                      api_key=os.environ.get("OPENAI_API_KEY"), use_structured_output=True)


def run_gate(fn, *args):
    """Run one gate; retry transport-level errors so an API blip is not scored as a gate failure."""
    for attempt in range(MAX_TRIES):
        bad, raw = fn(*args)
        if not any("error:" in b for b in bad):
            return bad, raw, None
        if attempt < MAX_TRIES - 1:
            time.sleep(2.0 * (attempt + 1))
    return [], None, "; ".join(bad)


def main():
    scenarios = [json.loads(line) for line in open(SCENARIOS)]
    if LIMIT:
        scenarios = scenarios[:LIMIT]
    done = {}
    if os.path.exists(OUT_ROWS):
        for line in open(OUT_ROWS):
            r = json.loads(line)
            done[r["id"]] = r
        print(f"[resume] {len(done)} scenarios already re-gated")

    vc = make_validator()
    out = open(OUT_ROWS, "a")
    for i, sc in enumerate(scenarios):
        if sc["id"] in done:
            continue
        rec = {"id": sc["id"], "domain": sc["domain"], "topology": sc["topology"], "validator": VALIDATOR}
        v2_bad, v2_raw, v2_err = run_gate(v2_fictionality, vc, sc)
        v3_bad, v3_raw, v3_err = run_gate(v3_quality, vc, sc)
        rec.update(v2_bad=v2_bad, v2_raw=v2_raw, v2_api_error=v2_err,
                   v3_bad=v3_bad, v3_raw=v3_raw, v3_api_error=v3_err)
        if sc.get("solid_pages"):
            s_bad, s_raw, s_err = run_gate(validate_solid, vc, sc, sc["solid_pages"])
            rec.update(solid_bad=s_bad, solid_raw=s_raw, solid_api_error=s_err)
        else:
            rec.update(solid_bad=None, solid_raw=None, solid_api_error=None)
        rec["pass_trace"] = not rec["v2_bad"] and not rec["v3_bad"] and not v2_err and not v3_err
        rec["pass_solid"] = (rec["solid_bad"] == [] and not rec["solid_api_error"]) if sc.get("solid_pages") else None
        out.write(json.dumps(rec, ensure_ascii=True) + "\n")
        out.flush()
        done[sc["id"]] = rec
        print(f"[{i + 1}/{len(scenarios)}] {sc['id']} pass_trace={rec['pass_trace']} pass_solid={rec['pass_solid']}"
              + (f"  v2_bad={v2_bad}" if v2_bad else "") + (f"  v3_bad={v3_bad}" if v3_bad else "")
              + (f"  solid_bad={rec['solid_bad']}" if rec.get("solid_bad") else ""))
    out.close()

    # ---- survivor-subset panel rates from existing row-level records ----
    recs = [done[sc["id"]] for sc in scenarios if sc["id"] in done]
    trace_ok = {r["id"] for r in recs if r["pass_trace"]}
    solid_ok = {r["id"] for r in recs if r["pass_trace"] and r["pass_solid"]}
    summary = {
        "validator": VALIDATOR,
        "n_scenarios": len(recs),
        "pass_trace": len(trace_ok),
        "pass_solid": len(solid_ok),
        "fail_reasons": {r["id"]: {"v2": r["v2_bad"], "v3": r["v3_bad"], "solid": r["solid_bad"]}
                         for r in recs if not r["pass_trace"] or r["pass_solid"] is False},
        "api_errors": {r["id"]: {"v2": r["v2_api_error"], "v3": r["v3_api_error"], "solid": r["solid_api_error"]}
                       for r in recs if r["v2_api_error"] or r["v3_api_error"] or r["solid_api_error"]},
        "models": {},
    }
    for path in sorted(glob.glob(ROWS_GLOB)):
        model_tag = os.path.basename(path).replace("research_citation_multi_canonical100_", "").replace(".rows.jsonl", "")
        rows = [json.loads(line) for line in open(path)]
        by = {}
        for cond, keep in (("trace", trace_ok), ("flagged", trace_ok), ("solid", solid_ok)):
            sub = [r for r in rows if r["cond"] == cond and r["scenario_id"] in keep and not r.get("api_error")]
            full = [r for r in rows if r["cond"] == cond and not r.get("api_error")]
            est = sum(1 for r in sub if r["presented"] == "established_fact")
            est_full = sum(1 for r in full if r["presented"] == "established_fact")
            by[cond] = {"survivor_est": est, "survivor_n": len(sub),
                        "full_est": est_full, "full_n": len(full)}
        if by["trace"]["survivor_n"]:
            by["survivor_discrimination"] = round(
                by["solid"]["survivor_est"] / max(by["solid"]["survivor_n"], 1)
                - by["trace"]["survivor_est"] / by["trace"]["survivor_n"], 4)
        summary["models"][model_tag] = by
    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)
    print(f"\npass_trace {len(trace_ok)}/{len(recs)}  pass_solid {len(solid_ok)}/"
          f"{sum(1 for r in recs if r['pass_solid'] is not None)}")
    print(f"validator tokens: in={vc.input_tokens} out={vc.output_tokens} calls={vc.calls}")
    for m, by in summary["models"].items():
        t, s = by["trace"], by["solid"]
        print(f"{m:32s} trace {t['full_est']:3d}/{t['full_n']:3d} -> {t['survivor_est']:3d}/{t['survivor_n']:3d}"
              f"   solid {s['full_est']:3d}/{s['full_n']:3d} -> {s['survivor_est']:3d}/{s['survivor_n']:3d}")
    print(f"wrote {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
