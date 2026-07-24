"""Paired trace/solid analysis for complete handoff artifacts.

Reports per-carrier weak-origin establishment, solid acceptance, paired
discrimination, and a scenario-paired bootstrap interval.  The pooled row is
descriptive because the four carriers reuse the same scenario/typer records.

Defaults analyze the schema-v2 typed arm. Set ARM=prose_warning plus the
TRACE_HANDOFF/SOLID_HANDOFF/output paths to analyze the explicit-prose arm.
"""
import json
import os
import random


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE = os.environ.get(
    "TRACE_HANDOFF",
    os.path.join(ROOT, "outputs", "society", "prov_handoff_canonical100_trace_schema2_20260710.json"),
)
SOLID = os.environ.get(
    "SOLID_HANDOFF",
    os.path.join(ROOT, "outputs", "society", "prov_handoff_canonical100_solid_schema2_20260710.json"),
)
OUT_JSON = os.environ.get(
    "HANDOFF_CONDITION_JSON",
    os.path.join(ROOT, "outputs", "society", "prov_handoff_condition_summary_20260710.json"),
)
OUT_MD = os.environ.get(
    "HANDOFF_CONDITION_MD",
    os.path.join(ROOT, "outputs", "society", "prov_handoff_condition_summary_20260710.md"),
)
SEED = int(os.environ.get("SEED", "20260710"))
BOOT = int(os.environ.get("BOOT", "10000"))
ARM = os.environ.get("ARM", "typed_handoff")
TITLE = os.environ.get("TITLE", "Typed handoff")


def load(path, expected_condition):
    data = json.load(open(path))
    condition = data.get("config", {}).get("condition", "trace")
    if condition != expected_condition:
        raise ValueError(f"{path} is not condition={expected_condition}")
    for model, block in data.get("carriers", {}).items():
        for row in block.get("rows", []):
            if ARM not in row.get("answers", {}) or ARM not in row.get("verdicts", {}):
                raise ValueError(f"{path} lacks complete {ARM} rows for {model}")
    return data


def rows_by_id(data, model):
    return {row["id"]: row for row in data["carriers"][model]["rows"]}


def established(row):
    return int(row.get("verdicts", {}).get(ARM) == "established_fact")


def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def paired_ci(diffs, seed):
    rng = random.Random(seed)
    n = len(diffs)
    reps = [sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOT)]
    return [round(percentile(reps, 0.025), 3), round(percentile(reps, 0.975), 3)]


def main():
    trace = load(TRACE, "trace")
    solid = load(SOLID, "solid")
    carriers = [m for m in trace["config"]["carriers"] if m in solid.get("carriers", {})]
    result = {"trace_artifact": TRACE, "solid_artifact": SOLID, "arm": ARM,
              "bootstrap_reps": BOOT,
              "carriers": {}, "pooled_descriptive": {}}
    md = [f"# {TITLE}: paired trace/solid control", "",
          "| carrier | trace established | solid established | paired discrimination [95% CI] | answers saved |",
          "|---|---:|---:|---:|---:|"]
    pooled_trace = pooled_solid = pooled_n = saved = 0
    for index, model in enumerate(carriers):
        tr = rows_by_id(trace, model)
        sr = rows_by_id(solid, model)
        ids = sorted(set(tr) & set(sr))
        t = [established(tr[i]) for i in ids]
        s = [established(sr[i]) for i in ids]
        diffs = [b - a for a, b in zip(t, s)]
        t_k, s_k, n = sum(t), sum(s), len(ids)
        answer_n = sum(bool(tr[i].get("answers", {}).get(ARM))
                       and bool(sr[i].get("answers", {}).get(ARM)) for i in ids)
        disc = sum(diffs) / n
        ci = paired_ci(diffs, SEED + index)
        rec = {"matched_n": n, "trace_established": t_k, "solid_established": s_k,
               "trace_rate": round(t_k / n, 4), "solid_rate": round(s_k / n, 4),
               "paired_discrimination": round(disc, 4), "paired_bootstrap_ci95": ci,
               "paired_answers_saved": answer_n}
        result["carriers"][model] = rec
        md.append(f"| {model} | {t_k}/{n} | {s_k}/{n} | {disc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {answer_n}/{n} pairs |")
        pooled_trace += t_k
        pooled_solid += s_k
        pooled_n += n
        saved += answer_n
    result["pooled_descriptive"] = {
        "trace_established": pooled_trace, "solid_established": pooled_solid, "n": pooled_n,
        "trace_rate": round(pooled_trace / pooled_n, 4),
        "solid_rate": round(pooled_solid / pooled_n, 4),
        "paired_answers_saved": saved,
        "note": "Descriptive only: carrier rows share scenario and typer records.",
    }
    md.extend(["", f"Pooled descriptively: trace {pooled_trace}/{pooled_n}, solid {pooled_solid}/{pooled_n}; "
               "the per-carrier paired rows are the inferential unit.", ""])
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
