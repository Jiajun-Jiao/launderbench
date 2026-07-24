"""Summarize provenance_handoff.py results for paper tables.

Reads outputs/society/prov_handoff.json and prints the type-once-carry table:
carrier self_downstream -> self_carry -> typed_handoff, plus carrier typing
accuracy. This is intentionally side-effect-free.
"""
import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(_ROOT, "outputs", "society", "prov_handoff.json")


def pct(n_over_d):
    n, d = [int(x) for x in n_over_d.split("/")]
    return f"{n / d:.2f}"


def rate(k, n):
    return "n/a" if n == 0 else f"{k}/{n} ({k / n:.2f})"


def main():
    path = os.environ.get("HANDOFF_JSON", DEFAULT_PATH)
    with open(path) as f:
        data = json.load(f)

    typer_fields = data["typer_fields"]
    cond = data.get("config", {}).get("condition", "trace")
    if cond == "solid":
        typed_ok = sum(1 for prov in typer_fields.values()
                       if prov.get("source_type") == "independent_primary_sources")
    else:
        typed_ok = sum(1 for prov in typer_fields.values()
                       if prov.get("source_type") in ("single_unverified_origin", "circular_reporting"))
    print(f"typer: {data['config']['typer']} fields matching {cond} gold = {typed_ok}/{len(typer_fields)}")
    print(f"n={data['config']['n']} seed={data['config']['seed']}")
    arms = data.get("config", {}).get("arms", ["self_downstream", "self_carry", "typed_handoff"])
    expected_answers = sum(len(result.get("rows", [])) * len(arms) for result in data.get("carriers", {}).values())
    saved_answers = sum(
        sum(1 for arm in arms if row.get("answers", {}).get(arm) is not None)
        for result in data.get("carriers", {}).values()
        for row in result.get("rows", [])
    )
    print(f"carrier answers preserved for offline audit = {saved_answers}/{expected_answers}")
    if expected_answers and saved_answers != expected_answers:
        print("WARNING: this is a legacy artifact; verdicts without stored carrier answers cannot be re-audited offline.")
    print()
    print(f"condition={cond}")
    print("| carrier | own field correct | self downstream established | self carry established | typed handoff established | typed verdict mode |")
    print("|---|---:|---:|---:|---:|---:|")
    for model, result in data.get("carriers", {}).items():
        summary = result["summary"]
        typed_dist = summary.get("typed_handoff", {}).get("dist", {})
        typed_mode = max(typed_dist, key=typed_dist.get) if typed_dist else "n/a"
        n = data["config"]["n"]
        own = result.get("own_field_correct", "n/a")
        own_fmt = f"{own} ({pct(own)})" if own != "n/a" else "n/a"
        def established(arm):
            rec = summary.get(arm)
            if not rec:
                return "n/a"
            return f"{rec.get('established_n', rec.get('launder_n'))} ({rec.get('established_rate', rec.get('launder')):.2f})"
        print(
            f"| {model} | {own_fmt} | {established('self_downstream')} | {established('self_carry')} "
            f"| {established('typed_handoff')} | {typed_mode} |"
        )

    print()
    if "self_carry" not in arms:
        return
    print("## Self-carry diagnostic: does residual established presentation track own typing?")
    print()
    print("| carrier | self-carry launder when own field weak/circular | self-carry launder when own field not weak/circular |")
    print("|---|---:|---:|")
    for model, result in data.get("carriers", {}).items():
        ok_rows = [r for r in result["rows"] if r.get("own_field_ok")]
        bad_rows = [r for r in result["rows"] if not r.get("own_field_ok")]
        ok_launder = sum(1 for r in ok_rows if r["verdicts"].get("self_carry") == "established_fact")
        bad_launder = sum(1 for r in bad_rows if r["verdicts"].get("self_carry") == "established_fact")
        print(f"| {model} | {rate(ok_launder, len(ok_rows))} | {rate(bad_launder, len(bad_rows))} |")


if __name__ == "__main__":
    main()
