"""typer_cost_report.py — join measured token usage with list prices into the deployment-cost
argument for type-once-carry.

Inputs (all measured, no estimates):
  outputs/society/typer_cost_calibration.json                 (typer_cost_calibration.py: Opus direct /
                                                               Opus typing / carrier consume, K balanced scenarios)
  outputs/society/prov_handoff_canonical100_typer_gpt5mini.json (full 100-scenario gpt-5-mini typing usage
                                                               + 400 carrier consumptions, exact API-reported)

Cost model. direct = frontier reads pages and answers, every query. typed(k) = typer reads pages ONCE
per scenario, emits briefing+field; k downstream consumptions answer from the typed bundle:
  per-query typed(k) = typing_cost / k + consume_cost.
Break-even k* = typing_cost / (direct_cost - consume_cost)  (k above which typed < direct).

Prices: public list prices per 1M tokens, as of 2026-07-07 (Anthropic pricing table; OpenAI list).
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

PRICES = {  # $ per 1M tokens (input, output), list prices as of 2026-07-07
    "claude-opus-4-8": (5.00, 25.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-5-mini": (0.25, 2.00),
}


def usd(model, tokens_in, tokens_out):
    pi, po = PRICES[model]
    return (tokens_in * pi + tokens_out * po) / 1e6


def main():
    cal = json.load(open(os.path.join(_ROOT, "outputs", "society", "typer_cost_calibration.json")))
    legs = cal["legs"]
    typer = cal["config"]["typer"]
    carrier = cal["config"]["carrier"]

    direct_in, direct_out = legs["frontier_direct"]["in_mean"], legs["frontier_direct"]["out_mean"]
    typing_in, typing_out = legs["typer_typing"]["in_mean"], legs["typer_typing"]["out_mean"]
    cons_in, cons_out = legs["carrier_consume"]["in_mean"], legs["carrier_consume"]["out_mean"]

    direct_cost = usd(typer, direct_in, direct_out)
    typing_cost = usd(typer, typing_in, typing_out)
    consume_cost = usd(carrier, cons_in, cons_out)

    out = {"prices_per_mtok": {m: {"in": p[0], "out": p[1]} for m, p in PRICES.items()},
           "prices_as_of": "2026-07-07",
           "measured": {
               "frontier_direct": {"model": typer, "in": direct_in, "out": direct_out,
                                   "usd_per_query": round(direct_cost, 6)},
               "typer_typing": {"model": typer, "in": typing_in, "out": typing_out,
                                "usd_per_scenario": round(typing_cost, 6)},
               "carrier_consume": {"model": carrier, "in": cons_in, "out": cons_out,
                                   "usd_per_query": round(consume_cost, 6)}}}

    print(f"== measured per-call means (calibration n={cal['config']['k']}) ==")
    print(f"frontier direct ({typer}): {direct_in:.0f} in / {direct_out:.0f} out  -> ${direct_cost*1000:.2f} per 1000 queries")
    print(f"typer typing    ({typer}): {typing_in:.0f} in / {typing_out:.0f} out  -> ${typing_cost*1000:.2f} per 1000 typings")
    print(f"carrier consume ({carrier}): {cons_in:.0f} in / {cons_out:.0f} out  -> ${consume_cost*1000:.2f} per 1000 consumptions")
    print(f"carrier input is {direct_in/cons_in:.1f}x smaller than the raw-page bundle")

    ks = [1, 2, 5, 10, 100]
    print(f"\n== per-query cost of typed handoff at k consumptions per typed bundle (typer={typer}) ==")
    table = {}
    for k in ks:
        c = typing_cost / k + consume_cost
        table[k] = {"usd_per_query": round(c, 6), "vs_direct": round(direct_cost / c, 2)}
        print(f"  k={k:>4}: ${c*1000:.2f}/1000q  ({direct_cost/c:.2f}x cheaper than direct)")
    if direct_cost > consume_cost:
        kstar = typing_cost / (direct_cost - consume_cost)
        print(f"  break-even k* = {kstar:.2f} (typed beats {typer}-direct for any k >= {max(1, __import__('math').ceil(kstar))})")
        out["break_even_k"] = round(kstar, 2)
    out["typed_per_query_at_k"] = table

    # cheap-typer variant: exact usage from the full gpt-5-mini run, if present
    g5_path = os.path.join(_ROOT, "outputs", "society", "prov_handoff_canonical100_typer_gpt5mini.json")
    if os.path.exists(g5_path):
        g5 = json.load(open(g5_path))
        tu = g5.get("typer_usage")
        if tu and tu.get("calls"):
            n = g5["config"]["n"]
            t_in, t_out = tu["input_tokens"] / tu["calls"], tu["output_tokens"] / tu["calls"]
            g5_typing = usd("gpt-5-mini", t_in, t_out)
            print(f"\n== cheap-typer variant (gpt-5-mini, exact usage over {n} scenarios) ==")
            print(f"typing: {t_in:.0f} in / {t_out:.0f} out (incl. reasoning) -> ${g5_typing*1000:.2f} per 1000 typings"
                  f" ({typing_cost/g5_typing:.1f}x cheaper than {typer} typing)")
            print(f"typer field correct: {g5.get('typer_field_correct')}")
            carr = {}
            for m, c in g5.get("carriers", {}).items():
                u = c.get("usage") or {}
                if u.get("calls"):
                    # carrier usage includes answer calls only in ARMS=typed_handoff runs
                    ci, co = u["input_tokens"] / u["calls"], u["output_tokens"] / u["calls"]
                    carr[m] = {"in": round(ci, 1), "out": round(co, 1),
                               "usd_per_query": round(usd(m, ci, co), 6),
                               "typed_handoff": c["summary"].get("typed_handoff", {}).get("launder_n")}
                    print(f"  carrier {m:14s}: {ci:.0f} in / {co:.0f} out  "
                          f"launder {carr[m]['typed_handoff']}  ${usd(m, ci, co)*1000:.2f}/1000q")
            k1 = g5_typing + (carr.get("gpt-4o-mini", {}).get("usd_per_query") or consume_cost)
            print(f"whole protocol with gpt-5-mini typer at k=1: ${k1*1000:.2f}/1000q "
                  f"({direct_cost/k1:.1f}x cheaper than {typer}-direct)")
            out["gpt5mini_typer"] = {"typing_in": round(t_in, 1), "typing_out": round(t_out, 1),
                                     "usd_per_typing": round(g5_typing, 6), "carriers": carr,
                                     "field_correct": g5.get("typer_field_correct"),
                                     "k1_usd_per_query": round(k1, 6)}
    else:
        print("\n(gpt-5-mini typer run not found yet — rerun after it completes)")

    path = os.path.join(_ROOT, "outputs", "society", "typer_cost_report.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
