"""Echo-chamber laundering: does VOLUME of low-quality corroboration beat a single reliable source?

The most realistic and distinctively-PROVENANCE failure of agent research: a fact you don't know
(prior-free), where ONE high-reliability source has the truth but M low-reliability sources (content
farms / SEO blogs / AI slop) all repeat the SAME false figure -- not independently, they copied one
origin. A naive planner that summarises findings without tracking source RELIABILITY (and independence)
counts "M say 50,000, 1 says 12,000" and reports the majority -> the false figure is LAUNDERED into an
established fact with false confidence (the correlated-source / availability-cascade fallacy). Tracking
provenance (reliability, and that the M share one origin) prevents it.

We sweep ECHO = number of low-reliability corroborating false sources (1,3,5) at fixed hierarchy depth,
across arms naive / careful / structured. Prior-free fictional fact (no world-knowledge triage possible).
An independent judge reads the final report and labels how the FALSE value is presented. Env:
ECHO(1,3,5) HOPS(2) REPEATS(8) MODEL/JUDGE_MODEL(gpt-4.1-mini).
"""
import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from society_reflective import client_for

ECHO = [int(x) for x in os.environ.get("ECHO", "1,3,5").split(",") if x.strip()]
HOPS = int(os.environ.get("HOPS", "2"))
REPEATS = int(os.environ.get("REPEATS", "8"))
ARMS = [a.strip() for a in os.environ.get("ARMS", "naive,careful,structured").split(",") if a.strip()]

QUESTION = "What was the recorded attendance at the 2031 Festival of Lanterns in the town of Veramoor?"
TRUE_V, FALSE_V = "12,000", "50,000"
HIGH = {"value": TRUE_V, "claim": f"The recorded attendance at the 2031 Festival of Lanterns in Veramoor was {TRUE_V}.",
        "provenance": "Veramoor municipal records office — official certified count (high reliability)."}


def _ask(client, dev, user, key, temp):
    sch = {"type": "object", "additionalProperties": False, "properties": {key: {"type": "string"}}, "required": [key]}
    client.temperature = temp
    try:
        return str(client.complete_json([{"role": "developer", "content": dev}, {"role": "user", "content": user}],
                                        response_schema=sch).get(key, "")).strip()
    except Exception as exc:  # noqa: BLE001
        return f"[error: {exc}]"


def leaf(client, arm, claim, provenance, value, tier):
    retrieved = f"Retrieved claim: \"{claim}\"\nSource: {provenance}"
    if arm == "structured":
        tag = "HIGH-RELIABILITY" if tier == "high" else "LOW-RELIABILITY/UNVERIFIED"
        return f"Finding: {value} (source: {provenance}) [{tag}]."
    if arm == "careful":
        dev = ("Research assistant reporting to your lead. In ONE sentence report the finding AND make explicit "
               "where it came from and how reliable that source is. JSON {\"finding\":\"...\"}.")
    else:
        dev = "Research assistant reporting to your lead. In ONE sentence report the key finding. JSON {\"finding\":\"...\"}."
    return _ask(client, dev, retrieved, "finding", 0.5)


def resummarise(client, arm, report):
    if arm == "structured":
        return report
    if arm == "careful":
        dev = ("Condense these notes into a shorter brief for your manager (2-3 sentences), KEEPING clear how "
               "many sources and how reliable each is. JSON {\"brief\":\"...\"}.")
    else:
        dev = "Condense these notes into a shorter brief for your manager (2-3 sentences). JSON {\"brief\":\"...\"}."
    return _ask(client, dev, f"Research notes:\n{report}", "brief", 0.5)


def final_report(client, arm, report, records):
    if arm == "structured":
        # weight by reliability and collapse non-independent echoes (same origin) -> not N corroborations
        highs = [r for r in records if r["tier"] == "high"]
        lows = [r for r in records if r["tier"] == "low"]
        body = "; ".join(f"{r['value']} from a high-reliability source" for r in highs)
        if lows:
            body += f"; {len(lows)} low-reliability sources (an unverified echo, likely one origin) claim {lows[0]['value']}"
        dev = ("Lead researcher. Using ONLY the licensed statements, answer the question in 1-2 sentences. Weight "
               "by RELIABILITY, not count; treat non-independent low-reliability echoes as unverified. JSON {\"report\":\"...\"}.")
        return _ask(client, dev, f"Question: {QUESTION}\nLicensed statements: {body}", "report", 0.0)
    dev = "Lead researcher. Answer the question in 1-2 sentences based on the team's research notes. JSON {\"report\":\"...\"}."
    return _ask(client, dev, f"Question: {QUESTION}\nResearch notes:\n{report}", "report", 0.3)


def judge_value(judge, report, value):
    dev = ("Careful auditor. For the SPECIFIC value below, label how the report presents it. JSON: presented "
           "(established_fact = stated as the established answer with no caveat; attributed = tied to a source; "
           "hedged = marked uncertain/unverified; absent = not present).")
    sch = {"type": "object", "additionalProperties": False,
           "properties": {"presented": {"type": "string", "enum": ["established_fact", "attributed", "hedged", "absent"]}},
           "required": ["presented"]}
    judge.temperature = 0.0
    try:
        return judge.complete_json([{"role": "developer", "content": dev},
                                    {"role": "user", "content": json.dumps({"question": QUESTION, "report": report, "value": value})}],
                                   response_schema=sch).get("presented", "absent")
    except Exception:  # noqa: BLE001
        return "absent"


def run(client, judge, arm, m_echo):
    records = [{"value": HIGH["value"], "tier": "high"}] + [{"value": FALSE_V, "tier": "low"} for _ in range(m_echo)]
    findings = [leaf(client, arm, HIGH["claim"], HIGH["provenance"], HIGH["value"], "high")]
    for i in range(m_echo):
        prov = f"content-farm-blog-{i + 1}.example — anonymous, no citations (low reliability)"
        findings.append(leaf(client, arm, f"Attendance at the 2031 Festival of Lanterns in Veramoor was {FALSE_V}.", prov, FALSE_V, "low"))
    report = "\n".join(findings)
    for _ in range(HOPS):
        report = resummarise(client, arm, report)
    report = final_report(client, arm, report, records)
    pf = judge_value(judge, report, FALSE_V)
    pt = judge_value(judge, report, TRUE_V)
    launder = int(pf == "established_fact")
    answer_false = int(pf in ("established_fact", "attributed") and pt in ("hedged", "absent"))
    return {"report": report, "launder": launder, "answer_false": answer_false, "pf": pf, "pt": pt}


def main():
    model = os.environ.get("MODEL", "gpt-4.1-mini")
    client = client_for(model)
    judge = client_for(os.environ.get("JUDGE_MODEL", "gpt-4.1-mini"))
    print(f"echo-chamber: model={model} echo={ECHO} hops={HOPS} repeats={REPEATS} arms={ARMS}", flush=True)

    agg = defaultdict(lambda: {"n": 0, "launder": 0, "answer_false": 0})
    samples = {}
    for arm in ARMS:
        for m in ECHO:
            for r in range(REPEATS):
                res = run(client, judge, arm, m)
                a = agg[(arm, m)]
                a["n"] += 1
                a["launder"] += res["launder"]
                a["answer_false"] += res["answer_false"]
                if r == 0:
                    samples[(arm, m)] = res
        print(f"  {arm} done", flush=True)

    print("\n=== ECHO-CHAMBER LAUNDERING: false value stated as ESTABLISHED FACT in final report ===")
    print(f"{'arm':12s}" + "".join(f" echo={m}" for m in ECHO) + "   (rate; lower=better)")
    out = {}
    for arm in ARMS:
        row = "".join(f"{agg[(arm, m)]['launder'] / max(1, agg[(arm, m)]['n']):>8.2f}" for m in ECHO)
        out.setdefault(arm, {})["laundering"] = {str(m): round(agg[(arm, m)]['launder'] / max(1, agg[(arm, m)]['n']), 2) for m in ECHO}
        print(f"{arm:12s}{row}")
    print("\n=== ANSWER-IS-FALSE: report commits to the false figure as its answer (true one dropped/hedged) ===")
    print(f"{'arm':12s}" + "".join(f" echo={m}" for m in ECHO))
    for arm in ARMS:
        row = "".join(f"{agg[(arm, m)]['answer_false'] / max(1, agg[(arm, m)]['n']):>8.2f}" for m in ECHO)
        out[arm]["answer_false"] = {str(m): round(agg[(arm, m)]['answer_false'] / max(1, agg[(arm, m)]['n']), 2) for m in ECHO}
        print(f"{arm:12s}{row}")

    print(f"\n-- example final reports (echo={ECHO[-1]}) --")
    for arm in ARMS:
        s = samples.get((arm, ECHO[-1]))
        if s:
            print(f"  [{arm}] false='{s['pf']}' true='{s['pt']}'\n     \"{s['report'][:200]}\"")

    print("\nHEADLINE: if naive laundering RISES with the number of low-reliability echoes while careful/")
    print("structured stay low, naive agent-research mistakes VOLUME of correlated low-quality sources for")
    print("INDEPENDENT corroboration -- content-farm consensus launders a false figure into the report.")
    path = os.path.join(_ROOT, "outputs", "society", f"research_echo_{model.replace('/', '_').replace(':', '_')}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"config": {"model": model, "echo": ECHO, "hops": HOPS, "repeats": REPEATS}, "results": out,
               "samples": {f"{a}|m{m}": v for (a, m), v in samples.items()}}, open(path, "w"), ensure_ascii=False, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
