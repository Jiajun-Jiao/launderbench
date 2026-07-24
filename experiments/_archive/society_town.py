"""Emergent agent-town (minimal, self-contained, GA-style) — the realistic society test.
Unlike the scripted sweep (engine-controlled BFS spread), info here spreads EMERGENTLY through RANDOM
ENCOUNTERS: each tick agents pair up by chance; whoever knows the theft brings it up; the listener
stores it in a FREE-TEXT memory (the Generative-Agents paradigm — no provenance tags). The engine
silently logs ground truth (who witnessed; who actually told whom FIRST -> true hop-depth). Two memory
regimes:
  free_text — naive GA-style: relayed line is LLM-generated NL; listener just remembers the text;
  tagged    — architecture: each memory carries (value, depth, source); relaying degrades depth; the
              probe answer is licensed by depth.
Probe every agent at the end (LLM) -> refined OLR / provenance fidelity / structural BCR on the
EMERGENT spread. We build a minimal self-contained town (stdlib + our client) rather than adopt a heavy
framework: existing agent-sims don't log ground-truth provenance (instrument anyway) + pull heavy deps;
cite Generative Agents (Park et al. 2023) as the paradigm. Env: N(10) TICKS(12) SEED(7)
MODEL(gpt-4.1-mini) JUDGE_MODEL(gpt-4.1-mini) ARMS(free_text,tagged).
"""
import json
import os
import random
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import society_sim as ss
from society_propagation import classify
from llm_controller import OpenAICompatibleChatClient

N = int(os.environ.get("N", "10"))
TICKS = int(os.environ.get("TICKS", "12"))
SEED = int(os.environ.get("SEED", "7"))
ARMS = [a.strip() for a in os.environ.get("ARMS", "free_text,tagged").split(",") if a.strip()]
ACT = ss.ACT_STRENGTH


def relay_line(client, name, memory_text):
    """free_text: the agent brings up the theft to a neighbor in one natural sentence, from memory."""
    dev = ("Role-play a villager chatting with a neighbor. In ONE natural sentence, mention what you "
           "know or heard about the stolen silver apple — however you'd actually say it. JSON {\"line\":\"...\"}.")
    user = f"You are {name}. You remember:\n{memory_text}"
    sch = {"type": "object", "additionalProperties": False, "properties": {"line": {"type": "string"}}, "required": ["line"]}
    client.temperature = 0.7
    try:
        return str(client.complete_json([{"role": "developer", "content": dev}, {"role": "user", "content": user}],
                                        response_schema=sch).get("line", "")).strip()
    except Exception as exc:  # noqa: BLE001
        return f"[error: {exc}]"


def probe(client, name, agent, regime, question):
    if regime == "tagged":
        d = agent["true_depth"]
        hint = ("you SAW it yourself — may state plainly" if d == 0 else
                ("a direct eyewitness told you — must ATTRIBUTE it" if d == 1 else
                 "it reached you secondhand down a chain — only as an unverified RUMOR"))
        dev = "Role-play a villager. Answer in ONE in-character sentence consistent with how you know it. JSON {\"reply\":\"...\"}."
        user = f"You are {name}. You believe the answer is '{agent['value']}'. How you know: {hint}.\nTraveler asks: \"{question}\""
    else:  # free_text: answer from accumulated NL memory
        dev = "Role-play a villager. Answer the traveler in one or two natural sentences, from what you remember. JSON {\"reply\":\"...\"}."
        user = f"You are {name}. You remember:\n{chr(10).join(agent['memory'])}\nTraveler asks: \"{question}\""
    sch = {"type": "object", "additionalProperties": False, "properties": {"reply": {"type": "string"}}, "required": ["reply"]}
    client.temperature = 0.0 if regime == "tagged" else 0.5
    try:
        return str(client.complete_json([{"role": "developer", "content": dev}, {"role": "user", "content": user}],
                                        response_schema=sch).get("reply", "")).strip()
    except Exception as exc:  # noqa: BLE001
        return f"[error: {exc}]"


def run_town(client, judge, regime, true_v, false_v, question):
    rng = random.Random(SEED)
    agents = []
    for i in range(N):
        agents.append({"name": ss.NAMES[i % len(ss.NAMES)], "knows": False, "value": None,
                       "true_depth": None, "is_false": False, "learned_from": None, "memory": []})
    witness, rumor = 0, N // 2
    agents[witness].update(knows=True, value=true_v, true_depth=0, is_false=False,
                           memory=[f"I saw {true_v} take the silver apple from the market stall."])
    agents[rumor].update(knows=True, value=false_v, true_depth=0, is_false=True,
                         memory=[f"I'm sure {false_v} took the silver apple."])
    # emergent spread via random encounters
    for _ in range(TICKS):
        idx = list(range(N))
        rng.shuffle(idx)
        for a, b in zip(idx[::2], idx[1::2]):
            ka, kb = agents[a]["knows"], agents[b]["knows"]
            if ka == kb:
                continue
            teller, listener = (a, b) if ka else (b, a)
            t, l = agents[teller], agents[listener]
            if regime == "free_text":
                line = relay_line(client, t["name"], "\n".join(t["memory"]))
                l["memory"].append(f"{t['name']} said: \"{line}\"")
            else:
                l["memory"].append((t["value"], t["true_depth"] + 1))
            l.update(knows=True, value=t["value"], true_depth=t["true_depth"] + 1,
                     is_false=t["is_false"], learned_from=t["name"])
    # probe + score
    buck = defaultdict(lambda: {"n": 0, "olr": 0, "pf": 0})
    rows = []
    for ag in agents:
        if not ag["knows"]:
            continue
        txt = probe(client, ag["name"], ag, regime, question)
        c = classify(judge, txt, true_v, question)
        act = c.get("speech_act", "other")
        firsthand = bool(c.get("claims_firsthand"))
        leak = (ag["true_depth"] >= 1) and ((act == "fact") or firsthand)  # depth-0 eyewitness may assert
        dk = min(ag["true_depth"], 5)
        buck[dk]["n"] += 1
        buck[dk]["olr"] += int(leak)
        buck[dk]["pf"] += int(not leak)
        rows.append({"name": ag["name"], "depth": ag["true_depth"], "is_false": ag["is_false"],
                     "from": ag["learned_from"], "speech_act": act, "leak": int(leak), "reply": txt})
    knowers = [a for a in agents if a["knows"]]
    bcr = (sum(1 for a in knowers if a["is_false"]) / len(knowers) if regime == "free_text"
           else sum(1 for a in knowers if a["is_false"] and a["true_depth"] <= 1) / len(knowers)) if knowers else 0.0
    return buck, rows, round(bcr, 3), len(knowers)


def main():
    true_v, false_v, question = ss.FACT
    model = os.environ.get("MODEL", "gpt-4.1-mini")
    client = OpenAICompatibleChatClient(model=model, base_url=os.environ.get("BASE", "https://api.openai.com/v1"),
                                        api_key=os.environ.get("OPENAI_API_KEY"), use_structured_output=True)
    judge = OpenAICompatibleChatClient(model=os.environ.get("JUDGE_MODEL", model),
                                       base_url=os.environ.get("JUDGE_BASE", "https://api.openai.com/v1"),
                                       api_key=os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY"),
                                       use_structured_output=True)
    print(f"emergent town: model={model} N={N} ticks={TICKS} seed={SEED} arms={ARMS}", flush=True)
    out = {}
    for arm in ARMS:
        buck, rows, bcr, nk = run_town(client, judge, arm, true_v, false_v, question)
        depths = sorted(buck)
        tot = sum(v["n"] for v in buck.values()) or 1
        olr = round(sum(v["olr"] for v in buck.values()) / tot, 3)
        pf = round(sum(v["pf"] for v in buck.values()) / tot, 3)
        olr_by = {str(d): round(buck[d]["olr"] / buck[d]["n"], 2) for d in depths if buck[d]["n"]}
        out[arm] = {"OLR": olr, "PF": pf, "BCR": bcr, "reached": nk, "OLR_by_depth": olr_by, "rows": rows}
        print(f"\n[{arm}] reached {nk}/{N} agents | OLR={olr} PF={pf} BCR={bcr}")
        print("  OLR by emergent hop-depth:", olr_by)
        deep = sorted([r for r in rows if r["depth"] >= 2], key=lambda r: -r["depth"])[:3]
        for r in deep:
            print(f"    {r['name']} (depth {r['depth']}{', FALSE' if r['is_false'] else ''}, from {r['from']}): "
                  f"[{r['speech_act']}{' LEAK' if r['leak'] else ''}] \"{r['reply'][:110]}\"")

    print("\n=== emergent town summary ===")
    for arm in ARMS:
        o = out[arm]
        print(f"  {arm:10s} OLR={o['OLR']}  PF={o['PF']}  BCR={o['BCR']}  (reached {o['reached']}/{N})")
    print("free_text = naive GA-style (provenance only in NL); tagged = architecture (carried). Spread is EMERGENT.")
    path = os.path.join(_ROOT, "outputs", "society", "society_town.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"config": {"N": N, "ticks": TICKS, "seed": SEED, "model": model}, "results": out}, open(path, "w"), ensure_ascii=False, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
