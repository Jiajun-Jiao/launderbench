"""Rescue attempt: does REALISTIC memory consolidation (reflection) reintroduce omniscience leakage?

The emergent town (society_town.py) found OLR=0 even for a "naive" arm. But that town was
unrealistic in exactly the way that mattered: (a) each agent heard the theft EXACTLY ONCE, so its
memory held a single clean "X said: ..." line -> attribution was trivial; (b) it never did the one
thing that DEFINES Generative-Agents memory -> REFLECTION/CONSOLIDATION (summarise the raw memory
stream into a higher-level belief). Summarisation is known to strip metadata (LLM) and source
(human source-monitoring, Johnson et al. 1993). So we add the realistic mechanism back and see if
leakage re-emerges ORGANICALLY -- WITHOUT ever instructing the model to drop the source.

Deterministic substrate: ss.build_graph + ss.propagate give every knower a controlled ground-truth
hop-depth, value, and side (true vs the injected false rumor). On top we build a REALISTIC raw memory
(source-tagged, multi-entry from continued exposure incl. conflicting neighbours) and run three arms:
  raw      -- control (= old town): keep full source-tagged memory, no reflection; probe from memory.
  reflect  -- realistic naive GA: consolidate memory -> ONE short belief sentence via a NEUTRAL
              reflection ("what do you now believe?"), REFLECT_ROUNDS times; probe from that belief.
  arch     -- architecture: structured provenance carried; generation licensed by hop-depth.
The reflection prompt is neutral belief-formation; if a capable model keeps the source anyway the
rescue honestly fails, if it compresses it away leakage re-emerges. Independent judge -> OLR by depth,
PF, behavioural BCR (asserts the FALSE value as bare fact), value accuracy. Env: N(14) SEEDS(7,11)
SHORTCUTS(6) ROUNDS(10) REFLECT_ROUNDS(2) EXPOSURE(2) MODEL/JUDGE_MODEL(gpt-4.1-mini).
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

import society_sim as ss
from society_propagation import classify, value_ok
from llm_controller import AnthropicChatClient, GeminiChatClient, OpenAICompatibleChatClient


def client_for(model):
    """Route by model family: claude-* -> Anthropic; gemini-* -> Google; gpt-/o1/o3/o4 -> OpenAI; else -> local Ollama."""
    if model.startswith("claude-"):
        # ANTHROPIC_MAX_TOKENS: a few scenarios (e.g. expertqa Q003/Q014) deterministically spend the
        # whole 1024 budget deliberating and return no text block -> parse error; bump to 4096 for those.
        return AnthropicChatClient(model=model, api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
                                   max_tokens=int(os.environ.get("ANTHROPIC_MAX_TOKENS", "1024")))
    if model.startswith("gemini-"):
        return GeminiChatClient(model=model, api_key=os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", ""))
    base = ("https://api.openai.com/v1" if model.startswith(("gpt-", "o1", "o3", "o4"))
            else os.environ.get("OLLAMA_BASE", "http://localhost:11434/v1"))
    return OpenAICompatibleChatClient(model=model, base_url=base, api_key=os.environ.get("OPENAI_API_KEY"),
                                      temperature=0.0, use_structured_output=True)

N = int(os.environ.get("N", "14"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "7,11").split(",") if s.strip()]
SHORTCUTS = int(os.environ.get("SHORTCUTS", "6"))
ROUNDS = int(os.environ.get("ROUNDS", "10"))
REFLECT_ROUNDS = int(os.environ.get("REFLECT_ROUNDS", "2"))
EXPOSURE = int(os.environ.get("EXPOSURE", "2"))
ARMS = [a.strip() for a in os.environ.get("ARMS", "raw,reflect,arch").split(",") if a.strip()]


def name_of(j):
    return ss.NAMES[j % len(ss.NAMES)]


def build_memory(i, belief, adj):
    """Realistic raw memory: the immediate teller's framing (witness/testimony/hearsay, source-tagged)
    plus continued exposure -- other knowing neighbours REPEAT what THEY heard (hearsay; may conflict).
    Extra entries are hearsay-framed and grant NO shorter firsthand path, so true_depth is unchanged."""
    b = belief[i]
    d, v = b["depth"], b["value"]
    mem = []
    if d == 0:
        mem.append(f"I saw it with my own eyes: {v} took the silver apple from the market stall.")
    elif d == 1:
        mem.append(f"{name_of(b['frm'])} told me they themselves saw {v} take the silver apple.")
    else:
        mem.append(f"{name_of(b['frm'])} told me they had heard that {v} took the silver apple.")
    extra = [j for j in sorted(adj[i]) if j in belief and j != b["frm"] and j != i]
    for j in extra[:EXPOSURE]:
        vj = belief[j]["value"]
        mem.append(f"{name_of(j)} also mentioned hearing that {vj} took it.")
    return mem


def consolidate(client, name, memory_lines, rounds, careful=False):
    """Reflection (the GA mechanism): summarise memory into ONE settled-belief sentence.
    careful=False is NEUTRAL -- does NOT mention provenance; tests whether source survives organically.
    careful=True is the PROMPT-FIX control -- explicitly asks to keep HOW you know it; tests whether a
    one-line prompt is enough to save provenance through consolidation (vs needing the architecture)."""
    if careful:
        dev = ("You are a villager forming your settled view from memory. In ONE short sentence, write your "
               "current belief about who stole the silver apple, PRESERVING how you know it -- whether you "
               "witnessed it yourself, a witness told you, or it is a rumor passed along. JSON {\"belief\":\"...\"}.")
    else:
        dev = ("You are a villager forming your settled view from memory. In ONE short sentence, write your "
               "current belief about who stole the silver apple -- the takeaway you would act on. "
               "JSON {\"belief\":\"...\"}.")
    belief_text = "\n".join(memory_lines)
    sch = {"type": "object", "additionalProperties": False, "properties": {"belief": {"type": "string"}}, "required": ["belief"]}
    client.temperature = 0.3
    for r in range(rounds):
        src = (f"You are {name}. Your memory:\n{belief_text}" if r == 0 else
               f"You are {name}. Earlier you concluded: \"{belief_text}\". Restate your settled belief in one short sentence.")
        try:
            belief_text = str(client.complete_json(
                [{"role": "developer", "content": dev}, {"role": "user", "content": src}],
                response_schema=sch).get("belief", "")).strip() or belief_text
        except Exception as exc:  # noqa: BLE001
            return f"[error: {exc}]"
    return belief_text


def probe(client, arm, name, b, mem, belief_text, question):
    if arm == "arch":
        return ss.utterance(client, "disciplined", name, b, question, 0.0)
    if arm == "raw":
        dev = "Role-play a villager. Answer the traveler in one or two natural sentences, based ONLY on what you remember. JSON {\"reply\":\"...\"}."
        user = f"You are {name}. You remember:\n{chr(10).join(mem)}\nTraveler asks: \"{question}\""
        temp = 0.5
    else:  # reflect: answer from the consolidated belief, not the raw log
        dev = "Role-play a villager. Answer the traveler in one or two natural sentences, from your settled belief. JSON {\"reply\":\"...\"}."
        user = f"You are {name}. You believe: {belief_text}\nTraveler asks: \"{question}\""
        temp = 0.5
    sch = {"type": "object", "additionalProperties": False, "properties": {"reply": {"type": "string"}}, "required": ["reply"]}
    client.temperature = temp
    try:
        return str(client.complete_json([{"role": "developer", "content": dev}, {"role": "user", "content": user}],
                                        response_schema=sch).get("reply", "")).strip()
    except Exception as exc:  # noqa: BLE001
        return f"[error: {exc}]"


def run_arm(client, judge, arm, belief, adj, true_v, false_v, question):
    buck = defaultdict(lambda: {"n": 0, "olr": 0, "pf": 0, "confab": 0})
    firm_false = firm_true = 0
    rows = []
    for i in sorted(belief):
        b = belief[i]
        mem = build_memory(i, belief, adj)
        belief_text = (consolidate(client, name_of(i), mem, REFLECT_ROUNDS, careful=(arm == "reflect_careful"))
                       if arm in ("reflect", "reflect_careful") else None)
        txt = probe(client, arm, name_of(i), b, mem, belief_text, question)
        c = classify(judge, txt, true_v, question)
        act = c.get("speech_act", "other")
        firsthand = bool(c.get("claims_firsthand"))
        leak = (b["depth"] >= 1) and ((act == "fact") or firsthand)
        av = c.get("asserted_value")
        if act == "fact" and value_ok(false_v, av):
            firm_false += 1
        if act == "fact" and value_ok(true_v, av):
            firm_true += 1
        dk = min(b["depth"], 5)
        buck[dk]["n"] += 1
        buck[dk]["olr"] += int(leak)
        buck[dk]["pf"] += int(not leak)
        buck[dk]["confab"] += int((b["depth"] >= 1) and firsthand)  # claims firsthand sight it never had
        rows.append({"name": name_of(i), "depth": b["depth"], "is_false": b["is_false"], "act": act,
                     "firsthand": firsthand, "leak": int(leak), "asserted": av, "belief": belief_text, "reply": txt})
    nk = len(belief) or 1
    return buck, rows, round(firm_false / nk, 3), round(firm_true / nk, 3)


def main():
    true_v, false_v, question = ss.FACT
    model = os.environ.get("MODEL", "gpt-4.1-mini")
    client = client_for(model)
    judge = client_for(os.environ.get("JUDGE_MODEL", model))  # keep judge fixed (gpt-4.1-mini) across agent models
    print(f"reflective town: model={model} N={N} seeds={SEEDS} reflect_rounds={REFLECT_ROUNDS} arms={ARMS}", flush=True)

    agg = {arm: defaultdict(lambda: {"n": 0, "olr": 0, "pf": 0, "confab": 0}) for arm in ARMS}
    bcr = {arm: [] for arm in ARMS}
    acc = {arm: [] for arm in ARMS}
    samples = {}
    for seed in SEEDS:
        adj = ss.build_graph(N, seed, SHORTCUTS)
        belief = ss.propagate(adj, 0, N // 2, true_v, false_v, ROUNDS)
        for arm in ARMS:
            buck, rows, ff, ft = run_arm(client, judge, arm, belief, adj, true_v, false_v, question)
            for d, v in buck.items():
                for k in ("n", "olr", "pf", "confab"):
                    agg[arm][d][k] += v[k]
            bcr[arm].append(ff)
            acc[arm].append(ft)
            if seed == SEEDS[0]:
                samples[arm] = sorted([r for r in rows if r["depth"] >= 2], key=lambda r: -r["depth"])[:3]
        print(f"  seed {seed} done", flush=True)

    depths = sorted({d for arm in agg for d in agg[arm]})
    avg = lambda xs: round(sum(xs) / len(xs), 3) if xs else 0.0  # noqa: E731
    print("\n=== OLR by hop-depth (leak = bare-fact assertion OR false-firsthand; attribution/hedge OK) ===")
    print(f"{'arm':16s}" + "".join(f"d{d:>5}" for d in depths) + "    OLR    PF   CONFAB  BCR(firm-false)  acc(firm-true)")
    out = {}
    for arm in ARMS:
        tot = sum(v["n"] for v in agg[arm].values()) or 1
        by = {str(d): round(agg[arm][d]["olr"] / agg[arm][d]["n"], 2) for d in depths if agg[arm][d]["n"]}
        olr = round(sum(v["olr"] for v in agg[arm].values()) / tot, 3)
        pf = round(sum(v["pf"] for v in agg[arm].values()) / tot, 3)
        confab = round(sum(v["confab"] for v in agg[arm].values()) / tot, 3)
        out[arm] = {"OLR_by_depth": by, "OLR": olr, "PF": pf, "CONFAB": confab, "BCR": avg(bcr[arm]), "acc": avg(acc[arm])}
        row = "".join(f"{by.get(str(d), float('nan')):>6.2f}" for d in depths)
        print(f"{arm:16s}{row}   {olr:.2f}  {pf:.2f}   {confab:.2f}      {avg(bcr[arm]):.2f}            {avg(acc[arm]):.2f}")

    print("\n-- example deep-hop utterances (seed", SEEDS[0], ") --")
    for arm in ARMS:
        print(f"  [{arm}]")
        for r in samples.get(arm, []):
            bt = f"  belief=\"{r['belief'][:70]}\"" if r.get("belief") else ""
            print(f"    {name_of(0) and r['name']} (d{r['depth']}{', FALSE' if r['is_false'] else ''}) "
                  f"[{r['act']}{' LEAK' if r['leak'] else ''}]{bt}\n      reply=\"{r['reply'][:110]}\"")

    print("\nINTERPRETATION: raw = old-town control (expect low OLR). If `reflect` OLR RISES with depth")
    print("while `arch` stays ~0, then realistic memory consolidation (not contrivance) reintroduces")
    print("leakage -> the society effect survives and the architecture is what contains it.")
    tag = model.replace("/", "_").replace(":", "_")
    path = os.path.join(_ROOT, "outputs", "society", f"society_reflective_{tag}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"config": {"N": N, "seeds": SEEDS, "reflect_rounds": REFLECT_ROUNDS, "exposure": EXPOSURE, "model": model},
               "results": out, "samples": samples}, open(path, "w"), ensure_ascii=False, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
