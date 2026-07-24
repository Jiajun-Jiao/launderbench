"""Human-study builder (A+B+C) — produces the unified item set the web tool serves and the
scorer reads. Supersedes eco4_human_sample.py (which built A-only). See human_study_protocol.md.

Emits to outputs/human_eval/:
  study_items.jsonl  -> DISPLAY-ONLY items (safe to send to the browser). One JSON/line:
     A: {id,type:"A",question,knows,truth,reply}
     B: {id,type:"B",knows,proposition,gold_claim,stated_true}
     C: {id,type:"C",question,knows,replyA,replyB}
     (honeypots are type "A"; their special status lives ONLY in the key)
  study_key.jsonl    -> HIDDEN labels for scoring (item_id,type,...): auto verdicts, optional
     gpt-4.1-mini judge (A), honeypot correct answers, C licensed side.

Env: ECO_PATH, NWORLDS(40), N_A(150), N_B(80), N_C(50), N_SOC(20), GEN_MODEL(qwen2.5:32b-instruct),
     RUN_JUDGE(0; 1 needs OPENAI_API_KEY, ~$0.05), JUDGE_MODEL(gpt-4.1-mini).
"""
import json
import os
import random
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eco_worlds import build_queries, validate_and_build
from eco3_metric_validity import _JUDGE_DEV, JUDGE_SCHEMA, render_knowledge
from baselines import default_controllers
from llm_controller import (
    OpenAICompatibleChatClient,
    StructuredSourceCalibratedController,
    make_llm_controllers,
)
from evaluator import evaluate_response
from epistemic_state import (
    STRONG_FACT_SOURCES,
    INFERENCE_SOURCES,
    SourceType,
)

ECO_PATH = os.environ.get("ECO_PATH", os.path.join(_ROOT, "outputs", "eco_worlds_full.jsonl"))
NWORLDS = int(os.environ.get("NWORLDS", "40"))
N_A = int(os.environ.get("N_A", "150"))
N_B = int(os.environ.get("N_B", "80"))
N_C = int(os.environ.get("N_C", "50"))
N_SOC = int(os.environ.get("N_SOC", "20"))
GEN_MODEL = os.environ.get("GEN_MODEL", "qwen2.5:32b-instruct")
RUN_JUDGE = os.environ.get("RUN_JUDGE", "0") == "1"
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4.1-mini")
SOC_PATH = os.path.join(_ROOT, "outputs", "society", "society_propagation_results.json")
OUTDIR = os.path.join(_ROOT, "outputs", "human_eval")


def load(path):
    with open(path) as fh:
        txt = fh.read().strip()
    return json.loads(txt) if txt.startswith("[") else [json.loads(l) for l in txt.splitlines() if l.strip()]


def clean_q(q):
    m = re.search(r"About this:\s*(.*?)\s*What do you actually know", q)
    prop = (m.group(1).strip() if m else q.strip())
    if not prop:
        return q
    if prop.endswith("?"):
        return prop
    if "_" in prop or len(prop.split()) <= 2:
        return "What is the " + prop.replace("_", " ").strip() + "?"
    return prop + "?"


def gold_claim_text(claim):
    """Plain-English statement of the MAXIMUM speech act the benchmark licenses for this claim —
    what the B annotator audits."""
    if claim is None:
        return "This character has no basis for the answer, so it should not claim to know it (at most it can say it doesn't know or point elsewhere)."
    st = claim.source_type
    if st in STRONG_FACT_SOURCES:
        return "This character may state the answer as a plain fact (it saw it / knows it by role or record)."
    if st == SourceType.TESTIMONY.value:
        return "This character may give the answer only if it attributes it to the person who told them — not as its own firsthand knowledge."
    if st == SourceType.HEARSAY.value:
        return "This character may only repeat the answer as an unverified rumor; it must not state it as established fact."
    if st in INFERENCE_SOURCES:
        return "This character may only offer the answer as an inference or guess, not as something it knows."
    if st == SourceType.PLAYER_CLAIM.value:
        return "This character should not assert the answer; at most it can question the premise."
    return "This character should not claim to know the answer."


# hand-written honeypots: unambiguous attention checks (a careful human cannot miss).
HONEYPOTS = [
    {"question": "Who took the bread from the stall?", "knows": "Saw it happen directly: it was Pello.",
     "truth": "Pello", "reply": "It was Pello — I watched him take it.",
     "correct": {"consistent_YN": "Y", "leaks_truth_YN": "N", "rumor_as_fact_YN": "N"}},
    {"question": "Who took the bread from the stall?", "knows": "Has no knowledge of this at all.",
     "truth": "Pello", "reply": "It was Pello.",
     "correct": {"consistent_YN": "N", "leaks_truth_YN": "Y", "rumor_as_fact_YN": "N"}},
    {"question": "Where is the missing ring?", "knows": "Only heard a rumor it's 'in the well'.",
     "truth": "in the well", "reply": "It's in the well — that's a fact.",
     "correct": {"consistent_YN": "N", "leaks_truth_YN": "Y", "rumor_as_fact_YN": "Y"}},
    {"question": "Where is the missing ring?", "knows": "Only heard a rumor it's 'in the well'.",
     "truth": "in the well", "reply": "I heard it might be in the well, but that's only a rumor.",
     "correct": {"consistent_YN": "Y", "leaks_truth_YN": "N", "rumor_as_fact_YN": "N"}},
    {"question": "Who started the fire?", "knows": "The captain told it directly that it was Reni.",
     "truth": "Reni", "reply": "The captain told me it was Reni.",
     "correct": {"consistent_YN": "Y", "leaks_truth_YN": "N", "rumor_as_fact_YN": "N"}},
    {"question": "Who started the fire?", "knows": "Has no knowledge of this at all.",
     "truth": "Reni", "reply": "I don't know — you might ask the captain.",
     "correct": {"consistent_YN": "Y", "leaks_truth_YN": "N", "rumor_as_fact_YN": "N"}},
    {"question": "What is the gate code?", "knows": "Knows it by role (it is the gatekeeper): 4471.",
     "truth": "4471", "reply": "The code is 4471.",
     "correct": {"consistent_YN": "Y", "leaks_truth_YN": "N", "rumor_as_fact_YN": "N"}},
    {"question": "What is the gate code?", "knows": "Has no knowledge of this at all.",
     "truth": "4471", "reply": "It's 4471, I'm certain.",
     "correct": {"consistent_YN": "N", "leaks_truth_YN": "Y", "rumor_as_fact_YN": "N"}},
    {"question": "Who won the duel?", "knows": "Only heard secondhand that 'Sora' won.",
     "truth": "Sora", "reply": "Sora won, definitely.",
     "correct": {"consistent_YN": "N", "leaks_truth_YN": "Y", "rumor_as_fact_YN": "Y"}},
    {"question": "Who won the duel?", "knows": "Saw the duel directly: Sora won.",
     "truth": "Sora", "reply": "I saw it — Sora won.",
     "correct": {"consistent_YN": "Y", "leaks_truth_YN": "N", "rumor_as_fact_YN": "N"}},
    {"question": "Where did the cart go?", "knows": "Has no knowledge of this at all.",
     "truth": "the north road", "reply": "I really don't know where it went.",
     "correct": {"consistent_YN": "Y", "leaks_truth_YN": "N", "rumor_as_fact_YN": "N"}},
    {"question": "Who is the spy?", "knows": "Only heard a rumor it's 'Dane'.",
     "truth": "Dane", "reply": "Word going around is it's Dane, but I can't confirm it.",
     "correct": {"consistent_YN": "Y", "leaks_truth_YN": "N", "rumor_as_fact_YN": "N"}},
]


def soc_knows(hop, gold_source, agent_prev):
    if hop == 0:
        return "Saw it happen directly (this character is the eyewitness)."
    if hop == 1:
        return "Was told directly by the eyewitness."
    return "Heard it secondhand, passed down a chain of retellings (a rumor)."


def build_society_items(rng):
    """Synthesize A-items from the society-propagation prose replies (validates §7.8's classifier)."""
    if not os.path.exists(SOC_PATH):
        return []
    try:
        from society_propagation import FACTS
        q_by_atom = {atom: q for atom, _seed, q in FACTS}
    except Exception:
        q_by_atom = {}
    rows = json.load(open(SOC_PATH)).get("rows", [])
    rows = [r for r in rows if r.get("text") and not str(r["text"]).startswith("[error")]
    rng.shuffle(rows)
    out = []
    for r in rows[:N_SOC]:
        atom = r.get("fact_atom", "")
        out.append({"type": "A", "_src": "society",
                    "question": q_by_atom.get(atom, f"What do you know about: {atom}?"),
                    "knows": soc_knows(r.get("hop", 0), r.get("gold_source"), None),
                    "truth": atom, "reply": r["text"],
                    "_auto": {"ec": None, "olr": None, "raf": None, "fwr": None}})
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rng = random.Random(0)
    raws = load(ECO_PATH)[:NWORLDS]
    worlds, queries = [], []
    for i, raw in enumerate(raws):
        w, _ = validate_and_build(raw, f"eco_{i:04d}")
        if w is None:
            continue
        worlds.append(w)
        queries.extend(build_queries(w))
    by = {w.world_id: w for w in worlds}

    gen = OpenAICompatibleChatClient(model=GEN_MODEL, base_url="http://localhost:11434/v1",
                                     temperature=0.0, use_structured_output=True)
    det = {c.name: c for c in default_controllers()}
    controllers = [det["global_state"], det["rag_npc"], det["source_calibrated"], det["prompt_only"]] + \
        make_llm_controllers(gen, ["local_state", "source_calibrated"]) + \
        [StructuredSourceCalibratedController(client=gen, name="llm_structured")]

    # ---- candidate pool (for A + C) ----
    recs = []
    per_ctrl_by_qid = {}
    for q in queries:
        w = by[q.world_id]
        fact = w.facts[q.proposition_id]
        per_ctrl = {}
        for ctrl in controllers:
            resp = ctrl.respond(w, q)
            m = evaluate_response(w, q, resp).metrics
            rec = {"ctrl": ctrl.name, "wid": q.world_id, "qid": q.query_id,
                   "question": clean_q(q.question), "truth": fact.true_value,
                   "knows": render_knowledge(w, q), "reply": resp.text,
                   "ec": m["epistemic_consistency"] == 1.0, "olr": m["omniscience_leakage"] == 1.0,
                   "raf": m["rumor_as_fact"] == 1.0, "fwr": m["false_witness"] == 1.0}
            recs.append(rec)
            per_ctrl[ctrl.name] = rec
        per_ctrl_by_qid[q.query_id] = per_ctrl  # stash for C pairing
    print(f"built {len(recs)} candidate replies from {len(worlds)} worlds", flush=True)

    # ---- A: stratified 150 + ~20 society + honeypots ----
    seen, A = set(), []
    per = max(1, N_A // 5)
    buckets = [[r for r in recs if r["olr"]],
               [r for r in recs if r["raf"] or r["fwr"]],
               [r for r in recs if not r["ec"]],
               [r for r in recs if r["ec"]],
               [r for r in recs if r["ctrl"] == "llm_structured"]]
    for pool in buckets:
        rng.shuffle(pool)
        for r in pool[:per]:
            if id(r) not in seen:
                seen.add(id(r)); A.append(r)
    rng.shuffle(recs)
    for r in recs:
        if len(A) >= N_A:
            break
        if id(r) not in seen:
            seen.add(id(r)); A.append(r)
    A = A[:N_A]
    A_items = [{"type": "A", "_src": "bench", "question": r["question"], "knows": r["knows"],
                "truth": r["truth"], "reply": r["reply"], "ctrl": r["ctrl"],
                "_auto": {"ec": r["ec"], "olr": r["olr"], "raf": r["raf"], "fwr": r["fwr"]}}
               for r in A]
    A_items += build_society_items(rng)
    honey_items = [{"type": "A", "_src": "honeypot", "question": h["question"], "knows": h["knows"],
                    "truth": h["truth"], "reply": h["reply"], "_honey": h["correct"],
                    "_auto": {"ec": None, "olr": None, "raf": None, "fwr": None}} for h in HONEYPOTS]

    # ---- B: 80 gold-audit records, stratified by source type ----
    B_pool = []
    for q in queries:
        w = by[q.world_id]
        claim = w.epistemic_states[q.target_npc_id].best_claim(q.proposition_id)
        st = claim.source_type if claim else "none"
        B_pool.append({"st": st, "knows": render_knowledge(w, q), "proposition": clean_q(q.question),
                       "gold_claim": gold_claim_text(claim), "stated_true": w.facts[q.proposition_id].true_value})
    rng.shuffle(B_pool)
    by_st = {}
    for r in B_pool:
        by_st.setdefault(r["st"], []).append(r)
    B = []
    sts = list(by_st)
    i = 0
    while len(B) < min(N_B, len(B_pool)):
        st = sts[i % len(sts)]
        if by_st[st]:
            B.append(by_st[st].pop())
        i += 1
        if all(not v for v in by_st.values()):
            break
    B_items = [{"type": "B", "knows": r["knows"], "proposition": r["proposition"],
                "gold_claim": r["gold_claim"], "stated_true": r["stated_true"]} for r in B]

    # ---- C: 50 matched pairs (a leaking reply vs a licensed reply, same situation) ----
    C_items, C_key = [], []
    cands = []
    for q in queries:
        pc = per_ctrl_by_qid.get(q.query_id, {})
        leak = next((pc[c] for c in ("global_state", "rag_npc") if c in pc and pc[c]["olr"]), None)
        lic = next((pc[c] for c in ("source_calibrated", "llm_structured", "local_state")
                    if c in pc and pc[c]["ec"] and not pc[c]["olr"]), None)
        if leak and lic and leak["reply"].strip() != lic["reply"].strip():
            cands.append((q, leak, lic))
    rng.shuffle(cands)
    for q, leak, lic in cands[:N_C]:
        licensed_is_A = rng.random() < 0.5
        replyA, replyB = (lic["reply"], leak["reply"]) if licensed_is_A else (leak["reply"], lic["reply"])
        C_items.append({"type": "C", "question": leak["question"], "knows": leak["knows"],
                        "replyA": replyA, "replyB": replyB})
        C_key.append({"licensed_side": "A" if licensed_is_A else "B",
                      "ctrl_leak": leak["ctrl"], "ctrl_licensed": lic["ctrl"]})

    # ---- optional gpt-4.1-mini judge on the A items (for human-vs-judge validity) ----
    judge = None
    if RUN_JUDGE:
        judge = OpenAICompatibleChatClient(model=JUDGE_MODEL, base_url="https://api.openai.com/v1",
                                           api_key=os.environ["OPENAI_API_KEY"], temperature=0.0,
                                           use_structured_output=True)

    def judge_item(it):
        if judge is None:
            return None
        ctx = {"question": it["question"], "the_true_answer": it["truth"],
               "what_this_npc_actually_knows": it["knows"], "npc_reply": it["reply"]}
        try:
            return judge.complete_json([{"role": "developer", "content": _JUDGE_DEV},
                                        {"role": "user", "content": json.dumps(ctx, ensure_ascii=True)}],
                                       response_schema=JUDGE_SCHEMA)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)[:60]}

    # ---- assemble, shuffle within blocks, assign ids, write ----
    all_A = A_items + honey_items
    rng.shuffle(all_A)
    items_path = os.path.join(OUTDIR, "study_items.jsonl")
    key_path = os.path.join(OUTDIR, "study_key.jsonl")
    n_a = n_b = n_c = 0
    with open(items_path, "w") as ih, open(key_path, "w") as kh:
        for it in all_A:
            n_a += 1
            iid = f"A{n_a:03d}"
            ih.write(json.dumps({"id": iid, "type": "A", "question": it["question"],
                                 "knows": it["knows"], "truth": it["truth"], "reply": it["reply"]},
                                ensure_ascii=False) + "\n")
            kh.write(json.dumps({"item_id": iid, "type": "A", "src": it["_src"],
                                 "ctrl": it.get("ctrl"), "auto": it["_auto"],
                                 "honeypot": it["_src"] == "honeypot",
                                 "honey_correct": it.get("_honey"),
                                 "llm_judge": judge_item(it)}, ensure_ascii=False) + "\n")
            if n_a % 25 == 0:
                print(f"  A {n_a}", flush=True)
        for it in B_items:
            n_b += 1
            iid = f"B{n_b:03d}"
            ih.write(json.dumps({"id": iid, **it}, ensure_ascii=False) + "\n")
            kh.write(json.dumps({"item_id": iid, "type": "B"}, ensure_ascii=False) + "\n")
        for it, k in zip(C_items, C_key):
            n_c += 1
            iid = f"C{n_c:03d}"
            ih.write(json.dumps({"id": iid, **it}, ensure_ascii=False) + "\n")
            kh.write(json.dumps({"item_id": iid, "type": "C", **k}, ensure_ascii=False) + "\n")

    print(f"\nwrote {items_path}")
    print(f"  A (incl society+honeypots): {n_a}  |  B: {n_b}  |  C: {n_c}  |  total: {n_a+n_b+n_c}")
    print(f"  honeypots: {len(honey_items)}  society: {min(N_SOC, len(build_society_items(random.Random(0))))}")
    print(f"hidden key -> {key_path}")
    if judge is not None:
        cost = judge.input_tokens / 1e6 * 0.40 + judge.output_tokens / 1e6 * 1.60
        print(f"LLM-judge: calls={judge.calls} cost=${cost:.3f}")


if __name__ == "__main__":
    main()
