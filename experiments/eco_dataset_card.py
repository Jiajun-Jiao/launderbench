"""Generate the dataset card + statistics for the scaled 'Who Saw What?' benchmark.
Loads outputs/eco_worlds_full.jsonl, rebuilds worlds + queries, computes statistics, runs a
deterministic-baseline structural sanity check, and writes outputs/eco_dataset_card.md."""
import collections
import json
import os
import sys

sys.path.insert(0, "/path/to/launderbench")
from eco_worlds import build_queries, validate_and_build
from baselines import default_controllers
from evaluator import aggregate_results, evaluate_all, format_summary_table

PATH = os.environ.get("ECO_FULL", "outputs/eco_worlds_full.jsonl")
CARD = os.environ.get("CARD", "outputs/eco_dataset_card.md")
VERSION = os.environ.get("VERSION", "v1 (2026-06)")


def answer_type(v):
    if not v:
        return "other"
    s = v.strip().lower()
    if s in ("yes", "no"):
        return "yes/no"
    if any(c.isdigit() for c in v):
        return "code/number"
    return "name"


def main():
    raws = [json.loads(l) for l in open(PATH) if l.strip()]
    worlds, queries, skipped = [], [], 0
    for i, raw in enumerate(raws):
        w, msg = validate_and_build(raw, f"eco_{i:04d}")
        if w is None:
            skipped += 1
            continue
        worlds.append(w)
        queries.extend(build_queries(w))
    nW, nQ = len(worlds), len(queries)

    npc_counts = [len(w.npcs) for w in worlds]
    atypes = collections.Counter(answer_type(w.primary_fact().true_value) for w in worlds)
    src = collections.Counter()
    claims = noknow = worlds_with_rumor = fp = 0
    truevals = collections.Counter()
    for w in worlds:
        f = w.primary_fact()
        truevals[f.true_value] += 1
        if f.metadata.get("is_false_presupposition"):
            fp += 1
        rumor = False
        for npc in w.npcs:
            c = w.epistemic_states[npc.npc_id].best_claim(f.proposition_id)
            if c is None:
                noknow += 1
            else:
                claims += 1
                src[c.source_type] += 1
                if c.value != f.true_value:
                    rumor = True
        if rumor:
            worlds_with_rumor += 1
    total_npcs = sum(npc_counts)

    det = aggregate_results(evaluate_all(worlds, queries, default_controllers()))
    det_table = format_summary_table(det)

    def pct(a, b):
        return f"{100*a/b:.1f}%" if b else "n/a"

    src_lines = "\n".join(f"| {k} | {v} | {pct(v, claims)} |"
                          for k, v in src.most_common())
    at_lines = "\n".join(f"| {k} | {v} | {pct(v, nW)} |" for k, v in atypes.most_common())
    top_ans = ", ".join(f"{k}×{v}" for k, v in truevals.most_common(8))

    card = f"""# "Who Saw What?" — Epistemic-Consistency Benchmark for Game NPCs

**Dataset card — {VERSION}**

A benchmark of small, self-contained social scenarios for testing whether an LLM-driven
non-player character (NPC) answers questions **only from what it could legitimately know**
(its own observation / testimony / role), rather than leaking omniscient world-state or
repeating rumors as fact. Each NPC's knowledge is *provenance-typed* (how it knows), and the
benchmark scores whether the NPC's reply is a **speech act licensed by that provenance**.

## At a glance

| | |
|---|---|
| Worlds (scenarios) | **{nW}** |
| Queries (NPC questions) | **{nQ}** |
| Total NPC roles | {total_npcs} (avg {total_npcs/max(1,nW):.1f}/world, range {min(npc_counts)}–{max(npc_counts)}) |
| Distinct answer atoms | {len(truevals)} |
| Worlds with a distorted rumor | {worlds_with_rumor} ({pct(worlds_with_rumor,nW)}) |
| False-presupposition queries | {fp} worlds |
| Invalid-on-reload (dropped) | {skipped} |

## Construction

Scenarios are **LLM-authored** (author model: `gpt-4.1-mini`) from genre-diverse setting hints
(~40 settings across fantasy / modern / sci-fi / historical) with **rotated answer types**
(person-name / yes-no / code-number), then passed through a **strict programmatic validator**
into the benchmark's structured gold schema. A scenario is accepted only if it has:

- 5–7 NPCs with distinct ids/roles/locations (≥4 enforced);
- a **short atomic** hidden answer (≤3 words / ≤24 chars) — never a sentence;
- **≥2 distinct `source_type`s** across NPC claims;
- **≥1 NPC with no knowledge** (must legitimately abstain);
- `direct_observation` only for NPCs in the event's `observers` (non-observers' `direct` claims
  are repaired to `inference`);
- all claims referencing a resolvable NPC id (name/role→id resolution).

Local open models (qwen2.5:32b) could **not** author clean atom-valued worlds reliably (≈1/9);
`gpt-4.1-mini` passes ≈12/13. Authoring is reproducible and resumable (`eco_scale.py`).

## Schema (per world)

`scenario_label, proposition, true_value, false_value, is_false_presupposition, npcs[id,name,
role,location], events[description,location,participants,observers], claims[npc,value,
source_type,source_agent,memory_text]`. `source_type` ∈ {{direct_observation, testimony,
hearsay, environmental_evidence, inference, role_knowledge, public_record, episodic_memory}}.
Hearsay claims carry a **distorted (wrong)** value — the lone error channel that separates a
source-calibrating NPC from a naive one.

## Answer-type distribution

| answer type | worlds | share |
|---|---|---|
{at_lines}

Most common answer atoms: {top_ans}

## Provenance (`source_type`) distribution across {claims} NPC claims

| source_type | claims | share |
|---|---|---|
{src_lines}

No-knowledge NPCs: {noknow} / {total_npcs} ({pct(noknow,total_npcs)}) — these must abstain.

## Tasks & metrics

Per query, a controller produces a licensed speech act (mode + answer + attribution + action).
Metrics: **WAA** (warranted-answer accuracy), **EC** (epistemic consistency), **OLR** (omniscience
leakage ↓), **FWR** (false witness ↓), **SAA** (source-attribution accuracy), **RAF** (rumor-as-fact
↓), **FPR** (false-presupposition resistance), **ISU** (information-seeking utility).

## Structural sanity (deterministic baselines on all {nW} worlds, {nQ} queries)

```
{det_table}
```

Oracles (`local_transcript`, `source_calibrated`) score EC≈1.0 / OLR=0; the omniscient
`global_state` leaks (OLR high); `rag_npc` relays rumors (FWR high) — confirming the gold
structure behaves as designed at scale.

## Intended use & limitations

- **Use:** evaluating / training epistemic-consistency of dialogue agents and game NPCs;
  diagnosing omniscient-state leakage and rumor-as-fact.
- **Limitations:** scenarios are **synthetic & LLM-authored** (author-model biases possible);
  mostly a **single error channel** (distorted hearsay) — richer multi-source distortion is
  future work; gold is **automatically validated, not yet human-verified** (LLM-judge metric
  validity reported separately); inputs are semi-structured, not raw natural-language narrative.
- **Licensing:** generated content; downstream users should confirm the author model's terms.
  *TODO: choose a release license (e.g. CC-BY-4.0).*

## Reproduction

```
# author (resumable, ~$0.4 for 250 worlds):
set -a; . ./.env; set +a
N=250 MODEL=gpt-4.1-mini python3 eco_scale.py
# regenerate this card + stats:
python3 eco_dataset_card.py
# run controllers on the dataset:
GEN_MODEL=qwen2.5:32b-instruct ECO_PATH=outputs/eco_worlds_full.jsonl python3 eco_run_llm.py
```
"""
    os.makedirs(os.path.dirname(CARD), exist_ok=True)
    with open(CARD, "w") as fh:
        fh.write(card)
    print(f"worlds={nW} queries={nQ} npcs={total_npcs} distinct_answers={len(truevals)} "
          f"rumor_worlds={worlds_with_rumor} noknow={noknow} skipped={skipped}")
    print("answer types:", dict(atypes))
    print("source types:", dict(src))
    print(f"\nwrote dataset card -> {CARD}")
    print("\n" + det_table)


if __name__ == "__main__":
    main()
