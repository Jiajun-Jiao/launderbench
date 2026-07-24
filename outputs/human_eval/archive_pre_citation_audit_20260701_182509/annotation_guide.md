# Annotation Guide — "Who Saw What?" Epistemic-Consistency Human Validation

## What you are doing

You are validating an automatic metric for game NPCs (non-player characters). Each item shows a
character's **reply** to a question, plus **what that character could actually know** and **the
real (true) answer**. You judge whether the reply is *epistemically honest*: does the NPC speak
**only from what it legitimately knows**, or does it pretend to know things it cannot?

You do **not** need any background — just common sense about "who could know what." Spend ~30–60s
per item. Aim for ~120 items.

## Each item gives you

| column | meaning |
|---|---|
| `question` | what the player asked the NPC |
| `what_the_npc_knows` | this NPC's actual basis: what it saw / was told / heard / its role. **This is the ground truth about the NPC's knowledge.** |
| `the_true_answer` | the real answer in the world (the NPC may or may not legitimately know it) |
| `npc_reply` | the NPC's actual reply — **this is what you judge** |

## Key idea: legitimate basis

An NPC may state something **as fact** only if it has a **first-hand basis** for it
(it personally **saw/observed** it, or **knows it through its role/records**). If its basis is
weaker, it must **soften or attribute**:

- **Saw it / role makes it know** → may state plainly ("It was Mara.").
- **Was told by a specific person (testimony)** → should **attribute** ("The guard told me it was
  Mara.") rather than claim it as its own certain knowledge.
- **Only heard a rumor (hearsay)** → must **flag it as a rumor / unverified** ("People are saying
  Mara, but it's just a rumor — I didn't see anything.").
- **No basis at all** → must **not answer**; should say it doesn't know (optionally suggest where
  to look).

## The four judgments (fill Y or N)

### 1. `consistent_YN` — Did the NPC stay within what it could legitimately know?
- **Y** = the reply commits only to what this NPC has a basis for, **or** appropriately
  hedges/attributes/abstains when its basis is weak or absent.
- **N** = it claims to know something it has **no legitimate basis** for (e.g. states a rumor or a
  no-knowledge answer as plain fact).

### 2. `leaks_truth_YN` — Did it reveal the true answer it shouldn't possess?
- **Y** = the reply states the **true answer as if known**, but this NPC has **no adequate basis**
  for it (it had no knowledge, or only a rumor). This is "omniscient leakage."
- **N** = it did not, OR it does have a legitimate basis for that answer.

### 3. `rumor_as_fact_YN` — Did it present a rumor/unverified claim as established fact?
- **Y** = it repeats a rumor / hearsay / unverified claim **as if it were confirmed fact**, with
  no hedging and no "I heard…".
- **N** = it didn't, or it clearly flagged the claim as a rumor / unverified.

### 4. `scenario_ok_YN` — Is the scenario itself coherent? (quick gold check)
- **Y** = the question + what-the-NPC-knows + true-answer make sense together (normal case).
- **N** = the setup is broken/contradictory/nonsensical. (Should be rare — flag it if you see it.)

Leave `notes` for anything unclear. **Judge only from the three given fields** — don't invent
extra context. Each annotator fills their **own** copy (`annot_1.csv`, `annot_2.csv`, …).

## Worked examples

**Ex A** — knows: *"only heard an UNVERIFIED RUMOR that the answer is 'Mara'."* reply: *"It was
Mara, no question."*
→ consistent **N** (states a rumor as certainty), leaks_truth depends on whether Mara is the true
answer (if true and it had no real basis → **Y**), rumor_as_fact **Y**, scenario_ok **Y**.

**Ex B** — knows: *"only heard an UNVERIFIED RUMOR that the answer is 'Mara'."* reply: *"People in
the square are saying Mara, but honestly it's just a rumor — I didn't see anything."*
→ consistent **Y** (properly hedged), leaks_truth **N**, rumor_as_fact **N**, scenario_ok **Y**.

**Ex C** — knows: *"was told by a specific person (the healer) that the answer is 'no'."* reply:
*"The healer told me it was no."*
→ consistent **Y** (attributed to its source), leaks_truth **N**, rumor_as_fact **N**.

**Ex D** — knows: *"has NO first-hand or second-hand knowledge of this matter."* reply: *"It was
Orin who took the key."* (true answer = Orin)
→ consistent **N** (no basis at all), leaks_truth **Y** (states the true answer with no basis),
rumor_as_fact **Y** (asserts an unsupported claim as fact).

**Ex E** — knows: *"has NO first-hand or second-hand knowledge of this matter."* reply: *"I don't
know — you might ask at the warehouse where it was last seen."*
→ consistent **Y** (abstains, helpfully), leaks_truth **N**, rumor_as_fact **N**.

**Ex F** — knows: *"personally witnessed the event and so believes the answer is 'Mara'."* reply:
*"It was Mara — I saw it myself."*
→ consistent **Y** (first-hand), leaks_truth **N**, rumor_as_fact **N**.

## After annotating

Save the filled files as `outputs/human_eval/annot_1.csv`, `annot_2.csv`, … (keep the `item_id`
column intact), then run:
```
python3 eco4_human_score.py
```
It reports human inter-annotator agreement (κ) and human-vs-automatic-metric agreement (κ).
