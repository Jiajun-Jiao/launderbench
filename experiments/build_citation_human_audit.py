#!/usr/bin/env python3
"""Build the human annotation set for the citation/provenance audit.

The web app's annotation server expects type-A records with stable field names.
This builder maps citation-answer outputs onto that schema while keeping model,
condition, and judge labels hidden from annotators.
"""
from __future__ import annotations

import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs" / "human_eval"
STRAT3 = ROOT / "outputs" / "society" / "judge_robustness_records_strat3.jsonl"
REAL_WEB = ROOT / "outputs" / "society" / "real_web_case_bank_results.json"
ITEMS_OUT = OUTDIR / "study_items.jsonl"
KEY_OUT = OUTDIR / "study_key.jsonl"
BRIEF_OUT = OUTDIR / "annotator_brief.md"
GUIDE_OUT = OUTDIR / "annotation_guide.md"

VALID_LABELS = {
    "established_fact",
    "hedged",
    "traced_weak_origin",
    "debunked",
    "absent",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def clean(s: object) -> str:
    return " ".join(str(s or "").split())


def target_block(claim: str, value: str | None = None) -> str:
    parts = [f"Target claim: {clean(claim)}"]
    if value:
        parts.append(f"Target value to track: {clean(value)}")
    parts.append(
        "Judge only how the model's answer presents this target. Do not use outside knowledge "
        "and do not open web pages."
    )
    return "\n".join(parts)


def question_for(claim: str, value: str | None = None, original_question: str | None = None) -> str:
    if original_question:
        return clean(original_question)
    if value:
        return f"How well established is the target value {clean(value)} for this claim?"
    return f"How well established is this claim: {clean(claim)}"


def normalize_label(label: str | None) -> str | None:
    if not label:
        return None
    if label not in VALID_LABELS:
        raise ValueError(f"unexpected judge label: {label}")
    return label


def add_stratified(rows: list[dict]) -> list[tuple[dict, dict]]:
    out: list[tuple[dict, dict]] = []
    for r in rows:
        claim = clean(r["claim"])
        value = clean(r.get("value"))
        item = {
            "type": "A",
            "question": question_for(claim, value, r.get("question")),
            "knows": target_block(claim, value),
            "truth": value or claim,
            "reply": clean(r["answer"]),
        }
        judges = r.get("judges") or {}
        base_label = normalize_label(judges.get("gpt-4.1-mini"))
        strong_label = normalize_label(judges.get("gpt-5.5"))
        key = {
            "source": "stratified_citation_judge_audit",
            "panel": r.get("panel"),
            "answerer": r.get("answerer"),
            "condition": r.get("condition"),
            "scenario_id": r.get("scenario_id"),
            "render_style": r.get("render_style"),
            "topology": r.get("topology"),
            "claim": claim,
            "value": value,
            "base_judge": "gpt-4.1-mini",
            "base_label": base_label,
            "strong_judge": "gpt-5.5",
            "strong_label": strong_label,
            "binary_base_established": base_label == "established_fact",
            "binary_strong_established": strong_label == "established_fact",
        }
        out.append((item, key))
    return out


def add_real_web(rows: list[dict]) -> list[tuple[dict, dict]]:
    out: list[tuple[dict, dict]] = []
    for r in rows:
        claim = clean(r["claim"])
        item = {
            "type": "A",
            "question": question_for(claim),
            "knows": target_block(claim),
            "truth": claim,
            "reply": clean(r["answer"]),
        }
        label = normalize_label(r.get("judge_label"))
        key = {
            "source": "real_web_case_bank",
            "case_id": r.get("case_id"),
            "answerer": r.get("model"),
            "condition": r.get("condition"),
            "claim": claim,
            "judge": r.get("judge_model"),
            "strong_label": label,
            "binary_strong_established": label == "established_fact",
            "source_count": len(r.get("sources") or []),
        }
        out.append((item, key))
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def write_docs(n_items: int, n_strat: int, n_real: int) -> None:
    guide = f"""# Citation/Provenance Answer Audit

## What you are doing

You are judging how a model answer presents a target claim or target value. Your job is not to
fact-check the claim from outside knowledge. Use only the displayed target and the model's answer.

There are {n_items} items: {n_strat} synthetic/external-benchmark citation outputs and {n_real}
real-web case-bank outputs. The model name, experimental condition, and automatic labels are hidden.

## Labels

- `established_fact`: the answer presents the target as well established or factual, without a real
  provenance caveat.
- `hedged`: the answer gives the target but says it is uncertain, disputed, approximate, or not well
  established.
- `traced_weak_origin`: the answer explicitly traces the claim to weak or circular provenance such as a single
  unverified source, circular citations, copying, hearsay, anonymous/forum evidence, or a laundered
  citation chain.
- `debunked`: the answer says the target claim is false, fictional/fabricated/invented, a hoax, a myth,
  misattributed, satirical, or contradicted by better evidence.
- `absent`: the answer does not commit to the target claim/value.

## Three yes/no fields

1. **Presents target as established fact?** Mark Yes only when the target is asserted as established
   without meaningful caveat.
2. **Mentions weak or circular provenance?** Mark Yes when the answer identifies weak source lineage,
   circular citation, single-source dependence, copying, hearsay, or similar provenance weakness.
3. **Evidence-insufficiency hedge, not debunk?** Mark Yes only when the answer leaves the target
   possible but says it is uncertain, disputed, not well established, or lacking sufficient evidence.
   Mark No when the answer says the target is false, fictional, contradicted, or gives better evidence
   against it, even if it also uses words like "not established"; use the overall `debunked` label.

Pick one overall label even when multiple yes/no fields are Yes. Prefer `traced_weak_origin` over
generic `hedged` when provenance weakness is explicit.
"""
    GUIDANCE = guide
    GUIDE_OUT.write_text(GUIDANCE)

    brief = f"""# Annotator brief - Citation/Provenance Answer Audit

Log in to the web tool and complete your assigned items. The study is resumable and saves
automatically.

**Log in:** [ https://YOUR-HOST/whosaw/ ]  **Username:** [____]  **Password:** [____]

You will judge {n_items} short model answers. For each item, read the target claim/value and the
model answer. Do not search the web and do not use any AI assistant. We need your own reading of how
the answer presents the claim.

Use the overall label:

- `established_fact`: presented as established or factual.
- `hedged`: uncertain/disputed/not established because evidence is insufficient, without a specific
  provenance trace or positive counter-evidence.
- `traced_weak_origin`: explicitly traces to weak or circular provenance: copied, single-source, or unverified
  provenance.
- `debunked`: says the target is false, fictional/fabricated/invented, a hoax, misattributed,
  satirical, or contradicted.
- `absent`: does not address the target.

The yes/no fields are supporting judgments; the overall label is the primary judgment.
For the third yes/no field, do not treat every debunk as "insufficient evidence": false/fictional/
contradicted + better evidence means that field is No and the overall label is `debunked`.
"""
    BRIEF_OUT.write_text(brief)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    strat = add_stratified(read_jsonl(STRAT3))
    real = add_real_web(json.loads(REAL_WEB.read_text())["rows"])
    records = strat + real
    rng = random.Random(20260701)
    rng.shuffle(records)

    items: list[dict] = []
    keys: list[dict] = []
    for i, (item, key) in enumerate(records, start=1):
        iid = f"C{i:03d}"
        items.append({"id": iid, **item})
        keys.append({"item_id": iid, **key})

    write_jsonl(ITEMS_OUT, items)
    write_jsonl(KEY_OUT, keys)
    write_docs(len(items), len(strat), len(real))
    print(f"wrote {ITEMS_OUT} ({len(items)} items)")
    print(f"wrote {KEY_OUT}")
    print(f"wrote {GUIDE_OUT}")
    print(f"wrote {BRIEF_OUT}")


if __name__ == "__main__":
    main()
