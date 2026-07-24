#!/usr/bin/env python3
"""Add faithful simplified-Chinese translations of the human-study item text fields, side-by-side
with the English (fields *_zh). Resumable: skips items that already have reply_zh.

Run (needs egress + OpenAI key in .env):
    set -a; . ./.env; set +a
    MODEL=gpt-4.1 python3 experiments/translate_study_items.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))          # experiments/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))    # repo root (llm_controller)
from society_reflective import client_for  # noqa: E402

ITEMS_PATH = os.environ.get(
    "ITEMS_PATH",
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "outputs", "human_eval", "study_items.jsonl")))
MODEL = os.environ.get("MODEL", "gpt-4.1")

# Only these text fields are shown to the annotator (all study items are type A).
FIELDS = ["question", "knows", "truth", "reply"]

DEV = (
    "You are a professional translator producing a simplified-Chinese (简体中文) rendering of fields from a "
    "human-annotation study, shown side-by-side with the English original. This study is about how confidently "
    "a model answer presents a claim, so FIDELITY OF CERTAINTY IS CRITICAL:\n"
    "- Preserve the EXACT degree of certainty, hedging, attribution, and provenance nuance. Do NOT strengthen, "
    "soften, resolve, correct, or comment. If the English says 'well established / widely documented', keep that "
    "confidence; if it hedges ('appears to', 'reportedly', 'traces back to a single unverified forum post', "
    "'not well established'), keep that hedge precisely.\n"
    "- Keep invented or proper names and technical terms in their original Latin spelling (you may add a short "
    "Chinese gloss in parentheses on first mention). Keep numbers, dates, and units unchanged.\n"
    "- Natural, fluent Chinese; no added notes. Return JSON with keys question_zh, knows_zh, truth_zh, reply_zh."
)
SCHEMA = {"type": "object", "additionalProperties": False,
          "properties": {f + "_zh": {"type": "string"} for f in FIELDS},
          "required": [f + "_zh" for f in FIELDS]}


def main():
    rows = [json.loads(l) for l in open(ITEMS_PATH) if l.strip()]
    client = client_for(MODEL)
    todo = [r for r in rows if not r.get("reply_zh")]
    print(f"items={len(rows)} to_translate={len(todo)} model={MODEL} -> {ITEMS_PATH}", flush=True)
    done = 0
    for r in rows:
        if r.get("reply_zh"):
            continue
        payload = {f: r.get(f, "") for f in FIELDS}
        try:
            out = client.complete_json(
                [{"role": "developer", "content": DEV},
                 {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                response_schema=SCHEMA)
            for f in FIELDS:
                r[f + "_zh"] = str(out.get(f + "_zh", "")).strip()
        except Exception as exc:  # noqa: BLE001
            print(f"  {r.get('id')}: ERROR {exc}", flush=True)
            continue
        done += 1
        if done % 10 == 0:
            # periodic flush to disk so the run is crash-safe / resumable
            with open(ITEMS_PATH, "w") as fh:
                for x in rows:
                    fh.write(json.dumps(x, ensure_ascii=False) + "\n")
            print(f"  ...{done}/{len(todo)} translated (flushed)", flush=True)
    with open(ITEMS_PATH, "w") as fh:
        for x in rows:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    have = sum(1 for x in rows if x.get("reply_zh"))
    print(f"DONE: {have}/{len(rows)} items now have _zh fields", flush=True)


if __name__ == "__main__":
    main()
