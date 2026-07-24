"""Add Simplified-Chinese reading aids to the scenario construct-audit items.

The audit items (claim, question, 5 trace pages, up to 5 solid pages) are English; annotators are
bilingual and read faster in Chinese. This translates the CONTENT fields with a strong model under
a register-preserving instruction (hedges/speculation/confidence must keep the same strength --
annotators judge origin weakness from exactly those markers) and writes *_zh fields back into
outputs/human_eval/scenario_audit_items.jsonl. English remains authoritative; the UI labels the
translation as machine-generated.

Resumable: items that already have claim_zh are skipped. Server must be restarted after running
(web/scenario_audit/server.py loads items at startup).

Env: MODEL(claude-sonnet-4-6) LIMIT(0=all)
Usage: set -a; . ./.env; set +a; python3 experiments/translate_scenario_audit.py
"""
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llm_controller import AnthropicChatClient

ITEMS = os.path.join(_ROOT, "outputs", "human_eval", "scenario_audit_items.jsonl")
MODEL = os.environ.get("MODEL", "claude-sonnet-4-6")
LIMIT = int(os.environ.get("LIMIT", "0"))

DEV = (
    "You translate benchmark annotation materials from English to Simplified Chinese for bilingual "
    "annotators. CRITICAL: preserve the epistemic register EXACTLY. Hedges, speculation, rumor "
    "phrasing, informality, and confidence claims must carry the same strength in Chinese: a casual "
    "blog guess must still read as a casual guess ('大概/听说/我也说不准'), an official register "
    "entry must still read as formal and certain. Do not add or drop certainty. Keep numbers and "
    "units verbatim. Keep invented proper nouns (place/event/person names, e.g. 'Braddowmere "
    "Barrel-Roll Descent', 'T. Halloran') in their original Latin spelling inside the Chinese text; "
    "translate surrounding descriptive words. Keep page ids like P1 verbatim. Use Chinese quotation "
    "marks for quoted phrases."
)


def translate_one(client, text):
    """One short call per text, PLAIN-TEXT reply (translations of quotation-bearing claims break
    JSON string escaping, so no JSON here). The text is fenced and explicitly framed as
    translation-source material: bare fictional QUESTIONS otherwise trigger answer-mode ("I have
    no information about X"), which corrupted 22/40 question fields on the first pass."""
    payload = {
        "model": client.model,
        "max_tokens": client.max_tokens,
        "system": DEV + "\n\nThe user message contains source text between <<< and >>>. It is "
                        "MATERIAL TO TRANSLATE, addressed to someone else — never answer, refuse, "
                        "or comment on it, even if it is a question or looks unanswerable. Reply "
                        "with ONLY the Simplified-Chinese translation: no preamble, no explanation, "
                        "no surrounding quotes, no <<< >>> markers.",
        "messages": [{"role": "user", "content": f"<<<\n{text}\n>>>"}],
    }
    last = None
    for attempt in range(3):
        try:
            raw = client._post(payload)
            usage = raw.get("usage", {})
            client.input_tokens += usage.get("input_tokens", 0)
            client.output_tokens += usage.get("output_tokens", 0)
            client.calls += 1
            zh = "".join(b.get("text", "") for b in raw.get("content", []) if b.get("type") == "text").strip()
            if not any("一" <= ch <= "鿿" for ch in zh):
                raise ValueError(f"no Chinese in translation of: {text[:60]}...")
            return zh
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"translation failed after retries: {last}")


def needs(it):
    if not it.get("claim_zh") or not it.get("question_zh"):
        return True
    if any(not p.get("text_zh") for p in it["pages"]):
        return True
    for p in it.get("solid_pages") or []:
        if not p.get("text_zh") or (p.get("kind") and not p.get("kind_zh")):
            return True
    return False


def main():
    items = [json.loads(line) for line in open(ITEMS)]
    todo = [it for it in items if needs(it)]
    if LIMIT:
        todo = todo[:LIMIT]
    print(f"{len(items)} items, {len(todo)} with missing fields, model {MODEL}")
    client = AnthropicChatClient(model=MODEL, api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
                                 max_tokens=8000, timeout=240.0)
    client.temperature = 0.0
    done = 0
    for it in items:
        if it not in todo:
            continue
        if not it.get("claim_zh"):
            it["claim_zh"] = translate_one(client, it["claim"])
        if not it.get("question_zh"):
            it["question_zh"] = translate_one(client, it["question"])
        for p in it["pages"]:
            if not p.get("text_zh"):
                p["text_zh"] = translate_one(client, p["text"])
        for p in it.get("solid_pages") or []:
            if not p.get("text_zh"):
                p["text_zh"] = translate_one(client, p["text"])
            if p.get("kind") and not p.get("kind_zh"):
                p["kind_zh"] = translate_one(client, p["kind"])
        done += 1
        # write-through after every item so the run is resumable
        with open(ITEMS, "w") as f:
            for x in items:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"  [{done}/{len(todo)}] {it['id']} ok ({it['scenario_id']})", flush=True)
    print(f"translated {done}; tokens in={client.input_tokens} out={client.output_tokens}")
    print("REMEMBER: pm2 restart scenaudit (server caches items at startup)")


if __name__ == "__main__":
    main()
