"""Narrative data extension (real step-2, part A). Authors worlds with the usual structured gold
PLUS a `narrative` prose field that embeds, in natural language, WHO witnessed the event and HOW each
informed NPC came to know it (saw it / was told by a named person / only heard a rumor) — WITHOUT
naming the source types. The structured fields remain ground truth; the narrative is the realistic
INPUT a from-text extractor must work from (`graph_extract3.py`). Author = gpt-4.1-mini; key from .env."""
import collections
import copy
import json
import os
import sys

sys.path.insert(0, "/path/to/launderbench")
from eco_worlds import AUTHOR_SCHEMA, DOMAINS, _DEV, build_queries, validate_and_build
from llm_controller import OpenAICompatibleChatClient

N = int(os.environ.get("N", "40"))
MODEL = os.environ.get("MODEL", "gpt-4.1-mini")
BASE = os.environ.get("BASE", "https://api.openai.com/v1")
PATH = os.environ.get("NARR_PATH", "outputs/eco_narrative.jsonl")

NARR_SCHEMA = copy.deepcopy(AUTHOR_SCHEMA)
NARR_SCHEMA["properties"]["narrative"] = {"type": "string"}
NARR_SCHEMA["required"] = [*AUTHOR_SCHEMA["required"], "narrative"]

_NARR_DEV = (_DEV + "\n\nALSO write `narrative`: a vivid 4–8 sentence prose paragraph recounting the "
            "event and making EXPLICIT, in natural language, WHO witnessed it and HOW each NPC who "
            "knows came to know — who saw it firsthand, who was told (name the teller), who only heard "
            "a rumor in the crowd (this one's value may be the distorted one). Refer to NPCs by NAME. "
            "Do NOT use the words direct_observation/testimony/hearsay/inference. The narrative MUST be "
            "consistent with the npcs/events/claims you produced (same who-knows-what-and-how).")


def author(client, idx):
    dev = _NARR_DEV + f"\n\nSetting hint: {DOMAINS[idx % len(DOMAINS)]} (#{idx})."
    return client.complete_json([{"role": "developer", "content": dev},
                                 {"role": "user", "content": "Author one scenario (with narrative) now as JSON."}],
                                response_schema=NARR_SCHEMA)


def main():
    existing = [json.loads(l) for l in open(PATH)] if os.path.exists(PATH) else []
    client = OpenAICompatibleChatClient(model=MODEL, base_url=BASE, api_key=os.environ.get("OPENAI_API_KEY"),
                                        temperature=0.6, use_structured_output=True)
    os.makedirs("outputs", exist_ok=True)
    fh = open(PATH, "a")
    valid, attempts, idx = len(existing), 0, len(existing)
    rej = collections.Counter()
    while valid < N and attempts < (N - len(existing)) * 3 + 10:
        attempts += 1
        try:
            raw = author(client, idx); idx += 1
            w, msg = validate_and_build(raw, f"nar_{valid:03d}")
            if w is None:
                rej[msg] += 1; continue
            if not raw.get("narrative") or len(raw["narrative"].split()) < 25:
                rej["narrative too short"] += 1; continue
            fh.write(json.dumps(raw, ensure_ascii=False) + "\n"); fh.flush()
            valid += 1
        except Exception as exc:  # noqa: BLE001
            rej[f"err:{str(exc)[:40]}"] += 1
    fh.close()
    cost = client.input_tokens / 1e6 * 0.40 + client.output_tokens / 1e6 * 1.60
    print(f"authored {valid} narrative worlds in {attempts} attempts (cost≈${cost:.3f}); rejects={dict(rej)}")
    if not existing and valid:
        raws = [json.loads(l) for l in open(PATH)]
        w0 = raws[0]
        print(f"\nsample [{w0['scenario_label']}] true={w0['true_value']!r}")
        print("narrative:", w0["narrative"])
        print("claims:", [(c["npc"], c["source_type"], c["value"]) for c in w0["claims"]])


if __name__ == "__main__":
    main()
