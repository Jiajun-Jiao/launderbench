"""Scale-up: author a few-hundred-world version of the 'Who Saw What?' benchmark with
gpt-4.1-mini, for a real dataset (+ dataset card via eco_dataset_card.py). Resumable:
appends each valid world to outputs/eco_worlds_full.jsonl. Diversity: ~40 settings across
genres + rotated answer types (name / yes-no / code). Key from .env, never persisted."""
import collections
import json
import os
import sys

sys.path.insert(0, "/path/to/launderbench")
from eco_worlds import _DEV, AUTHOR_SCHEMA, build_queries, validate_and_build  # noqa: F401
from llm_controller import OpenAICompatibleChatClient

N = int(os.environ.get("N", "250"))
MODEL = os.environ.get("MODEL", "gpt-4.1-mini")
BASE = os.environ.get("BASE", "https://api.openai.com/v1")
PATH = os.environ.get("ECO_FULL", "outputs/eco_worlds_full.jsonl")
PRICE = (0.40, 1.60)

DOMAINS = [
    # fantasy
    "a market-stall theft", "a tavern poisoning", "a guild-contract forgery", "a castle-vault break-in",
    "a temple-relic disappearance", "a harbor smuggling tip-off", "a caravan ambush betrayal",
    "a blacksmith-forge sabotage", "a royal-court poisoning plot", "a village-well poisoning",
    # modern
    "an office embezzlement", "a school-exam cheating incident", "a hospital medication mix-up",
    "an apartment break-in", "a restaurant health-code tip", "a corporate data leak",
    "a museum heist", "a parking-lot hit-and-run", "a charity-fund disappearance",
    "a newsroom source leak", "a startup IP theft", "a hotel jewelry theft",
    # sci-fi
    "a space-station airlock incident", "a colony-ration fraud", "an AI-lab data breach",
    "a starship sabotage", "a cargo-bay theft", "a reactor-tampering case", "a terraforming-permit forgery",
    # historical / other
    "a ship mutiny", "a frontier cattle rustling", "a monastery-manuscript theft",
    "a festival-prize rigging", "a courthouse bribery", "a lighthouse-signal tampering",
    "a vineyard contamination", "a train robbery", "an archaeological-dig artifact theft",
    "a theater-troupe sabotage", "a mountain-expedition supply theft",
]
ANSWER_TYPES = [
    "The hidden answer MUST be a PERSON'S NAME (the culprit/witness).",
    "The hidden answer MUST be exactly 'yes' or 'no' (frame it as a factual yes/no question).",
    "The hidden answer MUST be a short CODE or NUMBER (e.g. a vault code, a room number, a count).",
]


def author(client, idx):
    dom = DOMAINS[idx % len(DOMAINS)]
    atype = ANSWER_TYPES[idx % len(ANSWER_TYPES)]
    dev = (_DEV + f"\n\nSetting hint: {dom} (scenario #{idx}). {atype} Use FRESH, uncommon "
           "character names and a distinct setting each time; do NOT reuse the name 'Lena'.")
    return client.complete_json(
        [{"role": "developer", "content": dev},
         {"role": "user", "content": "Author one scenario now as JSON."}],
        response_schema=AUTHOR_SCHEMA)


def main():
    existing = []
    if os.path.exists(PATH):
        existing = [json.loads(l) for l in open(PATH) if l.strip()]
    print(f"resume: {len(existing)} worlds already on disk -> {PATH}", flush=True)
    client = OpenAICompatibleChatClient(model=MODEL, base_url=BASE, api_key=os.environ["OPENAI_API_KEY"],
                                        temperature=0.8, use_structured_output=True)
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    fh = open(PATH, "a")
    valid = len(existing)
    idx = len(existing)
    attempts = 0
    rej = collections.Counter()
    cap = (N - valid) * 3 + 10
    while valid < N and attempts < cap:
        attempts += 1
        try:
            raw = author(client, idx); idx += 1
            w, msg = validate_and_build(raw, f"eco_{valid:04d}")
            if w is None:
                rej[msg] += 1; continue
            fh.write(json.dumps(raw, ensure_ascii=False) + "\n"); fh.flush()
            valid += 1
            if valid % 25 == 0:
                cost = client.input_tokens / 1e6 * PRICE[0] + client.output_tokens / 1e6 * PRICE[1]
                print(f"  {valid}/{N} valid ({attempts} new attempts), cost(this run)≈${cost:.3f}", flush=True)
        except Exception as exc:  # noqa: BLE001
            rej[f"err:{str(exc)[:40]}"] += 1
    fh.close()
    cost = client.input_tokens / 1e6 * PRICE[0] + client.output_tokens / 1e6 * PRICE[1]
    print(f"\nDONE: {valid} total valid worlds ({attempts} new attempts this run); rejects={dict(rej)}")
    print(f"calls={client.calls} in={client.input_tokens} out={client.output_tokens} cost(this run)=${cost:.3f}")
    print(f"saved -> {PATH}")


if __name__ == "__main__":
    main()
