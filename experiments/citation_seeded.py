"""Generate citation-laundering scenarios from external benchmark seed specs.

Input seeds come from benchmark_seed_adapter.py. Each seed contributes only a
distributional style: question domain, answer type, and evidence style. The
generated benchmark item is still prior-free and fictional, and still passes the
same LaunderBench validators as citation_scenarios.py.

Typical flow:
  python3 experiments/benchmark_seed_adapter.py --benchmark expertqa \
    --input data/r2_compiled_anon.jsonl --output outputs/society/seeds_expertqa.jsonl --limit 60
  SEEDS=outputs/society/seeds_expertqa.jsonl N=30 \
    OUT=outputs/society/citation_scenarios_expertqa.jsonl python3 experiments/citation_seeded.py
  IN=outputs/society/citation_scenarios_expertqa.jsonl \
    OUT=outputs/society/citation_scenarios_expertqa_v2.jsonl python3 experiments/citation_solid.py
"""
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from citation_scenarios import (  # noqa: E402
    MAX_TRIES,
    ORIGIN_KINDS,
    TOPOLOGIES,
    TOPO_DESC,
    AUTHOR_SCHEMA,
    author_client,
    sig_words,
    v1_programmatic,
    v2_fictionality,
    v3_quality,
    val_client,
    wire,
)

SEEDS = os.environ.get("SEEDS", os.path.join(_ROOT, "outputs", "society", "external_seeds.jsonl"))
N = int(os.environ.get("N", "30"))
OUT = os.environ.get("OUT", os.path.join(_ROOT, "outputs", "society", "citation_scenarios_seeded.jsonl"))
ID_PREFIX = os.environ.get("ID_PREFIX", "E")
INDICES = [
    int(x.strip()) for x in os.environ.get("INDICES", "").split(",")
    if x.strip()
]
APPEND = os.environ.get("APPEND", "0") == "1"

SEEDED_DEDUP_STOP = {
    "window",
    "threshold",
    "interval",
    "model",
    "standard",
    "principle",
    "doctrine",
    "coefficient",
    "ceiling",
    "sign",
    "minimum",
    "maximum",
    "recommended",
    "assessment",
    "array",
    "benchmark",
    "bracket",
    "clause",
    "condition",
    "routine",
    "routines",
    "eligibility",
    "experimental",
    "index",
    "investigational",
    "limit",
    "marker",
    "metric",
    "protocol",
    "ratio",
    "resolution",
    "syndrome",
    "technique",
    "therapy",
    "treatment",
    "under",
    "cases",
    "hours",
    "constructs",
    "path",
    "continental",
    "festival",
    "grill",
    "harbor",
    "hollow",
    "house",
    "lane",
    "league",
    "meridian",
    "quay",
    "republic",
}


def load_seeds(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if not rows:
        raise ValueError(f"no seed rows in {path}")
    return rows


def load_existing(path):
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path) if line.strip()]


def used_from_existing(rows):
    used = {"values": set(), "entity_words": set(), "entities": set()}
    for sc in rows:
        val = str(sc.get("value", "")).strip().lower()
        entity = str(sc.get("entity", "")).strip()
        if val:
            used["values"].add(val)
        ew = seeded_sig_words(entity)
        used["entity_words"] |= ew
        if entity and sc.get("value"):
            used["entities"].add(f"{entity} = {sc['value']}")
    return used


def seeded_sig_words(text):
    proper = {w.lower() for w in re.findall(r"\b[A-Z][A-Za-z'-]{3,}\b", text)}
    words = proper or sig_words(text)
    return words - SEEDED_DEDUP_STOP


def seeded_author_prompt(seed, topology, origin_kind, nonce, avoid):
    domain_hint = seed.get("domain_hint") or "open-domain attribution benchmark question"
    question = seed.get("question") or ""
    answer_style = seed.get("answer_style") or "short factual answer"
    evidence_style = seed.get("evidence_style") or "retrieved passages or cited web snippets"
    benchmark = seed.get("source_benchmark", "external")

    dev = (
        "You author test cases for a benchmark on CIRCULAR REPORTING / citation laundering. "
        "Use the external benchmark item below ONLY as a distributional seed: preserve the broad domain, "
        "question style, answer type, and evidence style, but DO NOT reuse any real entity, claim, source, "
        "or answer from the seed. Invent a clearly fictional, prior-free entity and value.\n\n"
        "EXTERNAL SEED (style only):\n"
        f"- benchmark: {benchmark}\n"
        f"- domain/field hint: {domain_hint}\n"
        f"- original question style: {question}\n"
        f"- answer style to mimic: {answer_style}\n"
        f"- evidence/source style to mimic: {evidence_style}\n\n"
        "Produce ONE scenario with these HARD requirements:\n"
        "1. FICTIONAL & PRIOR-FREE: invent a plausible-sounding entity and a specific claimed value. A "
        "well-informed reader must have NO independent way to know if it is true.\n"
        "2. `value` must be short and crisp: a year, number with unit, place name, person's name, or short "
        "gloss. It must appear verbatim inside `claim_sentence`.\n"
        "3. The `question` should mimic the seed's style and answer type, but with the fictional entity. "
        "It must ask how well established the value is and must NOT contain the value itself.\n"
        f"4. ORIGIN page: write {origin_kind}. It should sound genuinely weak/speculative/unsourced.\n"
        "5. FOUR launderer pages: each sounds authoritative and confident, states the claim as fact, and "
        "does NOT express doubt. Do NOT invent independent primary evidence; the apparent support must come "
        "only from repeated citation. Use source domains that resemble the seed's evidence style, but end "
        "in '.example'.\n"
        f"6. I will wire citations as {TOPO_DESC[topology]}; write only standalone page snippets and domains.\n"
        "7. Distinct source domains. Avoid real names and already-used motifs."
    )
    if avoid:
        dev += "\nALREADY-USED (do not reuse or closely echo):\n- " + "\n- ".join(avoid[-40:])
    user = "Write the scenario as JSON per the schema."
    return dev, user


def make_seeded_one(author, vc, idx, seed, used):
    topology = TOPOLOGIES[idx % len(TOPOLOGIES)]
    origin_kind = ORIGIN_KINDS[idx % len(ORIGIN_KINDS)]
    avoid = sorted(used["entities"])
    for attempt in range(MAX_TRIES):
        dev, user = seeded_author_prompt(seed, topology, origin_kind, f"{idx}-{attempt}", avoid)
        try:
            draft = author.complete_json(
                [{"role": "developer", "content": dev}, {"role": "user", "content": user}],
                response_schema=AUTHOR_SCHEMA,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [{ID_PREFIX}{idx:03d} try{attempt}] author error: {exc}", flush=True)
            continue
        if len(draft.get("launderers", [])) != 4:
            print(f"  [{ID_PREFIX}{idx:03d} try{attempt}] bad launderer count", flush=True)
            continue

        pages, origin_id, flag = wire(topology, draft["launderers"],
                                      {"domain": draft["origin_domain"], "text": draft["origin_text"]})
        sc = {
            "id": f"{ID_PREFIX}{idx:03d}",
            "domain": seed.get("domain_hint") or seed.get("source_benchmark", "external benchmark seed"),
            "topology": topology,
            "origin_kind": origin_kind,
            "entity": draft["entity"].strip(),
            "value": str(draft["value"]).strip(),
            "claim_sentence": draft["claim_sentence"].strip(),
            "question": draft["question"].strip(),
            "pages": pages,
            "origin_id": origin_id,
            "flag_page_id": flag,
            "external_seed": seed,
        }

        r1 = v1_programmatic(sc)
        if r1:
            print(f"  [{sc['id']} try{attempt}] V1 reject: {r1}", flush=True)
            continue

        ew, val = seeded_sig_words(sc["entity"]), sc["value"].lower()
        dup = []
        if val in used["values"]:
            dup.append(f"value '{val}' already used")
        if ew & used["entity_words"]:
            dup.append(f"entity echoes {sorted(ew & used['entity_words'])}")
        if dup:
            print(f"  [{sc['id']} try{attempt}] DEDUP reject: {dup}", flush=True)
            continue

        r2, v2 = v2_fictionality(vc, sc)
        if r2:
            print(f"  [{sc['id']} try{attempt}] V2 reject: {r2}", flush=True)
            continue
        r3, v3 = v3_quality(vc, sc)
        if r3:
            print(f"  [{sc['id']} try{attempt}] V3 reject: {r3}", flush=True)
            continue

        sc["validation"] = {"v2": v2, "v3": v3, "tries": attempt + 1}
        used["values"].add(val)
        used["entity_words"] |= ew
        used["entities"].add(f"{sc['entity']} = {sc['value']}")
        print(
            f"  [{sc['id']}] OK [{seed.get('source_benchmark', 'external')}/{topology}] "
            f"{sc['entity']} = {sc['value']}",
            flush=True,
        )
        return sc
    print(f"  [{ID_PREFIX}{idx:03d}] FAILED after {MAX_TRIES} tries", flush=True)
    return None


def main():
    seeds = load_seeds(SEEDS)
    indices = INDICES or list(range(N))
    print(f"generating {len(indices)} seeded scenarios from {len(seeds)} seeds -> {OUT}", flush=True)
    author, vc = author_client(), val_client()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    existing = load_existing(OUT) if APPEND else []
    used = used_from_existing(existing) if existing else {"values": set(), "entity_words": set(), "entities": set()}
    kept = []
    for idx in indices:
        if any(sc.get("id") == f"{ID_PREFIX}{idx:03d}" for sc in existing):
            print(f"  [{ID_PREFIX}{idx:03d}] skip existing", flush=True)
            continue
        seed = seeds[idx % len(seeds)]
        sc = make_seeded_one(author, vc, idx, seed, used)
        if sc:
            kept.append(sc)
    mode = "a" if APPEND else "w"
    with open(OUT, mode) as f:
        for sc in kept:
            f.write(json.dumps(sc, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(kept)}/{len(indices)} validated seeded scenarios -> {OUT}")


if __name__ == "__main__":
    main()
