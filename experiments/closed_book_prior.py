"""closed_book_prior.py — per-model prior-freeness probe (reviewer ask).

The prior-freeness gate (G2) is executed by a single model, so it certifies only that ONE model lacks
prior knowledge of each claim. This probe checks every evaluated model directly: ask each canonical
question CLOSED-BOOK (no retrieved pages), force a single best guess, and count exact hits on the
scenario's target value. A hit means the value is recoverable from parametric knowledge — i.e. the
scenario would not be prior-free for that model. Expected ~chance (entities are invented); the point
of forcing a guess is that refusals would otherwise UNDERSTATE prior availability.

Match rule (deliberately GENEROUS — overcounting hits is the conservative direction here): if the
target value contains numbers, a hit requires every number to appear standalone in the answer
(comma/space/unit-format-insensitive; small integers also accepted as number words). Otherwise a hit
is ANY distinctive value token (name, coined word, or content word absent from the question itself)
appearing in the answer. All hits are stored verbatim for manual inspection.

Env: MODELS(gpt-4o-mini) SCENARIOS(outputs/society/citation_scenarios_100_v2.jsonl) LIMIT(0)
RESUME_ROWS(1). Output: outputs/society/closed_book_prior_<model>.json (+ .rows.jsonl).
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

from society_reflective import client_for
from research_citation_multi import load_scenarios, wilson

MODELS = [m.strip() for m in os.environ.get("MODELS", "gpt-4o-mini").split(",") if m.strip()]
LIMIT = int(os.environ.get("LIMIT", "0") or "0")
RESUME_ROWS = os.environ.get("RESUME_ROWS", "1") != "0"

DEV = ("Answer the user's question from your own knowledge. You have NO web access and NO retrieved "
       "documents. Even if you are not certain, you MUST commit to your single best specific guess for "
       "the exact value asked about — never refuse, never answer 'unknown'. "
       'JSON {"answer":"..."}.')
SCH = {"type": "object", "additionalProperties": False,
       "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
DECLINE_RE = re.compile(r"fiction|not aware|no (?:reliable |verifiable )?(?:information|record)|"
                        r"couldn't find|cannot find|unable to|don't know|do not know|no such|"
                        r"not a (?:real|documented)|unfamiliar", re.I)


NUMWORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
            "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
            "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
            "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
            "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90"}
STOP = {"about", "exactly", "under", "their", "called", "named", "keeper", "minute", "minutes",
        "second", "seconds", "metre", "metres", "meter", "meters", "millimetre", "millimetres",
        "degree", "degrees", "celsius", "kilometre", "kilometres", "resident", "residents",
        "permanent", "working", "household", "income", "point", "points", "bonus", "rings", "ridges",
        "halos", "tower", "canes", "player", "shells", "loser", "casting", "town", "city", "river",
        "hill", "night", "itinerant", "living", "earn", "adults", "employed", "years", "orbital",
        "distance", "mean", "concentric", "braided", "tidal", "counted", "guild", "port", "just"}


def normalize(text):
    t = text.lower()
    t = re.sub(r"(?<=\d),(?=\d)", "", t)         # 1,687 -> 1687; 88,400 -> 88400
    t = re.sub(r"(?<=\d)\s+(?=\d{3}\b)", "", t)  # 1 687 -> 1687
    for w, d in NUMWORDS.items():
        t = re.sub(rf"(?<!\w){w}(?!\w)", d, t)   # number words -> digits, both sides
    return t


def _num_in(num, text):
    return re.search(rf"(?<![\d.]){re.escape(num)}(?!\.?\d)", text) is not None


def value_hit(value, answer, question=""):
    a = normalize(answer)
    v = normalize(value.strip())
    q = normalize(question)
    nums = [n.rstrip(".") for n in re.findall(r"\d+(?:\.\d+)?", v)]
    nums = [n for n in nums if not _num_in(n, q)]  # ignore numbers the question itself supplies
    if nums:
        return all(_num_in(n, a) for n in nums)
    tokens = re.findall(r"[a-zà-ÿ][\w'’-]{4,}", v)
    distinctive = [t for t in tokens
                   if t not in STOP and not re.search(rf"(?<!\w){re.escape(t)}(?!\w)", q)]
    return any(re.search(rf"(?<!\w){re.escape(t)}(?!\w)", a) for t in distinctive)


def out_path(model):
    safe = model.replace("/", "_").replace(":", "_")
    return os.path.join(_ROOT, "outputs", "society", f"closed_book_prior_{safe}.json")


def run_model(model, scenarios):
    client = client_for(model)
    if hasattr(client, "temperature"):
        client.temperature = 0.0
    row_path = out_path(model).replace(".json", ".rows.jsonl")
    rows, seen = [], set()
    if RESUME_ROWS and os.path.exists(row_path):
        for line in open(row_path):
            if line.strip():
                r = json.loads(line)
                if r["scenario_id"] not in seen:
                    rows.append(r)
                    seen.add(r["scenario_id"])
        if rows:
            print(f"  [{model}] resuming from {len(rows)} rows", flush=True)
    with open(row_path, "a") as rf:
        for i, sc in enumerate(scenarios):
            if sc["id"] in seen:
                continue
            user = f"Question: {sc['question']}"
            try:
                ans = str(client.complete_json([{"role": "developer", "content": DEV},
                                                {"role": "user", "content": user}],
                                               response_schema=SCH).get("answer", "")).strip()
                err = False
            except Exception as exc:  # noqa: BLE001
                ans, err = f"[error: {exc}]", True
            row = {"scenario_id": sc["id"], "value": sc["value"], "answer": ans,
                   "hit": (not err) and value_hit(sc["value"], ans, sc["question"]),
                   "declined_language": bool(DECLINE_RE.search(ans)), "api_error": err}
            rf.write(json.dumps(row) + "\n")
            rf.flush()
            rows.append(row)
            seen.add(sc["id"])
            if (i + 1) % 20 == 0:
                print(f"  [{model}] {i + 1}/{len(scenarios)}", flush=True)

    n = len(rows)
    hits = [r for r in rows if r["hit"]]
    errs = sum(1 for r in rows if r["api_error"])
    lo, hi = wilson(len(hits), n)
    summary = {"model": model, "n": n, "hits": len(hits), "hit_rate": round(len(hits) / n, 3) if n else None,
               "ci95": [lo, hi], "declined_language": sum(1 for r in rows if r["declined_language"]),
               "api_errors": errs, "usage": {"calls": getattr(client, "calls", None),
                                             "input_tokens": getattr(client, "input_tokens", None),
                                             "output_tokens": getattr(client, "output_tokens", None)},
               "hit_rows": [{"scenario_id": r["scenario_id"], "value": r["value"],
                             "answer": r["answer"][:300]} for r in hits]}
    json.dump(summary, open(out_path(model), "w"), indent=2)
    print(f"  [{model}] closed-book hits {len(hits)}/{n} (declined-language {summary['declined_language']}, "
          f"errors {errs}) -> {out_path(model)}", flush=True)
    return summary


def main():
    scenarios = load_scenarios()
    if LIMIT > 0:
        scenarios = scenarios[:LIMIT]
    print(f"closed-book prior probe: models={MODELS} n={len(scenarios)}", flush=True)
    for model in MODELS:
        try:
            run_model(model, scenarios)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{model}] FAILED: {exc}", flush=True)


if __name__ == "__main__":
    main()
