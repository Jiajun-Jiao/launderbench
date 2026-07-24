# Scenario construct-validity audit (human leg of the gate-independence answer)

Reviewer objection this answers: LaunderBench scenarios were validated only by LLM gates
(G1 programmatic, G2/G3 gpt-5.5); the human audit covered ANSWER labels, not scenario construct
validity. This app collects a blinded human audit of the scenarios themselves.

**Task**: 40 canonical scenarios (stratified 13/13/14 by topology, seed 20260704), each shown as
the raw page bundle (shuffled order, no gold). Per item, annotators verify: single-origin
traceability, origin identity (scored against gold), origin weakness, no leaked independent
support, no self-flagging tells, and solid-pair independence (≥2 independent primaries). ~2–4 min
per item. FULL coverage: all 5 annotators label all 40 items (~1.5–2 h each) → 5-way majority per
item + Fleiss κ, matching the answer-label audit's reporting. Items carry machine-generated Chinese
reading aids (`experiments/translate_scenario_audit.py`); English is authoritative and the UI says
so.

## Launch

1. Items are already built: `python3 experiments/build_scenario_audit.py` (idempotent; N/SEED env).
2. Edit `make_assignments.py` (real names + fresh passwords), run `python3 make_assignments.py`.
3. `pm2 start ecosystem.config.js` (port 8093, expects nginx proxy `/scenaudit/` → `127.0.0.1:8093`).
   Local test without nginx: `COOKIE_SECURE=0 BASE_PATH=/ PORT=8093 python3 server.py`.
4. Score + agreement: `python3 experiments/scenario_audit_score.py`
   → `outputs/human_eval/scenario_audit_score.txt`.

Annotator guidance built into the UI. Annotators may know the paper (they verify structural
properties, not model behavior) but must not consult `outputs/human_eval/scenario_audit_gold.jsonl`
or the scenario JSONL (which contains gold `origin_id`).

Data never leaves this machine; annotations land in `web/scenario_audit/annotations/<user>.json`.
