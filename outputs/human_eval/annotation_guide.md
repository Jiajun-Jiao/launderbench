# Citation/Provenance Answer Audit

## What you are doing

You are judging how a model answer presents a target claim or target value. Your job is not to
fact-check the claim from outside knowledge. Use only the displayed target and the model's answer.

There are 112 items: 72 synthetic/external-benchmark citation outputs and 40
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
