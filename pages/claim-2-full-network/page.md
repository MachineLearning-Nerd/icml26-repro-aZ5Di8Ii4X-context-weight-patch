# Claim 2 — 26-layer inductive extension

**Result: VERIFIED · HIGH confidence**

The construction was applied sequentially through all 26 Gemma 3 1B blocks.
Each child layer used the previous patched state, directly testing the
inductive composition rather than isolated layers.

- Layers checked: `26/26`
- Maximum layer-output L∞: `6.103515625e-05`
- Final-logit L∞: `2.47955322265625e-05`
- Final top-1 prediction: identical
- Unpatched reduced-context logit L∞: `23.490646362304688`

The negative control separates by more than five orders of magnitude. The
construction remains token-specific; this result does not assert one patch
works for arbitrary future tokens.

Evidence:
[contract](../../.openresearch/artifacts/claim_2/claim_contract.json),
[source audit](../../.openresearch/artifacts/claim_2/source_audit.md),
[method](../../.openresearch/artifacts/claim_2/method.md),
[raw results](../../.openresearch/artifacts/claim_2/raw_results.json),
[verifier](../../.openresearch/artifacts/claim_2/claim_verifier.py),
[independent checker](../../.openresearch/artifacts/claim_2/independent_checker_output.json),
[negative control](../../.openresearch/artifacts/claim_2/negative_control_output.json),
and [evaluation](../../.openresearch/artifacts/claim_2/EVAL.md).
