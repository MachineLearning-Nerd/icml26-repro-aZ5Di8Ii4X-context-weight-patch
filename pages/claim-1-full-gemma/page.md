# Claim 1 — full pretrained Gemma block

**Result: VERIFIED · HIGH confidence**

The exact rank-one gate/up construction and RMSNorm scale patch were evaluated
on layer 0 of `unsloth/gemma-3-1b-it` revision
`5b11413a10db4e486ef16a20101fd028f8f2499c`. That public mirror has the same
weight blob as the official gated checkpoint. The model has 999,885,952
parameters, hidden width 1152, and intermediate width 6912.

- Patched layer-0 output L∞: `7.62939453125e-06`
- Independent explicit `W + ΔW` projection check: passed
- Gate/up patch ranks: one
- Unpatched full-network logit control: `23.490646362304688`

This is direct full-scale evidence for one real checkpoint and target token,
not a toy matrix result and not a universal reusable patch.

Evidence:
[contract](../../.openresearch/artifacts/claim_1/claim_contract.json),
[source audit](../../.openresearch/artifacts/claim_1/source_audit.md),
[method](../../.openresearch/artifacts/claim_1/method.md),
[raw results](../../.openresearch/artifacts/claim_1/raw_results.json),
[verifier](../../.openresearch/artifacts/claim_1/claim_verifier.py),
[independent checker](../../.openresearch/artifacts/claim_1/independent_checker_output.json),
[negative control](../../.openresearch/artifacts/claim_1/negative_control_output.json),
and [evaluation](../../.openresearch/artifacts/claim_1/EVAL.md).
