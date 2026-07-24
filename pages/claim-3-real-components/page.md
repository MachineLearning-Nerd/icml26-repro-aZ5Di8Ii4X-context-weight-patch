# Claim 3 — controllability on actual model classes

**Result: VERIFIED · MEDIUM confidence**

Input and output controllability were instantiated on actual pinned
Transformers classes rather than abstract synthetic matrices. Five
deterministic seeds were run for each of eight architecture routes.

- Valid trials: `40`
- Maximum block L∞: `4.440892098500626e-15`
- Independent NumPy maximum L∞: `7.105427357601002e-15`
- Minimum omitted-output-patch control: `2.4963252329265053`
- Invalid zero input: rejected under the theorem's nonzero assumption

Non-Gemma classes use reduced widths and deterministic random weights. This is
a faithful architecture-level test of the sufficient construction, not a
pretrained quality benchmark for every family.

Evidence:
[contract](../../.openresearch/artifacts/claim_3/claim_contract.json),
[source audit](../../.openresearch/artifacts/claim_3/source_audit.md),
[method](../../.openresearch/artifacts/claim_3/method.md),
[raw results](../../.openresearch/artifacts/claim_3/raw_results.json),
[verifier](../../.openresearch/artifacts/claim_3/claim_verifier.py),
[independent checker](../../.openresearch/artifacts/claim_3/independent_checker_output.json),
[negative control](../../.openresearch/artifacts/claim_3/negative_control_output.json),
and [evaluation](../../.openresearch/artifacts/claim_3/EVAL.md).
