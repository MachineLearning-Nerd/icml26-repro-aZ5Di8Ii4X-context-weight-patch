# Claim 5 — stable bfloat16 inversion

**Result: FALSIFIED under the exact joint contract · MEDIUM confidence**

The judged rendering reports that stable RMSNorm inversion raises agreement
from 87.5% to 98%. The arXiv v3 prose instead gives a 100% stable endpoint.
Under either endpoint, the stated improvement requires the naive method to
start at 87.5%.

Observed on five prompts × 20 generated tokens:

| Method | Observed agreement | Maximum logit L∞ |
| --- | ---: | ---: |
| bfloat16 naive | `100/100` | `0.375` |
| bfloat16 stable | `100/100` | `0.25` |

Stable inversion reduces the worst logit error, but agreement cannot rise
because the naive route already matches every token. The repaired stable solver
satisfies its inversion constraint to `9.43689570931383e-15`.

Three materially different routes were completed: a full captured-action
generation sweep, explicit full-matrix materialization, and an independent
source/mathematical audit. The explicit route is 10/10 concordant but covers
only five source tokens and is not treated as a percentage estimate.

Evidence:
[contract](../../.openresearch/artifacts/claim_5/claim_contract.json),
[source audit](../../.openresearch/artifacts/claim_5/source_audit.md),
[three-route audit](../../.openresearch/artifacts/claim_5/three_route_uncertainty_audit.md),
[raw 100-token results](../../.openresearch/artifacts/claim_5/raw_results.json),
[explicit matrix results](../../.openresearch/artifacts/claim_5/explicit_materialization_results.json),
[verifier](../../.openresearch/artifacts/claim_5/claim_verifier.py),
[negative control](../../.openresearch/artifacts/claim_5/negative_control_output.json),
and [evaluation](../../.openresearch/artifacts/claim_5/EVAL.md).
