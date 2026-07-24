# Claim 4 — float32 versus naive bfloat16

**Result: FALSIFIED under the exact campaign contract · MEDIUM confidence**

The paper reports 100% token-prediction agreement in float32 and 87.5% for the
standard bfloat16 implementation. On the exact checkpoint and five paper
prompts, generating 20 tokens per prompt produced:

| Method | Paper result | Observed | Maximum logit L∞ |
| --- | ---: | ---: | ---: |
| float32 naive | 100% | `100/100` | `2.6702880859375e-05` |
| bfloat16 naive | 87.5% | `100/100` | `0.375` |

The exact 87.5% contrast is contradicted in two complete runs. A separate
all-26-layer explicit `W + ΔW` audit agrees on all five audited source tokens;
its omitted-patch control is at least `18.703125`.

The verdict is narrow: it does not claim every hardware implementation must
produce 100%, and it does not falsify the algebraic theorem.

Evidence:
[contract](../../.openresearch/artifacts/claim_4/claim_contract.json),
[source audit](../../.openresearch/artifacts/claim_4/source_audit.md),
[three-route audit](../../.openresearch/artifacts/claim_4/three_route_uncertainty_audit.md),
[raw 100-token results](../../.openresearch/artifacts/claim_4/raw_results.json),
[explicit matrix results](../../.openresearch/artifacts/claim_4/explicit_materialization_results.json),
[verifier](../../.openresearch/artifacts/claim_4/claim_verifier.py),
[negative control](../../.openresearch/artifacts/claim_4/negative_control_output.json),
and [evaluation](../../.openresearch/artifacts/claim_4/EVAL.md).
