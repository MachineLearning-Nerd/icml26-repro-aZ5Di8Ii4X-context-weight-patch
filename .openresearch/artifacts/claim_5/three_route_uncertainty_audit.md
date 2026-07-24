# Claims 4 and 5: three-route uncertainty audit

This document closes the mandatory three-route loop for the precision claims.
The routes use different interpretations and methods; they are not seed or
runtime repetitions.

## Route 1 — Direct five-prompt autoregressive comparison

Interpretation: the Section 4 percentages quantify greedy token agreement over
the five source prompts and twenty generated tokens per prompt.

Command: `uv run --frozen python repro/run_campaign.py`

Result on the full Gemma 3 1B checkpoint:

- float32 captured-action patch: 100/100;
- naive bfloat16 captured-action patch: 100/100;
- stable bfloat16 captured-action patch: 100/100;
- maximum logit L-infinity errors: `2.6703e-05`, `0.375`, and `0.25`.

The observed naive percentage contradicts 87.5%. Stable inversion reduces
worst logit error but cannot raise token agreement when naive is already 100%.
Risk: captured rank-one actions can omit rounding from storing `W + delta_W`.

## Route 2 — Explicit bfloat16 matrix materialization

Interpretation: the paper's words “weight-patched model” require constructing
the updated matrices in the implementation dtype, so the Route 1 shortcut must
be checked against actual bfloat16 `W + delta_W`.

Command: `uv run --frozen python repro/run_campaign.py`

Method: all 26 layers, both naive and stable methods, first generated token of
each of the five source prompts. Gate/up matrices and the stable down matrix are
materialized. This is a fidelity spot check, not a replacement percentage.

Result: written by the experiment to
`explicit_materialization_results.json` and independently classified as
`CONCORDANT` or `DIVERGENT`.

## Route 3 — Source-revision and mathematical audit

Interpretation: before treating 98% as the claim endpoint, the authoritative
source revision, figure counts, and stated inversion proof must agree.

Commands and sources:

- `orx paper 2511.17864`
- arXiv v3 source archive
  `https://export.arxiv.org/e-print/2511.17864v3`, retrieved with an explicit
  User-Agent; SHA-256
  `c90b94ebadaf527640c52ed61e4de497ae8ebff7ab449c47c8939584ef3a74d3`
- ar5iv rendering `https://ar5iv.labs.arxiv.org/html/2511.17864`, SHA-256
  `b00456cc08317ac80c6c3478e1d33e11a10f92d8952d81037f90cd7485c40cf8`
- targeted searches for an author implementation; no public repository was
  linked in the paper source or located in the searches on 2026-07-23/24.

Findings:

- arXiv v3 prose reports stable bfloat16 at 100%, whereas the judged ar5iv
  rendering records 98%;
- v3 prose reports naive bfloat16 at 87.5%, while the visible Mars example is
  18/20 (90%), so the denominator is aggregate rather than per-prompt;
- the Appendix-B existence proof misses the trust-region hard case when the
  numerator vanishes on the minimum-scale eigenspace. The real checkpoint
  triggers this case; the reproduction implements and tests the boundary
  solution with inversion residual at most `9.44e-15`.

This route establishes that the endpoint itself is revision-sensitive and that
the published interior-root algorithm is incomplete on the tested checkpoint.
It does not manufacture a replacement percentage.

## Confidence consequence

After exactly these three routes, Claims 4 and 5 may be assigned MEDIUM or HIGH
only if Route 2 is concordant with Route 1. If it diverges, both claims remain
LOW/BLOCKED unless an author implementation or a full explicit 100-token run
becomes available. No fourth route is started solely to satisfy retry policy.
