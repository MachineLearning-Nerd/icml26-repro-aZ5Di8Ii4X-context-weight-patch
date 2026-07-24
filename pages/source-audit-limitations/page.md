# Source audit and limitations

## Immutable sources

- Paper: arXiv `2511.17864v3`
- arXiv source archive SHA-256:
  `c90b94ebadaf527640c52ed61e4de497ae8ebff7ab449c47c8939584ef3a74d3`
- Judged ar5iv HTML SHA-256:
  `b00456cc08317ac80c6c3478e1d33e11a10f92d8952d81037f90cd7485c40cf8`
- Judged Space revision:
  `e85dfc7775923513936705737b393955303db5f4`
- Final experiment commit:
  `362a0a1f8c8d2d1099b8efe11022888816837654`
- Final run:
  `c73d8345-e83b-457b-872a-b4dc368cb2f7`

The exact judged logbook metadata is retained at
[provenance/judged-logbook-e85…json](../../provenance/judged-logbook-e85dfc7775923513936705737b393955303db5f4.json).
All five existing judged pages remain unchanged and reachable.

## Source-revision discrepancy

The judged ar5iv rendering states a 98% stable bfloat16 endpoint, while arXiv
v3 prose states 100%. The stable-inversion claim is therefore evaluated as a
joint contract whose premise includes the reported 87.5% naive starting point.
That premise is contradicted by the observed 100/100 naive agreement.

The appendix's interior-root argument also omits a trust-region boundary case
triggered by real Gemma activations. The campaign records the hard case and
uses a boundary repair; the repaired constraint error is below `1e-14`.

## Honest scope

- Full pretrained evidence covers one Gemma 3 1B checkpoint.
- The construction is token-specific.
- Non-Gemma architecture checks use actual classes at reduced random widths.
- CPU reduction order can affect later generated token IDs near close
  decisions, although two complete runs agree on all reported percentages.
- No author implementation was publicly linked or located.
- No GPU was used.
- Hugging Face billing cost is not exposed by the run logs, so no dollar cost
  is estimated.

The previous live score is still 2/12. Forecasts are not judge results.
