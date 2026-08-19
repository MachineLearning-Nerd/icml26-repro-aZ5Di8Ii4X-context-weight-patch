# Scoped reproduction report

## Final verdict

| Claim | Verdict | Meaning |
| --- | --- | --- |
| C1 | `VERIFIED_SCOPED` | One full Gemma 3 1B block/token and five seeded toy trials pass the rank-one construction with negative controls. |
| C2 | `VERIFIED_SCOPED` | The sequential construction passes through all 26 Gemma layers for one target history and token. |
| C3 | `VERIFIED_ARCHITECTURE_SCOPED` | Forty trials across eight actual Transformers class forms pass topology, NumPy, and nonzero-assumption controls. |
| C4 | `FALSIFIED_SCOPED` | The paper's exact 87.5% naive-bfloat16 percentage is not observed; the pinned five-prompt run is 100/100. |
| C5 | `FALSIFIED_SCOPED` | Stable bfloat16 reduces worst logit error but does not improve token agreement beyond the already 100/100 naive run. |
| C6 | `VERIFIED_ARCHITECTURE_SCOPED` | Eight named architecture forms pass at reduced deterministic widths. |

Overall status is `VERIFIED_SCOPED_WITH_NARROW_PRECISION_BOUNDARY`; `publication_allowed` is
`false` for a complete theorem, pretrained-model, hardware, or benchmark reproduction. The C4/C5
falsifications are limited to the pinned checkpoint and CPU protocol; they do not refute the
algebraic construction universally.

## Claim production and evidence boundary

The producer paths are recorded in the README and [CLAIM_EVIDENCE.md](CLAIM_EVIDENCE.md):
`run_gemma_claims_1_2.py` produces C1/C2, `run_architecture_claims_3_6.py` produces C3/C6,
and `run_gemma_claims_4_5.py` plus explicit materialization checks produce C4/C5. Durable raw
evidence is retained under `.openresearch/artifacts/`; the normalized scope summary is
[evidence/claim_summary.json](evidence/claim_summary.json).

The one-checkpoint Gemma result is token-specific, non-Gemma routes use reduced-width random
weights, and precision percentages are five-prompt/100-token protocol results. The historical
2/12 judge score and published Hugging Face revision in `publication_gate.json` are provenance,
not a new score or endorsement.

## Branch and publication policy

`main` is the publication surface. The nine `evidence/*` branches preserve cumulative experiment
snapshots and are mapped in [BRANCH_AUDIT.md](BRANCH_AUDIT.md). The root
[verify_final.py](verify_final.py) checks the exact ten-branch public set, canonical
MachineLearning-Nerd attribution, claim records, durable evidence, hashes, and the inner verifier
without downloading model weights.

Thank you to Adrian Goldwaser, Michael Munn, Javier Gonzalvo, and Benoit Dherin for making the
mathematical construction concrete enough for independent checking; see
[AUTHOR_THANK_YOU.md](AUTHOR_THANK_YOU.md).
