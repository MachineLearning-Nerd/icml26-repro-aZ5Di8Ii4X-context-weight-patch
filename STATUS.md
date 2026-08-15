# Reproduction status

## Repository

- Intended final name: MachineLearning-Nerd/icml26-context-parameter-equivalence
- Original name: MachineLearning-Nerd/icml26-repro-aZ5Di8Ii4X-context-weight-patch
- Publication branch: main
- Paper: [Equivalence of Context and Parameter Updates in Modern Transformer Blocks](https://arxiv.org/abs/2511.17864v3)
- OpenReview: [aZ5Di8Ii4X](https://openreview.net/forum?id=aZ5Di8Ii4X)

## Scoped verdicts

- C1 Gemma block: VERIFIED_SCOPED.
- C2 26-layer Gemma extension: VERIFIED_SCOPED.
- C3 controllability framework: VERIFIED_ARCHITECTURE_SCOPED.
- C4 exact float32/naive-bfloat16 percentages: FALSIFIED_SCOPED.
- C5 exact stable-bfloat16 percentage improvement: FALSIFIED_SCOPED.
- C6 named architecture forms: VERIFIED_ARCHITECTURE_SCOPED.

## Verification

The focused suite contains five tests. The four committed claim verifiers
check the durable JSON evidence and negative controls. The final verifier
checks the scorecard, evidence paths, branch naming documentation, and cleanup
invariants without downloading model weights.

The historical publication gate records a live judge score of 2/12 and a
published Hugging Face revision. No new judge score or author endorsement is
claimed by this repository.

## Attribution

All publication commits and rewritten reachable history use:

MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>

The paper authors are thanked and cited in README.md, CITATION.cff, and
SOURCE_MANIFEST.md. No author endorsement is implied.
