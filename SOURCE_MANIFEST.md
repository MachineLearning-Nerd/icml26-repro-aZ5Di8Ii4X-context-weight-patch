# Source manifest

## Paper

| Field | Value |
|---|---|
| Title | Equivalence of Context and Parameter Updates in Modern Transformer Blocks |
| Authors | Adrian Goldwaser; Michael Munn; Javier Gonzalvo; Benoit Dherin |
| arXiv | https://arxiv.org/abs/2511.17864v3 |
| OpenReview | https://openreview.net/forum?id=aZ5Di8Ii4X |
| Version audited | v3, revised 6 July 2026 |
| Audit date | 2026-08-15 |
| Paper source archive SHA-256 | c90b94ebadaf527640c52ed61e4de497ae8ebff7ab449c47c8939584ef3a74d3 |

## Model and software inputs

| Input | Pin or boundary |
|---|---|
| Public model mirror | unsloth/gemma-3-1b-it |
| Model revision | 5b11413a10db4e486ef16a20101fd028f8f2499c |
| Official weight identity | google/gemma-3-1b-it revision dcc83ea841ab6100d6b47a070329e1ba4cf78752 |
| Python | 3.12 |
| Dependency lock | uv.lock |
| Transformers classes | actual pinned classes, source hashes recorded in claim 3/6 artifacts |

No author implementation was linked by the paper or committed here. The
experiment code is an independent implementation of the paper equations and
protocols. The full Gemma checkpoint is downloaded at execution time and is
not committed.

## Generated evidence

| Producer | Evidence |
|---|---|
| repro/src/run_patch.py | outputs/patch_summary.json |
| repro/src/run_gemma_claims_1_2.py | .openresearch/artifacts/claim_1/ and claim_2/ |
| repro/src/run_architecture_claims_3_6.py | .openresearch/artifacts/claim_3/ and claim_6/ |
| repro/src/run_gemma_claims_4_5.py | .openresearch/artifacts/claim_4/ and claim_5/ |
| repro/src/run_explicit_materialization_claims_4_5.py | explicit_materialization_results.json in claim 4/5 |
| repro/verifiers/*.py | independent checker outputs and EVAL.md files |

The historical publication gate and Hugging Face revision under publication_gate.json
and release/awaiting-judge.json are provenance records, not new evidence or
current judge results.

## Thank-you note

Thank you to Adrian Goldwaser, Michael Munn, Javier Gonzalvo, and Benoit Dherin
for releasing the paper and its precise mathematical construction. The paper
made it possible to separate exact algebraic identities from assumptions,
finite-precision behavior, and the limits of empirical reproduction.
