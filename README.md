# Context–Parameter Equivalence Audit — ICML 2026

Independent reproduction audit of *Equivalence of Context and Parameter
Updates in Modern Transformer Blocks* by Adrian Goldwaser, Michael Munn, Javier
Gonzalvo, and Benoit Dherin.

- Paper: [arXiv:2511.17864v3](https://arxiv.org/abs/2511.17864v3)
- OpenReview: [aZ5Di8Ii4X](https://openreview.net/forum?id=aZ5Di8Ii4X)
- Intended repository:
  [MachineLearning-Nerd/icml26-context-parameter-equivalence](https://github.com/MachineLearning-Nerd/icml26-context-parameter-equivalence)
- Original repository: icml26-repro-aZ5Di8Ii4X-context-weight-patch

The paper proves that, under controllability and nonzero-activation
assumptions, context-dependent changes in modern Transformer blocks can be
represented by token-specific rank-one updates to MLP weights and, for
Gemma-style blocks, an RMSNorm scale update. It extends the construction from
one block to multilayer models and discusses numerical stability.

This repository is an independent audit, not the authors' implementation. It
does not claim theorem proofs, universal hardware behavior, pretrained
evidence for every architecture family, or author endorsement.

## Audit scorecard

| Claim | Scope of the evidence | Status |
|---|---|---|
| C1 — Gemma block equivalence | One full Gemma 3 1B layer/token plus five-seed toy checks and negative controls | VERIFIED_SCOPED |
| C2 — multilayer extension | Sequential construction through all 26 Gemma layers for one target token | VERIFIED_SCOPED |
| C3 — controllability framework | 40 trials across eight actual pinned Transformers class forms, with topology and assumption controls | VERIFIED_ARCHITECTURE_SCOPED |
| C4 — float32 versus naive bfloat16 percentages | Five prompts × 20 greedy tokens; observed 100/100 bfloat16 instead of the paper's 87.5% | FALSIFIED_SCOPED |
| C5 — stable bfloat16 percentage improvement | Naive and stable both 100/100; stable improves worst logit error but not token agreement | FALSIFIED_SCOPED |
| C6 — named architecture coverage | Gemma, Llama, Falcon, Mistral, Mixtral, Qwen, GPT-2, and GPT-J actual classes | VERIFIED_ARCHITECTURE_SCOPED |

“Falsified” is deliberately narrow: the exact percentage contract is
contradicted under the pinned checkpoint and CPU protocol. It is not a claim
that the algebraic construction is false on every implementation.

## How each claim is produced

| Claim | Producer path | Independent or durable evidence |
|---|---|---|
| C1 | repro/src/run_patch.py establishes the rank-one identity; repro/src/run_gemma_claims_1_2.py checks full Gemma layer 0 | outputs/patch_summary.json and .openresearch/artifacts/claim_1/ |
| C2 | repro/src/run_gemma_claims_1_2.py computes layer-specific patches from the previous reduced-context state through all 26 blocks | .openresearch/artifacts/claim_2/ |
| C3 | repro/src/run_architecture_claims_3_6.py instantiates actual Transformers classes and checks input/output controllability, topology, and negative controls | .openresearch/artifacts/claim_3/ |
| C4 | repro/src/run_gemma_claims_4_5.py runs the five-prompt, 100-token precision comparison; the verifier checks the evidence contract | .openresearch/artifacts/claim_4/ |
| C5 | The same precision producer evaluates naive/stable bfloat16; repro/src/run_explicit_materialization_claims_4_5.py audits explicit full matrices on five tokens | .openresearch/artifacts/claim_5/ |
| C6 | repro/src/run_architecture_claims_3_6.py covers sequential, Conv1D, sparse-MoE, and parallel residual routes | .openresearch/artifacts/claim_6/ |

The machine-readable scope and verdict ledger is
[evidence/claim_summary.json](evidence/claim_summary.json). The detailed
claim-to-evidence explanation is [CLAIM_EVIDENCE.md](CLAIM_EVIDENCE.md).

## What the evidence actually shows

### Full Gemma evidence

The pinned public mirror is
unsloth/gemma-3-1b-it at revision
5b11413a10db4e486ef16a20101fd028f8f2499c, with 999,885,952 parameters, 26
layers, hidden width 1152, and intermediate width 6912. The official weight
blob identity is recorded in the Claim 1 contract.

- Layer-0 output L-infinity error: 7.62939453125e-06.
- Maximum error across 26 sequential layers: 6.103515625e-05.
- Final logit error: 2.47955322265625e-05.
- Unpatched negative-control logit error: 23.490646362304688.
- The final top-1 token agrees for the tested prompt and target position.

These are one-checkpoint, token-specific results. They do not show that one
patch can be reused for arbitrary future queries.

### Precision discrepancy

The paper reports 100% float32 agreement and 87.5% naive bfloat16 agreement.
The arXiv v3 prose gives a 100% stable endpoint, while the judged rendering
retained in the provenance record says 98%. This audit observes:

| Method | Paper or judged endpoint | Observed | Maximum logit error |
|---|---:|---:|---:|
| float32 naive | 100% | 100/100 | 2.6702880859375e-05 |
| bfloat16 naive | 87.5% | 100/100 | 0.375 |
| bfloat16 stable | 98% judged / 100% arXiv v3 | 100/100 | 0.25 |

Stable inversion improves the worst logit error, but there is no token
agreement improvement because the naive run already matches every audited
token. The explicit full-matrix route agrees on 10/10 audited source-token
pairs; five tokens are a fidelity audit, not a replacement 100-token estimate.

### Architecture scope

Five deterministic seeds are used for each of eight actual Transformers class
routes. The maximum valid block error is 4.440892098500626e-15 and the minimum
omitted-patch negative control is 2.4963252329265053. Non-Gemma routes use
reduced-width random weights, so this is architecture-level evidence rather
than pretrained language-quality evidence.

## Branch map

The final repository keeps the experiment snapshots as descriptive evidence
branches. The complete original-to-final mapping and pre-rename tips are in
[BRANCH_AUDIT.md](BRANCH_AUDIT.md).

| Final branch | Purpose |
|---|---|
| main | Combined publication surface and cumulative evidence |
| evidence/baseline-toy | Original five-seed rank-one toy baseline |
| evidence/architecture-matrix | Initial architecture-conformance scaffold |
| evidence/gemma-block-multilayer | Full Gemma block and multilayer snapshot |
| evidence/precision-cross-check | Gemma precision setup and metadata cross-check |
| evidence/stable-inversion | Stable-inversion and precision implementation |
| evidence/precision-sweep | Batched five-prompt precision sweep |
| evidence/precision-verdict-architecture | Joint precision verdict plus architecture checks |
| evidence/explicit-bfloat16-audit | Explicit bfloat16 matrix-materialization audit |
| evidence/release-candidate | Final cumulative evidence release candidate |

No branch named or prefixed orx remains in the final remote state.

## Reproduce or verify

The lockfile requires Python 3.12 and CPU-capable PyTorch. The full campaign
downloads the pinned public checkpoint and can take hours on CPU.

~~~bash
uv sync --frozen
uv run --frozen pytest -q
uv run --frozen python repro/verifiers/verify_claims_1_2.py
uv run --frozen python repro/verifiers/verify_claims_3_6.py
uv run --frozen python repro/verifiers/verify_claims_4_5.py
uv run --frozen python repro/verifiers/verify_explicit_materialization_claims_4_5.py
uv run --frozen python repro/src/verify_final.py
~~~

To regenerate the complete campaign evidence, including the model download:

~~~bash
uv run --frozen python repro/run_campaign.py
~~~

The report and page-by-page evidence are under
[reports/context-parameter-equivalence-2026-07-24/](reports/context-parameter-equivalence-2026-07-24/)
and [pages/](pages/). Source provenance and input boundaries are documented in
[SOURCE_MANIFEST.md](SOURCE_MANIFEST.md).

Machine-readable citation metadata is in [CITATION.cff](CITATION.cff).

## Citation

~~~bibtex
@article{goldwaser2026equivalence,
  title   = {Equivalence of Context and Parameter Updates in Modern Transformer Blocks},
  author  = {Goldwaser, Adrian and Munn, Michael and Gonzalvo, Javier and Dherin, Benoit},
  journal = {arXiv preprint arXiv:2511.17864},
  year    = {2026},
  doi     = {10.48550/arXiv.2511.17864},
  url     = {https://arxiv.org/abs/2511.17864}
}
~~~

## Thank you

Thank you to Adrian Goldwaser, Michael Munn, Javier Gonzalvo, and Benoit Dherin
for publishing the paper and making the mathematical construction concrete
enough to audit. The distinction between exact algebraic equivalence,
controllability assumptions, and finite-precision behavior makes this a useful
reproducibility case study.
