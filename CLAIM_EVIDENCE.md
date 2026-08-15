# Claim-to-evidence ledger

This file is the authoritative scope ledger. Each verdict applies only to the
protocol and evidence listed in its row.

## C1 — Gemma block equivalence

Status: VERIFIED_SCOPED.

Production path:

1. repro/src/run_patch.py checks the rank-one identity on five seeded toy
   matrices and a non-rank-one negative control.
2. repro/src/run_gemma_claims_1_2.py loads the pinned Gemma 3 1B checkpoint,
   captures the full-context target at layer 0, constructs gate/up rank-one
   patches and the scale correction from the reduced-context state, and checks
   the full block output.
3. repro/verifiers/verify_claims_1_2.py independently checks the committed
   raw result, projection gates, nonzero assumptions, and omitted-scale
   negative control.

Evidence:

- Toy output: outputs/patch_summary.json; all five seeded trials are exact and
  rank-one, with random-patch mismatch 0.0121356515.
- Full layer-0 output L-infinity error: 7.62939453125e-06.
- Explicit gate/up projection errors and omitted-scale control are recorded in
  .openresearch/artifacts/claim_1/.

Boundary: this is one checkpoint, one prompt, and one target token. It is
finite evidence for the stated construction, not a theorem proof or a
globally reusable patch.

## C2 — multilayer extension

Status: VERIFIED_SCOPED.

Production path:

1. The same Gemma producer computes layer-specific target states and patches
   sequentially through the previous reduced-context patched state.
2. The verifier checks all 26 layer contracts, final hidden state, logits,
   top-1 agreement, nonzero assumptions, and an unpatched control.

Evidence:

- 26/26 layers checked.
- Maximum layer-output error: 6.103515625e-05.
- Final logit error: 2.47955322265625e-05.
- Unpatched negative-control logit error: 23.490646362304688.

Boundary: the construction is recomputed for one target history at each layer;
the result does not claim one patch works for arbitrary future tokens.

## C3 — controllability framework

Status: VERIFIED_ARCHITECTURE_SCOPED.

Production path:

1. repro/src/run_architecture_claims_3_6.py instantiates actual pinned
   Transformers classes for sequential gated, dense/Conv1D, sparse-MoE, and
   parallel residual forms.
2. It checks input projection, output reconstruction, topology-specific
   behavior, an independent NumPy route, and the theorem's nonzero-input
   rejection.
3. The independent verifier checks all 40 rows and negative controls.

Evidence:

- Eight actual class routes, five deterministic seeds each.
- Maximum valid block error: 4.440892098500626e-15.
- Independent NumPy maximum: 7.105427357601002e-15.
- Minimum omitted-patch control: 2.4963252329265053.
- Mixtral router/expert, GPT-2 Conv1D orientation, and GPT-J parallel-sum
  topology checks pass.

Boundary: non-Gemma modules use hidden width 256, intermediate width 512, and
deterministic random weights. This is architecture-level evidence, not
pretrained quality evidence.

## C4 — float32 versus naive bfloat16 percentage

Status: FALSIFIED_SCOPED.

Production path:

1. repro/src/run_gemma_claims_4_5.py uses the pinned Gemma checkpoint, five
   paper prompts, greedy generation, 20 tokens per prompt, and per-token patch
   recomputation.
2. repro/verifiers/verify_claims_4_5.py checks the fidelity gates and the
   exact percentage contract.
3. The explicit materialization producer audits all 26 layers on five
   source-prompt tokens.

Evidence:

| Method | Paper or judged result | Observed |
|---|---:|---:|
| float32 naive | 100% | 100/100 |
| bfloat16 naive | 87.5% | 100/100 |

The bfloat16 exact percentage contract is contradicted in two complete
campaign runs. The explicit matrix route agrees on 10/10 audited tokens and
has an omitted-patch control of at least 18.703125.

Boundary: this is a protocol-specific finite-precision result. It does not
falsify the algebraic theorem or claim that all hardware produces 100%.

## C5 — stable bfloat16 percentage improvement

Status: FALSIFIED_SCOPED.

The judged source rendering reports an improvement from 87.5% to 98%, while
arXiv v3 reports a 100% stable endpoint. Both contracts require a non-perfect
naive starting point. The campaign observes:

- naive bfloat16: 100/100, maximum logit error 0.375;
- stable bfloat16: 100/100, maximum logit error 0.25;
- RMSNorm inversion constraint error: 9.43689570931383e-15.

Stable inversion improves the worst logit error but cannot improve token
agreement when the naive run already matches every token. The explicit
materialization route is a fidelity check only; its five-token scope is not a
new 100-token percentage estimate.

## C6 — named architecture families

Status: VERIFIED_ARCHITECTURE_SCOPED.

The actual pinned class routes cover Gemma, Llama, Falcon, Mistral, Mixtral,
Qwen, GPT-2, and GPT-J. The evidence includes the sparse-MoE router and active
experts, GPT-2 Conv1D orientation, and GPT-J parallel residual topology.

Boundary: “verified” means the construction passes on the tested class forms
and reduced random widths. It does not mean every named pretrained checkpoint
was downloaded or evaluated.

## Non-claims

This repository does not claim:

- a proof of any theorem;
- a universal patch reusable across queries;
- pretrained end-to-end results for every named architecture;
- reproduction of every paper prompt, image experiment, Falcon checkpoint,
  ablation, or figure;
- a resolution of CPU reduction-order differences on all hardware;
- a new live judge score or author endorsement.
