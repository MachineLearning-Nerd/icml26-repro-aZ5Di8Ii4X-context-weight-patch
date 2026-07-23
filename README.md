# Reproducing context-equivalent weight patches on Gemma 3

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/MachineLearning-Nerd/icml26-repro-aZ5Di8Ii4X-context-weight-patch/blob/master/notebooks/context_weight_patch_reproduction.py)

This CPU-only campaign tests all six judged claims from
[*Equivalence of Context and Parameter Updates in Modern Transformer Blocks*](https://arxiv.org/abs/2511.17864)
on the exact public Gemma 3 1B weights and actual Transformers architecture
classes. The previous live judge score is **2/12**; no score increase is claimed
until a new Hugging Face revision is published with approval and judged.

The central algebraic and multilayer claims are verified on the full 26-layer,
999,885,952-parameter model. The reported precision contrast is not observed:
float32, naive bfloat16, and stable bfloat16 each match **100/100** greedy token
predictions, rather than the paper's 100% / 87.5% contrast or 87.5%-to-98%
improvement. Stable inversion still reduces worst logit error from **0.375 to
0.25**. We therefore classify the exact percentage Claims 4 and 5 as
**FALSIFIED under this setup**, not as “failed to reproduce.”

The non-Gemma architecture checks use the actual pinned Transformers classes
at reduced width with deterministic weights. This is a direct test of the
dimension-independent construction, but not pretrained quality evidence for
every named model. No GPU was used.

- [Illustrated technical report](reports/context-parameter-equivalence-2026-07-24/report.md)
- [Self-contained marimo notebook](notebooks/context_weight_patch_reproduction.py)
- Reproduce everything: `uv run --frozen python repro/run_campaign.py`

## Claim results

| Claim | Paper statement | Observed evidence | Assessment |
|---|---|---|---|
| 1 | A Gemma block admits rank-one gate/up patches plus an RMS scale patch | Full Gemma 3 1B layer-0 output L∞ `7.629e-06`; explicit patch checks pass | VERIFIED |
| 2 | The construction extends inductively through all layers | 26/26 layers; maximum layer L∞ `6.104e-05`; final logit L∞ `2.480e-05`; same top-1 | VERIFIED |
| 3 | Input/output controllability suffices | 40 actual-class trials; maximum block L∞ `4.441e-15`; invalid zero input rejected | VERIFIED |
| 4 | Float32 is 100% while naive bfloat16 is 87.5% | Both are `100/100`; repeated complete runs agree on the percentage | FALSIFIED |
| 5 | Stable inversion raises 87.5% to 98% | Naive and stable are both `100/100`; stable lowers worst logit error | FALSIFIED |
| 6 | The construction covers the named architecture families | Gemma, Llama, Falcon, Mistral, Mixtral/MoE, Qwen, GPT-2 and GPT-J all pass | VERIFIED (architecture level) |

## Experiment log

The command below is copied verbatim from `orx exp status`; it is identical on
every experiment node.

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| [`orx/baseline-judged-toy-reproduction`](https://github.com/MachineLearning-Nerd/icml26-repro-aZ5Di8Ii4X-context-weight-patch/tree/orx/baseline-judged-toy-reproduction) | Freeze the judged toy baseline and uv lock | `uv run --frozen python repro/run_campaign.py` | TOY baseline retained | Local CPU, 15s |
| [`orx/real-gemma-3-1b-block-and-multilayer`](https://github.com/MachineLearning-Nerd/icml26-repro-aZ5Di8Ii4X-context-weight-patch/tree/orx/real-gemma-3-1b-block-and-multilayer) | Full pretrained Gemma Claims 1–2 | `uv run --frozen python repro/run_campaign.py` | Claims 1–2 VERIFIED | HF `cpu-upgrade`, 1m30s |
| [`orx/batched-five-prompt-precision-sweep`](https://github.com/MachineLearning-Nerd/icml26-repro-aZ5Di8Ii4X-context-weight-patch/tree/orx/batched-five-prompt-precision-sweep) | Five prompts × 20 tokens × precision modes | `uv run --frozen python repro/run_campaign.py` | 100/100 for all modes; provisional Claim 5 label rejected | HF `cpu-upgrade`, 57m05s |
| [`orx/exact-precision-verdict-plus-architecture-confor`](https://github.com/MachineLearning-Nerd/icml26-repro-aZ5Di8Ii4X-context-weight-patch/tree/orx/exact-precision-verdict-plus-architecture-confor) | Correct joint contract; add named architectures | `uv run --frozen python repro/run_campaign.py` | Claims 3/6 VERIFIED; Claims 4/5 FALSIFIED | HF `cpu-upgrade`, 1h26m |
| [`orx/explicit-bfloat16-matrix-materialization-audit`](https://github.com/MachineLearning-Nerd/icml26-repro-aZ5Di8Ii4X-context-weight-patch/tree/orx/explicit-bfloat16-matrix-materialization-audit) | Materialize every layer's bfloat16 matrices on five audit tokens | `uv run --frozen python repro/run_campaign.py` | 10/10 contextual/action/explicit token matches; route CONCORDANT | HF `cpu-upgrade`, 23m09s |
| `master` | README, report, notebook, and publication manifest | Not run as an experiment (publication surface) | Awaiting explicit publication approval | None |

## Reproduce

Requirements: Python 3.12, `uv`, CPU RAM sufficient for Gemma 3 1B, and access
to the public checkpoint mirror. The lockfile is authoritative.

```bash
uv sync --frozen
uv run --frozen python repro/run_campaign.py
```

The formal runs used Hugging Face `cpu-upgrade` only after local CPU proved
insufficient. The fixed command never changes; variants live in committed code.

## Previous toy-only surface

The repository originally confirmed the rank-one identity on small random
matrices (`d=64`, hidden size 128). Those tests and negative controls remain in
the cumulative suite, but they are labeled **TOY** and are not presented as
full-scale evidence.
