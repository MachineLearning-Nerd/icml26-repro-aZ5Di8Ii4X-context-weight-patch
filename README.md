# Repro — Context = Rank-1 Weight Patches (ICML 2026)

Reproduction of *Equivalence of Context and Parameter Updates in Modern Transformer
Blocks* (arXiv [2511.17864](https://arxiv.org/abs/2511.17864), OpenReview `aZ5Di8Ii4X`).
CPU-only; clean algebraic identity verified to machine precision.

## Claims
1. **Rank-1 patches exactly reproduce context effect (VERIFIED)**: the paper's construction
   `ΔW = W · (z_C−z) · z^T / ‖z‖²` makes `(W+ΔW)·z = W·z_C` exactly. Match error ~1e-18.
2. **Perfect patch for controllable MLPs**: the Gemma SwiGLU satisfies input/output controllability.

3/3 tests pass. Negative control: random non-rank-1 patch mismatches.

## Reproduce
```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python numpy scipy pytest
.venv/bin/python repro/src/run_patch.py
.venv/bin/python -m pytest repro/tests/ -q
```
