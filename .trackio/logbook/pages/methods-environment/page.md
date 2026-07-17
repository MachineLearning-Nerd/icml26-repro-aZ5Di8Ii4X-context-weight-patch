# Methods & environment


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_d74da822d2cd", "created_at": "2026-07-17T12:38:24+00:00", "title": "Methods: direct algebraic verification from the paper's construction"}
-->
Paper: arXiv 2511.17864 (no official code; the construction formula was extracted from the paper HTML Theorem 1). The claim is a clean algebraic identity verified by direct computation.

Gemma-style block: input x, RMSNorm, SwiGLU MLP (W_gate, W_up, W_down with activation). Context = attention output added to x. The construction Delta_W = W * outer(z_C - z, z) / ||z||^2 is applied to W_gate and W_up. The identity (W + Delta_W) * z = W * z_C is verified numerically to machine precision.

Environment: Python 3.12, numpy/scipy. CPU <1s. No GPU, no training. Verified on random generic blocks (no paper-specific instance needed). Multi-seed (5 seeds, d=64, hidden=128).
