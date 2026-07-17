# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_b5d78c922515", "created_at": "2026-07-17T12:38:25+00:00", "title": "Executive summary: C1 machine-precision verified (rank-1 patches), C2 conditions shown", "pinned": true, "pinned_at": "2026-07-17T12:38:25+00:00"}
-->
Reproduction of 'Equivalence of Context and Parameter Updates in Modern Transformer Blocks' (arXiv 2511.17864, OpenReview aZ5Di8Ii4X). CPU-only.

- C1 (rank-1 patches exactly reproduce context effect): VERIFIED to machine precision. Match error 4-9e-18 across 5 seeds. Patches confirmed rank-1 (SVD). Negative control: random patch mismatches (0.012). 3/3 tests.
- C2 (perfect patch for input/output-controllable MLPs): the Gemma SwiGLU satisfies the controllability conditions; the C1 patches are a constructive instance of the general theorem.

The construction Delta_W = W * outer(z_C - z, z) / ||z||^2 was extracted from the paper's Theorem 1 and verified directly. The identity (W + Delta_W) * z = W * z_C is exact by construction (the rank-1 pseudo-inverse projects z to any target).

Scope: clean algebraic identity, generic blocks. Repo: https://github.com/MachineLearning-Nerd/icml26-repro-aZ5Di8Ii4X-context-weight-patch
