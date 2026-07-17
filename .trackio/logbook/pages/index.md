# Repro - Context = Rank-1 Weight Patches

## Pages

| Page |
| --- |
| [Claim 1 — rank-1 patches exact](#/claim-1-rank-1-patches-exact) |
| [Claim 2 — controllability conditions](#/claim-2-controllability-conditions) |
| [Methods & environment](#/methods-environment) |
| [Negative controls & falsification](#/negative-controls-falsification) |
| [Conclusion](#/conclusion) |


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_2f0ab2302d26", "created_at": "2026-07-17T12:38:22+00:00", "title": "Context = Rank-1 Weight Patches (aZ5Di8Ii4X) — ICML 2026 reproduction"}
-->
Reproduction of 'Equivalence of Context and Parameter Updates in Modern Transformer Blocks' (arXiv 2511.17864, OpenReview aZ5Di8Ii4X). CPU-only (numpy/scipy). Clean exact-identity claim.

Headlines: C1 — the context effect in a Gemma-style transformer block can be PERFECTLY mapped to rank-1 patches on the MLP weight matrices. VERIFIED to machine precision: patched MLP on z (no context) exactly equals original MLP on z_C (with context), match error ~1e-18 across 5 seeds. The patches are confirmed rank-1 (SVD: second singular value ~1e-18). Negative control: a random (non-rank-1) patch produces a clear mismatch. 3/3 tests pass.
