# Claim 2 — controllability conditions


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_d3b9a8cc5b58", "created_at": "2026-07-17T12:38:24+00:00", "title": "C2: perfect patch exists for input/output-controllable MLPs"}
-->
Claim: a perfect implicit weight patch is possible for any MLP block where the inner function is input-controllable and the outer function is output-controllable.

The paper introduces input controllability (Definition 3) and output controllability (Definition 4) as the conditions for the rank-1 patch to exist. Our C1 verification demonstrates this concretely for the Gemma SwiGLU MLP: the gate function sigma(W_gate z) is input-controllable (different z produce different gate activations) and the output function W_down(...) is output-controllable (different intermediate values produce different outputs). The rank-1 patches we verified (C1) are a constructive instance of this general theorem.
