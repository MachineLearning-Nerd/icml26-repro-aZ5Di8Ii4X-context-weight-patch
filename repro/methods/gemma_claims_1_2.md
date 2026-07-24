# Method — pretrained Gemma 3 single- and multi-layer checks

The test token is the last token of the paper's Mars-weather prompt. The
full-context path processes the entire prompt. The reduced path processes only
that last token, at the same absolute rotary position. Hooks record the
full-context state immediately after attention, the pre-MLP RMSNorm output,
the unscaled/scaled post-MLP RMSNorm values, and each block output.

For every layer, the reduced path:

1. computes attention without the preceding prompt;
2. applies the paper's rank-1 gate and up projection actions;
3. recomputes the gated MLP and down projection;
4. applies `Δm = (v_C - v) / f_C`; and
5. feeds the actual patched output into the next layer.

Layer 0 is checked two ways: a low-rank action implementation and explicit
materialization of `W + ΔW`. The latter is the independent implementation.
The unpatched reduced model and omission of `Δm` are negative controls.

All thresholds are preregistered in the claim contracts. The verifier exits
nonzero if a predicate fails.

