# Method: Gemma generation precision and stable inversion

The experiment loads the full pretrained 26-layer Gemma 3 1B instruction-tuned
checkpoint twice: once as float32 and once as bfloat16. For each precision and
each of the five source prompts it greedily generates exactly 20 baseline
tokens.

At every token, a contextual forward pass records the target residual,
pre-MLP RMSNorm output, down-projection input/output, and layer output for the
last token. A reduced pass starts from only the last token at the same absolute
RoPE position. The full 100-token sweep applies each rank-1 matrix through its
exact low-rank action, avoiding an otherwise prohibitive outer-product
materialization on CPU. On the first source token in each precision, an
independent bounded check explicitly constructs `W + delta_W` at layer 0 and
records its difference from the low-rank action.

The naive route applies the paper's component-wise RMSNorm scale patch. The
stable route independently solves the Appendix-B scalar root in float64,
casts the target vector to the tested model dtype, applies the exact rank-1
down-projection action, and applies the component-wise remainder patch.
The root residual and target RMS are recorded for every layer and token.
If the paper's asserted root interval has no root, the implementation records
that fact and uses the exact trust-region hard-case solution at
`mu=min(m²)`; it never silently treats the missing root as an interior root.

For every row the run records baseline and patched token IDs, maximum absolute
logit error, total-variation distance, inversion residuals, and patch
diagnostics. Histories always advance with the baseline token, as required by
the paper. An unpatched reduced-context pass is the negative control.

The independent checker reloads the raw JSON, recomputes all aggregate
agreements and maxima from token rows, checks row counts and source prompt
identity, and fails on any disagreement with the stored summary.

An initial full-materialization attempt was deliberately cancelled after more
than six minutes without completing one precision token. Projecting that
method to all 100 tokens exceeded the fixed one-hour CPU job contract. It is
retained as a performance-bounded route, not mislabeled as completed evidence.
