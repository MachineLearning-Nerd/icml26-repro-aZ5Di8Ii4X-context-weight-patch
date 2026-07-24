# Method: controllability and architecture coverage

The verifier imports the named implementations from the pinned Transformers
version and instantiates their real MLP/MoE modules in float64. For five seeds
per route it:

1. evaluates the contextual residual output;
2. applies the paper's rank-one patch to every input projection;
3. applies the output-controllability patch to the actual output projection;
4. evaluates the reduced-context residual output;
5. checks equality, projection identities, nonzero assumptions and rank-one
   structure; and
6. repeats the evaluation with the output patch omitted as a negative control.

Sequential gated and standard MLPs, Mixtral's routed sparse MoE, and GPT-J's
parallel block are separate code paths. GPT-2's `Conv1D` orientation is handled
explicitly. A second checker recomputes stored aggregates and independently
uses NumPy to check the core rank-one identities. Source-file hashes bind the
results to the exact installed implementation.

The reduced width is 256 with intermediate width 512. This avoids pretending
that randomly initialized multi-billion-parameter weights add scientific
information to an exact dimension-independent identity. The already-cumulative
Gemma 3 1B checkpoint checks provide the full pretrained anchor.

Command: `uv run --frozen python repro/run_campaign.py`

Seed base: 20260723; five consecutive seeds per route.
