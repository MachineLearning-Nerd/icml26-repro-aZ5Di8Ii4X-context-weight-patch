# Source audit: Claims 3 and 6

## Authoritative source

- URL: `https://export.arxiv.org/e-print/2511.17864v3`
- Retrieved: 2026-07-23 with explicit browser User-Agent
- Source archive SHA-256: `c90b94ebadaf527640c52ed61e4de497ae8ebff7ab449c47c8939584ef3a74d3`
- TeX anchors: `dfn:input-controllability`, `dfn:output-controllability`,
  `thm:unified_residual`, `tab:results`,
  `lem:input_controllability_mlp`, `lem:input_controllability_norm`,
  `lem:output_bias`, `lem:output_weight`,
  `lem:output_elem_multiply`, `lem:output_moe`, and
  `lem:parallel_block`.

## Exact scope and quantifiers

Definition 3 quantifies over every nonzero input pair. Definition 4 quantifies
over every fixed nonzero inner activation and every desired output delta.
Theorem 5 is conditional: it concludes existence of a perfect residual-block
patch only when both controllability properties and the nonzero-activation
assumptions hold.

Table 1 lists algebraic component forms. The surrounding paragraph says those
forms encompass Gemma, Llama, Falcon, Mistral/Mixtral, Qwen, GPT-2 and GPT-J.
It does **not** say that the paper empirically ran all of those pretrained
models; Section 4 reports Gemma 3 and Falcon experiments. Claim 6 is therefore
audited as architectural conformance, not model-quality equivalence.

## Important interpretation constraints

- GPT-J is a parallel block: its MLP input is context-independent, so only the
  output projection must absorb the attention delta.
- Mixtral requires both router/input controllability and the MoE output lemma.
  The active top-k weights are normalized in the actual implementation, hence
  their sum is nonzero and normally equals one.
- GPT-2 uses `Conv1D`, whose stored weight orientation is the transpose of
  `torch.nn.Linear`; the same rank-one map must be applied in that orientation.
- Zero inputs and zero inner activations are outside the theorem. They are
  explicit assumption controls, not counterexamples.

## Deviation

Except for the cumulative full pretrained Gemma 3 1B checks, the dynamic
architecture suite instantiates the actual Transformers classes at reduced
hidden width so all named forms fit a CPU campaign. This is a faithful test of
the dimension-independent algebra and executable topology, but not a
pretrained end-to-end evaluation of every named model.
