# Claim 6 — named architecture families

**Result: VERIFIED at architecture level · MEDIUM confidence**

The construction was executed on actual pinned class forms for Gemma, Llama,
Falcon, Mistral, Mixtral, Qwen, GPT-2, and GPT-J.

| Route | Coverage |
| --- | --- |
| sequential gated MLP | Gemma, Llama, Mistral, Qwen |
| sequential dense/Conv1D | Falcon, GPT-2 |
| sparse routed MoE | Mixtral router plus active experts |
| parallel residual | GPT-J |

Across 40 seeded trials, maximum block L∞ is
`4.440892098500626e-15`; every omitted-patch control is at least
`2.4963252329265053`. Mixtral uses the actual rounded active gate sum rather
than assuming it is exactly one. GPT-2 validates Conv1D orientation and GPT-J
validates the parallel-sum topology.

The non-Gemma modules use hidden width 256, intermediate width 512, and random
deterministic weights. This verifies the architecture-specific algebra, not
pretrained language quality for every named family.

Evidence:
[contract](../../.openresearch/artifacts/claim_6/claim_contract.json),
[source audit](../../.openresearch/artifacts/claim_6/source_audit.md),
[method](../../.openresearch/artifacts/claim_6/method.md),
[raw results](../../.openresearch/artifacts/claim_6/raw_results.json),
[verifier](../../.openresearch/artifacts/claim_6/claim_verifier.py),
[independent checker](../../.openresearch/artifacts/claim_6/independent_checker_output.json),
[negative control](../../.openresearch/artifacts/claim_6/negative_control_output.json),
and [evaluation](../../.openresearch/artifacts/claim_6/EVAL.md).
