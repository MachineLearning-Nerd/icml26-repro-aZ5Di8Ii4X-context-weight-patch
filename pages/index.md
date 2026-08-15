# ICML 2026 context–parameter equivalence audit

This logbook records a scoped audit of
[arXiv:2511.17864v3](https://arxiv.org/abs/2511.17864v3). The core rank-one
construction and its multilayer extension pass on one real Gemma 3 1B
checkpoint. The exact reported bfloat16 percentage contrast does not appear in
the pinned CPU campaign: naive and stable bfloat16 both match 100/100 tokens.

| Claim | Status |
|---|---|
| C1 Gemma block | VERIFIED_SCOPED |
| C2 multilayer Gemma | VERIFIED_SCOPED |
| C3 controllability | VERIFIED_ARCHITECTURE_SCOPED |
| C4 naive bfloat16 percentage | FALSIFIED_SCOPED |
| C5 stable bfloat16 percentage improvement | FALSIFIED_SCOPED |
| C6 architecture families | VERIFIED_ARCHITECTURE_SCOPED |

Read the [claim evidence ledger](../../CLAIM_EVIDENCE.md) for production paths,
boundaries, controls, and durable artifact links.
