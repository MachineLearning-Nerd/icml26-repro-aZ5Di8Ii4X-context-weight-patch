# 2026-07-24 cumulative reproduction campaign

This additive campaign replaces the toy-only conclusions with direct evidence
from the exact Gemma 3 1B checkpoint and actual pinned Transformers classes.
The previous live judge score remains **2/12** until a new revision is
published with approval and judged.

| Claim | Result | Key observed evidence | Confidence |
| --- | --- | --- | --- |
| 1 | VERIFIED | full Gemma layer-0 output L∞ `7.629e-06` | HIGH |
| 2 | VERIFIED | all 26 layers; final-logit L∞ `2.480e-05`; same top-1 | HIGH |
| 3 | VERIFIED | 40 actual-class trials; max L∞ `4.441e-15` | MEDIUM |
| 4 | FALSIFIED | float32 and naive bfloat16 both `100/100`, not `100%/87.5%` | MEDIUM |
| 5 | FALSIFIED | naive and stable bfloat16 both `100/100` | MEDIUM |
| 6 | VERIFIED at architecture level | eight named class forms plus MoE/parallel routes | MEDIUM |

The fixed command for every experiment is:

```bash
uv run --frozen python repro/run_campaign.py
```

Final cumulative run:
`c73d8345-e83b-457b-872a-b4dc368cb2f7`, commit
`362a0a1f8c8d2d1099b8efe11022888816837654`, Hugging Face
`cpu-upgrade`, 58m12s, no GPU.

The complete machine-readable evidence is archived under
`.openresearch/artifacts/`; its top-level evaluation is
[EVAL.md](../../.openresearch/artifacts/EVAL.md).

No score increase is claimed here. Only the live judge can change the score.
