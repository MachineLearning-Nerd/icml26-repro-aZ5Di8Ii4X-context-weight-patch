# Explicit bfloat16 materialization audit

The main Section 4 reproduction evaluates all five paper prompts for twenty
generated tokens, but accelerates each input rank-one update by applying its
known action to the current vector. That is algebraically exact but can omit
rounding introduced by constructing `W + delta_W` in bfloat16.

This audit uses the same checkpoint and the first generated token of every
paper prompt. At all 26 layers and separately for naive and stable methods it
constructs the full rank-one gate/up matrices, adds them to the pretrained
bfloat16 weights, and invokes the actual linear operator. The stable method
also materializes its down-projection update. Real reduced-context attention,
RMSNorm, final normalization, and the language-model head are unchanged.

The audit compares:

- contextual versus explicitly patched logits and tokens;
- captured-action versus explicitly materialized logits and tokens;
- each materialized projection against its contextual target; and
- contextual logits against an unpatched reduced-context control.

Five tokens cannot estimate the paper's 100-token percentage. Their role is to
test the principal fidelity risk in the faster complete sweep.

Command: `uv run --frozen python repro/run_campaign.py`
