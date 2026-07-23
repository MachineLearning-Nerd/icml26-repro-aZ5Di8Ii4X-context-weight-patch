# Source audit: Claims 4 and 5

## Sources and hashes

- Authoritative TeX: `https://export.arxiv.org/e-print/2511.17864v3`,
  retrieved with an explicit browser User-Agent on 2026-07-23 at 14:15:31Z.
  Archive SHA-256:
  `c90b94eb23cda27c53818ef0a4a1aac80198aaace336e9a720d8a3325a718af6`.
- ar5iv HTML: `https://ar5iv.labs.arxiv.org/html/2511.17864`, retrieved
  with an explicit browser User-Agent on 2026-07-23 at 14:12:19Z. SHA-256:
  `b00456cc4aa6773fb0b2ec09a891c69f8b4bf2ac8072ec2c768a6821e752126c`.

## Exact statements and revision discrepancy

Section 4 evaluates an instruction-tuned Gemma 3 1B and 4B model using greedy
token generation. The contextual baseline receives the full prompt and
generated history. For every new token, the patched model receives no explicit
history except the current token at its absolute position; all layer patches
are recomputed. After a top-1 mismatch, subsequent comparisons are forced onto
the baseline history.

The authoritative arXiv v3 TeX says naive bfloat16 has 87.5% agreement and the
stable update raises this to 100%. The current ar5iv rendering, and therefore
the judged claim, says 98% for the stable update. Both interpretations are
retained and evaluated. Neither percentage is silently substituted.

The supplied v3 `generation_metrics.pdf` contains 20 Mars tokens and marks two
naive-bfloat16 mismatches, which is 90%, not 87.5%. Its aggregate summary bar
also appears above 95%. This is an internal source inconsistency, not an
experimental result.

## Recovered quantifiers and assumptions

- Five text prompts are used: the Mars prompt in Section 4 plus four exact
  prompts embedded in the source PDFs.
- The per-prompt figure establishes a 20-token horizon. The prose does not
  state the horizon independently.
- Generation uses argmax. No sampling uncertainty is involved.
- Patches are token/history-specific and are recomputed at every token.
- The paper says division by zero is replaced by division by one.
- The stable construction fixes the target pre-normalization RMS, solves the
  constrained RMSNorm inverse, patches the down projection by rank one, and
  assigns the remaining component-wise error to the trainable scale vector.

## Exact five prompts

1. `Write a single-sentence weather forecast for Mars, from the perspective of a slightly annoyed robot:`
2. `Constraint: No 'e's. Topic: The meaning of life. Length: Exactly 20 words. Write:`
3. `Explain what a 'bug' is in computer programming, but explain it to a 16th-century medieval peasant. Use 20 words or less:`
4. `Topic: Three siblings named Jack, Fred, and Alice. Task: Write one 5-7-5 haiku about them:`
5. `We can absorb the impact of a prompt for a modern language model such as Gemma into the MLP weights. Write a two-line rhyming couplet about this fact:`

## Model deviation

The official `google/gemma-3-1b-it` repository is gated. The run uses the public
mirror `unsloth/gemma-3-1b-it` at a pinned revision. Its single model weight
file has the exact same Hugging Face blob ID and byte size as the official
checkpoint: `cf8782aa8cd58000ba309c9f5ac84a810c3bf7bf`,
1,999,811,208 bytes.
