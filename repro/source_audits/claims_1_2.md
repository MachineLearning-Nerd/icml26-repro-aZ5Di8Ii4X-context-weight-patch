# Source audit — Claims 1 and 2

Audited 2026-07-23 UTC with an explicit
`OpenResearch-Reproduction/1.0` User-Agent.

- Authoritative source archive:
  `https://export.arxiv.org/e-print/2511.17864v3`
- Archive SHA-256:
  `c90b94eb23cda27c53818ef0a4a1aac80198aaace336e9a720d8a3325a718af6`
- Primary HTML candidate:
  `https://ar5iv.labs.arxiv.org/html/2511.17864`
- HTML SHA-256:
  `b00456cc4aa6773fb0b2ec09a891c69f8b4bf2ac8072ec2c768a6821e752126c`

## Claim 1

Theorem 1 is `paper.tex` label `thm:gemma_block` and HTML anchor
`#Thmtheorem1` under Section 3.1 (`#S3.SS1`). It quantifies over a
full-context attention state `v_C` and reduced-context state `v`, and states
that the full-context block output can be *perfectly* replicated by the
specified updates. Its explicit coordinate-wise precondition is that the
unscaled post-MLP activation is nonzero in every coordinate. The rank-1
construction also requires `||z||² > 0`.

The experiment uses the exact 1,999,811,208-byte safetensors blob shared by
`google/gemma-3-1b-it` and the ungated `unsloth/gemma-3-1b-it` mirror. The
mirror is pinned by revision and the shared blob ID is recorded in the
contract. This is a pretrained 26-layer Gemma 3 1B model, not a random or
reduced-width block.

## Claim 2

Theorem 2 is `paper.tex` label `thm:multilayer` and HTML anchor
`#Thmtheorem2` under Section 3.2 (`#S3.SS2`). It states existence for an
L-layer transformer whose blocks have the Theorem 1 structure, *assuming the
Theorem 1 conditions at every layer*. Algorithm 1 (`alg:multilayer`) makes the
construction sequential and self-correcting: each next patch uses the actual
output produced after the previous patch.

The quantifier is token/history specific. The paper explicitly says the
patches are recomputed for each generated token and does not claim a single
global patch that works for arbitrary future queries.

