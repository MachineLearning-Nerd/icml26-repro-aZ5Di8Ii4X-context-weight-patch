# Cumulative evaluation

**Baseline:** TOY checks retained and passing.

**Claim 1: VERIFIED in this experiment.** The exact pretrained Gemma 3 1B layer-0 contract and independent explicit-patch check pass.

**Claim 2: VERIFIED in this experiment.** All 26 sequential layer contracts, final hidden state, logits, and top-1 prediction pass.

**Claim 3: VERIFIED in this experiment.** The exact conditional controllability construction passes on actual Transformers classes, including assumption and negative controls.

**Claims 4 and 5: FALSIFIED in this experiment.** Float32, naive bfloat16 and stable bfloat16 each agreed on 100/100 tokens. This contradicts the exact 87.5% contrast and 87.5%-to-endpoint joint improvement, although stable reduced worst logit error.

**Claim 6: VERIFIED at architecture level in this experiment.** Every named class form, including Mixtral MoE and GPT-J parallel blocks, passes the construction. Non-Gemma modules are reduced-width random initializations; this is not pretrained quality evidence.

- Git SHA: `362a0a1f8c8d2d1099b8efe11022888816837654`
- Lock SHA-256: `b53e0b2a191fa07534677470c9816c8d8164da5d72df922a410267307a86a140`
- Runtime: 3449.401 seconds
- Tests: passed
