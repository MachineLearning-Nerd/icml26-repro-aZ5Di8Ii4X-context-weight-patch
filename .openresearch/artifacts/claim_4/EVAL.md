# Claim 4 evaluation

**Verdict: FALSIFIED**

- Float32 naive agreement: 100/100 (100.0%)
- Bfloat16 naive agreement: 100/100 (100.0%)
- Bfloat16 stable agreement: 100/100 (100.0%)
- Float32 maximum logit L-infinity error: 2.670288e-05
- Bfloat16 naive/stable maximum logit L-infinity error: 3.750000e-01 / 2.500000e-01

The verdict applies to the exact joint percentage statement. Stable inversion reduced worst logit error but did not increase token agreement because the naive method was already 100%.
