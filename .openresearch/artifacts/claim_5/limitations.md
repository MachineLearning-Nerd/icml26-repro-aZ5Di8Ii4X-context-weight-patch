# Limitations and deviations

- The full Gemma 3 1B model is tested; the paper also mentions 4B.
- Exact prompt strings are recovered from the paper source PDFs.
- The 20-token horizon is recovered from the plotted Mars panel; the prose does not state it.
- The stable scalar root is solved in float64 before its target is cast to the tested dtype; the model and low-rank actions use the declared float32 or bfloat16 precision.
- The 100-token sweep uses captured exact target actions. Explicit materialization is bounded to layer 0 of the first token in each precision after full-materialization and repeated-matvec CPU attempts exceeded their preregistered runtime contract.
- The authoritative v3 and ar5iv/judge percentages disagree; both are classified separately.
