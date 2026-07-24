- Previous live judged score: `2/12`
- Conservative projected score range after proposed change: `6/12–12/12`
- Best-supported possible new score, clearly labeled as a forecast rather than a judge result: `12/12` forecast

# Publication approval report

The current total score remains **2/12**. The projected range is not earned
credit: only a new live judge verdict can change the score.

Publication was explicitly approved and completed to the existing Space. The
new revision is `e2cc20512271c1bbbe2ee41865137c84af0c693f`; all 81 text paths
were downloaded from that exact revision and rechecked against the approved
manifest. The state is now **AWAITING JUDGE REEVALUATION**. No score increase
is claimed.

## Claim forecast

| Claim | Current points | Possible points | Confidence | Evidence status | Basis and remaining risk |
| --- | ---: | ---: | --- | --- | --- |
| 1 | 1 | 2 | HIGH | VERIFIED | Exact pretrained Gemma 3 1B layer; explicit rank-one matrix checker; output L∞ `7.629e-06`; unpatched control separates. Remaining scope is one checkpoint/target token. |
| 2 | 0 | 2 | HIGH | VERIFIED | All 26 sequential blocks, final hidden state, logits, and top-1 pass; final-logit L∞ `2.480e-05`, unpatched `23.491`. Construction remains token-specific. |
| 3 | 1 | 2 | MEDIUM | VERIFIED | Forty trials on actual pinned Transformers classes, independent NumPy checker, invalid-input and omitted-patch controls. Most classes use reduced random widths rather than pretrained weights. |
| 4 | 0 | 2 | MEDIUM | FALSIFIED | Two complete 100-token runs give float32 `100/100` and naive bfloat16 `100/100`, contradicting the stated 87.5%; explicit full-matrix audit is concordant. Risk: unavailable author code and CPU implementation semantics. |
| 5 | 0 | 2 | MEDIUM | FALSIFIED | Naive and stable bfloat16 both give `100/100`; stable lowers worst logit error `0.375→0.25`. Three materially different routes completed. Risk: the judged 98% endpoint conflicts with arXiv v3's 100% prose. |
| 6 | 0 | 2 | MEDIUM | VERIFIED | Actual class forms for Gemma, Llama, Falcon, Mistral, Mixtral, Qwen, GPT-2, and GPT-J pass sequential, MoE, Conv1D, and parallel routes. Evidence is architecture-level, not pretrained quality evidence for every family. |

All six claims changed materially since the previous judge result:

- Claims 1 and 3 move from toy-only evidence to direct full-model or
  actual-class evidence.
- Claim 2 now directly tests the 26-layer inductive theorem.
- Claims 4 and 5 now have exact percentage-contract falsifications.
- Claim 6 now instantiates every named architecture form.

No claim is BLOCKED. Claims 4 and 5 each completed the required three distinct
routes: full captured-action generation, explicit all-layer matrix
materialization, and an independent source/mathematical audit.

## Final cumulative result

- Winning branch:
  `orx/release-candidate-evidence-and-public-report`
- Git SHA:
  `362a0a1f8c8d2d1099b8efe11022888816837654`
- Final run:
  `c73d8345-e83b-457b-872a-b4dc368cb2f7`
- Fixed command:
  `uv run --frozen python repro/run_campaign.py`
- Environment:
  Python 3.12, one repository `.venv`, frozen `uv.lock`
- Backend:
  Hugging Face `cpu-upgrade`; no GPU
- Final run duration:
  58m12s
- Final regression:
  5/5 tests passed

Observed claim metrics:

| Claim | Terminal evidence |
| --- | --- |
| 1 | layer-0 output L∞ `7.62939453125e-06` |
| 2 | 26 layers; max layer L∞ `6.103515625e-05`; final-logit L∞ `2.47955322265625e-05`; same top-1 |
| 3 | 40 trials; max block L∞ `4.440892098500626e-15`; min control `2.4963252329265053` |
| 4 | float32 `100/100`; naive bfloat16 `100/100`, not 87.5% |
| 5 | naive/stable bfloat16 `100/100`; max logit L∞ `0.375/0.25`; constraint error `9.437e-15` |
| 6 | eight named classes; sequential, sparse-MoE, Conv1D, and parallel routes pass |
| Explicit control | 10/10 contextual/action/explicit tokens; max action-to-matrix logit L∞ `0.125`; min unpatched `18.703125` |

## Experiment tree and compute

The stacked path was:

1. frozen judged toy baseline;
2. exact Gemma block and 26-layer network;
3. precision-route bush with measured failed/cancelled approaches;
4. batched five-prompt precision winner;
5. exact joint verdict plus architecture conformance;
6. explicit bfloat16 matrix fidelity audit;
7. final evidence-and-report regression.

The frozen local baseline used 15 seconds. Formal Hugging Face CPU work totals
about **5h13m** including failed and cancelled routes; the final accepted run
used 58m12s. Hugging Face billing cost is not exposed in `orx` logs, so no
dollar cost is estimated. GPU runtime and GPU cost are both zero.

The complete command record is in the
[scientific command ledger](../command-ledger.md).
The illustrated report is
[here](../report.md).

## Evidence archive

The terminal run emitted 71 unique UTF-8 evidence files under
`.openresearch/artifacts/`. Each file was decoded from bounded log chunks and
validated against its emitted byte count and SHA-256. The CLI cache repeated
two complete Claim 6 records during range stitching; both duplicates were
hash-identical and were deduplicated by path.

Each claim directory contains its contract, exact source audit, method, raw
results, executable verifier, independent checker output, negative-control
output, environment, limitations, and `EVAL.md`. Claims 4 and 5 additionally
contain their explicit-materialization and three-route uncertainty evidence.

Candidate archive:
[`master/.openresearch/artifacts`](../../../.openresearch/artifacts/EVAL.md).

## Protected Space and publication check

- Existing Space: `DineshAI/aZ5Di8Ii4X`
- Previous judged HF Head: `e85dfc7775923513936705737b393955303db5f4`
- Published and verified HF Head: `e2cc20512271c1bbbe2ee41865137c84af0c693f`
- Current Judge Head: `e85dfc7775923513936705737b393955303db5f4`
- Judged paths: 21
- Missing judged paths in candidate: 0
- Byte-identical judged paths: 20
- Intentionally extended path: `logbook.json`
- All five old pages: byte-identical
- Exact old `logbook.json`: preserved at
  `provenance/judged-logbook-e85dfc7775923513936705737b393955303db5f4.json`
  with the judged SHA-256
  `7c2192996d7ffc9a779ffa300eb00013f845e6266a451bea196368041f0d49ae`

The candidate logbook has 14 reachable pages and no missing local evidence
links.

## Release gate

- Every claim has an honest VERIFIED or FALSIFIED result.
- Every cumulative regression check passes.
- Every current judge criticism is explicitly answered.
- Raw evidence regenerates from the fixed command.
- Negative controls fail as intended.
- Toy evidence remains labeled TOY.
- Every old judged page remains reachable and unchanged.
- `logbook.json` parses and every referenced page exists.
- The upload is an exact text-only allowlist.
- Secret-pattern scan: no hits across all 81 proposed paths.

Candidate totals:

- Upload paths: 81
- Upload bytes: 583,844
- Allowlist SHA-256:
  `75cc7e41c8272b6d84c532ef071b48afcc7098bd8207ba03cc5ca5780a9b4474`
- Upload-manifest SHA-256:
  `c341e9f1026e7af37c09747d1dd2c172e5f05b12dd08cba6202912247557df07`

Machine-readable checks:
[candidate summary](candidate-validation-summary.json),
[old/new subset proof](old-new-subset-check.json),
[secret scan](secret-scan.json), and
[SHA-256 manifest](upload-manifest.sha256.tsv).

## Exact text-only upload allowlist

The authoritative file is [upload-allowlist.txt](upload-allowlist.txt).

```text
.openresearch/artifacts/EVAL.md
.openresearch/artifacts/claim_1/EVAL.md
.openresearch/artifacts/claim_1/claim_contract.json
.openresearch/artifacts/claim_1/claim_verifier.py
.openresearch/artifacts/claim_1/environment.json
.openresearch/artifacts/claim_1/independent_checker_output.json
.openresearch/artifacts/claim_1/limitations.md
.openresearch/artifacts/claim_1/method.md
.openresearch/artifacts/claim_1/negative_control_output.json
.openresearch/artifacts/claim_1/raw_results.json
.openresearch/artifacts/claim_1/source_audit.md
.openresearch/artifacts/claim_2/EVAL.md
.openresearch/artifacts/claim_2/claim_contract.json
.openresearch/artifacts/claim_2/claim_verifier.py
.openresearch/artifacts/claim_2/environment.json
.openresearch/artifacts/claim_2/independent_checker_output.json
.openresearch/artifacts/claim_2/limitations.md
.openresearch/artifacts/claim_2/method.md
.openresearch/artifacts/claim_2/negative_control_output.json
.openresearch/artifacts/claim_2/raw_results.json
.openresearch/artifacts/claim_2/source_audit.md
.openresearch/artifacts/claim_3/EVAL.md
.openresearch/artifacts/claim_3/claim_contract.json
.openresearch/artifacts/claim_3/claim_verifier.py
.openresearch/artifacts/claim_3/environment.json
.openresearch/artifacts/claim_3/independent_checker_output.json
.openresearch/artifacts/claim_3/limitations.md
.openresearch/artifacts/claim_3/method.md
.openresearch/artifacts/claim_3/negative_control_output.json
.openresearch/artifacts/claim_3/raw_results.json
.openresearch/artifacts/claim_3/source_audit.md
.openresearch/artifacts/claim_4/EVAL.md
.openresearch/artifacts/claim_4/claim_contract.json
.openresearch/artifacts/claim_4/claim_verifier.py
.openresearch/artifacts/claim_4/environment.json
.openresearch/artifacts/claim_4/explicit_materialization_checker_output.json
.openresearch/artifacts/claim_4/explicit_materialization_contract.json
.openresearch/artifacts/claim_4/explicit_materialization_method.md
.openresearch/artifacts/claim_4/explicit_materialization_results.json
.openresearch/artifacts/claim_4/independent_checker_output.json
.openresearch/artifacts/claim_4/limitations.md
.openresearch/artifacts/claim_4/method.md
.openresearch/artifacts/claim_4/negative_control_output.json
.openresearch/artifacts/claim_4/raw_results.json
.openresearch/artifacts/claim_4/source_audit.md
.openresearch/artifacts/claim_4/three_route_uncertainty_audit.md
.openresearch/artifacts/claim_5/EVAL.md
.openresearch/artifacts/claim_5/claim_contract.json
.openresearch/artifacts/claim_5/claim_verifier.py
.openresearch/artifacts/claim_5/environment.json
.openresearch/artifacts/claim_5/explicit_materialization_checker_output.json
.openresearch/artifacts/claim_5/explicit_materialization_contract.json
.openresearch/artifacts/claim_5/explicit_materialization_method.md
.openresearch/artifacts/claim_5/explicit_materialization_results.json
.openresearch/artifacts/claim_5/independent_checker_output.json
.openresearch/artifacts/claim_5/limitations.md
.openresearch/artifacts/claim_5/method.md
.openresearch/artifacts/claim_5/negative_control_output.json
.openresearch/artifacts/claim_5/raw_results.json
.openresearch/artifacts/claim_5/source_audit.md
.openresearch/artifacts/claim_5/three_route_uncertainty_audit.md
.openresearch/artifacts/claim_6/EVAL.md
.openresearch/artifacts/claim_6/claim_contract.json
.openresearch/artifacts/claim_6/claim_verifier.py
.openresearch/artifacts/claim_6/environment.json
.openresearch/artifacts/claim_6/independent_checker_output.json
.openresearch/artifacts/claim_6/limitations.md
.openresearch/artifacts/claim_6/method.md
.openresearch/artifacts/claim_6/negative_control_output.json
.openresearch/artifacts/claim_6/raw_results.json
.openresearch/artifacts/claim_6/source_audit.md
logbook.json
pages/campaign-2026-07-24/page.md
pages/claim-1-full-gemma/page.md
pages/claim-2-full-network/page.md
pages/claim-3-real-components/page.md
pages/claim-4-precision/page.md
pages/claim-5-stable-inversion/page.md
pages/claim-6-architectures/page.md
pages/source-audit-limitations/page.md
provenance/judged-logbook-e85dfc7775923513936705737b393955303db5f4.json
```

## Approved action and awaiting-judge state

The approved action was:

1. upload the 81 allowlisted text files to the existing Space
   `DineshAI/aZ5Di8Ii4X` using the text-only Hugging Face API path;
2. verify and report the resulting exact HF revision;
3. mark the paper awaiting judge without claiming a score increase; and
4. mirror the exact published text paths plus the final report, README, and
   marimo notebook to GitHub `master`, then verify the remote Git SHA with
   `git ls-remote`.

The Space upload and hash verification are complete. The exact publication
record is [published-revision.json](published-revision.json). No second Space
was created and no GPU was used. The live judge remains on the previous
revision and the live score remains 2/12.
