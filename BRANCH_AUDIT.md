# Branch audit

## Final branch policy

- main is the publication branch.
- Evidence branches preserve the cumulative experiment snapshots with
  descriptive names.
- Duplicate tips are retained when the original workflow created separate
  experiment names for the same snapshot; their distinct provenance remains
  inspectable.
- No branch named or prefixed orx remains in the final remote state.

## Original-to-final mapping

| Final branch | Original branch | Original tip before audit | Scope |
|---|---|---|---|
| main | master | e5c5e06c0e09a1633881e5a547b0d57bd0eff3bb | Combined publication surface |
| evidence/architecture-matrix | orx/architecture-conformance-matrix | 384f786183bdd77eb884e1450646dd9869903187 | Initial architecture scaffold |
| evidence/baseline-toy | orx/baseline-judged-toy-reproduction | 384f786183bdd77eb884e1450646dd9869903187 | Five-seed rank-one toy baseline |
| evidence/precision-sweep | orx/batched-five-prompt-precision-sweep | 97cadbc754c5c7e02afadb1a0e6f7eaa087d7f12 | Batched five-prompt precision sweep |
| evidence/precision-verdict-architecture | orx/exact-precision-verdict-plus-architecture-confor | 7494d8f8992a350466f39f65882dc98af672028f | Precision contract and architecture checks |
| evidence/explicit-bfloat16-audit | orx/explicit-bfloat16-matrix-materialization-audit | d48d20beaca320fbae5bec711eb0da57d9a5f084 | Explicit bfloat16 matrix fidelity |
| evidence/stable-inversion | orx/gemma-generation-precision-and-stable-inversion | 51a65de78bc91954ff855c0246b937cc09da8d1c | Stable inversion and captured generation |
| evidence/precision-cross-check | orx/precision-functional-cross-check-and-source-cons | bdda1cbf9f7709fab65ab7a792596dd07cd8889c | Precision setup and evidence metadata |
| evidence/gemma-block-multilayer | orx/real-gemma-3-1b-block-and-multilayer | bdda1cbf9f7709fab65ab7a792596dd07cd8889c | Full Gemma block and multilayer checks |
| evidence/release-candidate | orx/release-candidate-evidence-and-public-report | 362a0a1f8c8d2d1099b8efe11022888816837654 | Cumulative release candidate |

The final main branch adds the paper-first audit documents to the cumulative
release candidate. The evidence branches intentionally retain their historical
snapshot content.

## Branch contents

- main contains the combined code, committed evidence, report, citation,
  source manifest, branch map, and final verifier.
- evidence/baseline-toy and evidence/architecture-matrix preserve the initial
  toy and architecture setup.
- evidence/gemma-block-multilayer preserves the first full Gemma evidence.
- evidence/precision-cross-check and evidence/stable-inversion preserve the
  precision implementation lineage.
- evidence/precision-sweep and evidence/precision-verdict-architecture
  preserve the complete precision sweep and its contract verdict.
- evidence/explicit-bfloat16-audit preserves the explicit matrix audit.
- evidence/release-candidate preserves the cumulative pre-publication snapshot.
