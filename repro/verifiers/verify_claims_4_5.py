#!/usr/bin/env python3
"""Verify integrity and classify the deterministic Claims 4/5 evidence."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / ".openresearch" / "artifacts" / "claim_4" / "raw_results.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"BLOCKED: {message}")


def main() -> None:
    data = json.loads(RAW.read_text())
    rows = data["rows"]
    prompts = data["prompts"]
    require(len(prompts) == 5, "not all five source prompts were evaluated")
    require(data["tokens_per_prompt"] == 20, "token horizon is not 20")
    require(data["model"]["layers"] == 26, "not the full 26-layer model")
    require(data["negative_controls"]["minimum_logit_linf"] >= 0.01,
            "negative control did not separate")

    recomputed = {}
    for precision in ("float32", "bfloat16"):
        precision_rows = [row for row in rows if row["precision"] == precision]
        require(len(precision_rows) == 100,
                f"{precision} does not contain 100 rows")
        methods = ("naive",) if precision == "float32" else ("naive", "stable")
        for method in methods:
            key = f"{precision}_{method}"
            matches = [
                row["baseline_token_id"] == row[f"{method}_token_id"]
                for row in precision_rows
            ]
            recomputed[key] = {
                "token_count": len(matches),
                "matches": sum(matches),
                "agreement": sum(matches) / len(matches),
                "max_logit_linf": max(
                    row[f"{method}_logit_linf"] for row in precision_rows
                ),
                "max_tvd": max(row[f"{method}_tvd"] for row in precision_rows),
            }
            require(recomputed[key] == data["summary"][key],
                    f"stored {key} aggregate differs from token rows")

    max_constraint_error = max(
        row["stable_max_inversion_constraint_error"] for row in rows
    )
    require(max_constraint_error <= 1e-6,
            f"inversion constraint error {max_constraint_error} exceeds gate")

    fp32 = recomputed["float32_naive"]
    bf16 = recomputed["bfloat16_naive"]
    stable = recomputed["bfloat16_stable"]

    claim_4 = (
        "VERIFIED"
        if fp32["agreement"] == 1.0
        and fp32["max_logit_linf"] <= 1e-3
        and bf16["agreement"] == 0.875
        else "FALSIFIED"
    )
    if stable["agreement"] == 0.98:
        claim_5 = "VERIFIED_JUDGED_RENDERING"
    elif stable["agreement"] == 1.0:
        claim_5 = "VERIFIED_ARXIV_V3"
    elif stable["agreement"] > bf16["agreement"]:
        claim_5 = "FALSIFIED_PERCENTAGE_MECHANISM_ALIGNED"
    else:
        claim_5 = "FALSIFIED"

    output = {
        "claim_4": claim_4,
        "claim_5": claim_5,
        "float32_naive": fp32,
        "bfloat16_naive": bf16,
        "bfloat16_stable": stable,
        "max_inversion_constraint_error": max_constraint_error,
        "checks": "all evidence-integrity and fidelity gates passed",
    }
    for claim in (4, 5):
        path = ROOT / ".openresearch" / "artifacts" / f"claim_{claim}"
        (path / "independent_checker_output.json").write_text(
            json.dumps(output, indent=2) + "\n"
        )
    print("CLAIMS_4_5_VERDICT=" + json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
