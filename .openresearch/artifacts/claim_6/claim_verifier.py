#!/usr/bin/env python3
"""Independently verify Claims 3/6 architecture-conformance evidence."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / ".openresearch" / "artifacts" / "claim_3" / "raw_results.json"
REQUIRED = {
    "gemma",
    "llama",
    "falcon",
    "mistral",
    "mixtral",
    "qwen",
    "gpt2",
    "gptj",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"BLOCKED: {message}")


def main() -> None:
    data = json.loads(RAW.read_text())
    rows = data["rows"]
    counts = Counter(row["architecture"] for row in rows)
    require(set(counts) == REQUIRED, "named architecture coverage is incomplete")
    require(all(counts[name] == 5 for name in REQUIRED), "not five seeds per route")
    require(data["dtype"] == "float64", "dynamic checks are not float64")
    require(data["hidden_size"] == 256, "unexpected hidden size")
    require(
        all(data["topology_checks"].values()),
        "an actual implementation topology check failed",
    )
    require(
        data["zero_assumption_control"]["rejected"],
        "zero-input assumption control was not rejected",
    )
    require(
        all(item["source_sha256"] for item in data["class_sources"].values()),
        "class source hash missing",
    )

    max_block = max(row["block_linf"] for row in rows)
    max_input = max(row["input_projection_linf"] for row in rows)
    min_negative = min(row["negative_control_linf"] for row in rows)
    min_activation = min(row["outer_activation_norm"] for row in rows)
    min_gate_sum = min(
        row.get("active_gate_sum", float("inf")) for row in rows
    )
    require(max_block <= 1e-10, f"block error {max_block} exceeds contract")
    require(max_input <= 1e-10, f"input error {max_input} exceeds contract")
    require(
        min_negative >= 1e-4,
        f"negative control {min_negative} did not separate",
    )
    require(min_activation > 0.0, "zero outer activation entered a valid case")
    require(min_gate_sum > 0.0, "MoE gate-sum assumption failed")
    require(
        all(rank == 1 for row in rows for rank in row["patch_ranks"]),
        "a constructed rank-one factor was degenerate",
    )

    independent = data["independent_numpy_cases"]
    require(len(independent) == 10, "independent NumPy case count differs")
    numpy_max = max(case["projection_linf"] for case in independent)
    numpy_min_negative = min(
        case["negative_control_linf"] for case in independent
    )
    require(numpy_max <= 1e-10, "independent rank-one identity failed")
    require(
        numpy_min_negative >= 1e-4,
        "independent negative control did not separate",
    )

    output = {
        "claim_3": "VERIFIED",
        "claim_6": "VERIFIED",
        "architectures": sorted(REQUIRED),
        "moe_covered": True,
        "seeds_per_route": 5,
        "max_block_linf": max_block,
        "max_input_projection_linf": max_input,
        "min_negative_control_linf": min_negative,
        "independent_numpy_max_linf": numpy_max,
        "zero_assumption_control": data["zero_assumption_control"],
        "checks": "all contract, topology, assumption and control gates passed",
        "limitation": (
            "Non-Gemma dynamic modules use reduced widths and deterministic "
            "random weights; the result is architecture-level conformance."
        ),
    }
    for claim in (3, 6):
        path = ROOT / ".openresearch" / "artifacts" / f"claim_{claim}"
        (path / "independent_checker_output.json").write_text(
            json.dumps(output, indent=2) + "\n"
        )
        (path / "EVAL.md").write_text(
            f"# Claim {claim} evaluation\n\n"
            f"**Verdict: {output[f'claim_{claim}']}**\n\n"
            f"- Maximum block L-infinity error: {max_block:.6e}\n"
            f"- Maximum input-projection error: {max_input:.6e}\n"
            f"- Minimum negative-control error: {min_negative:.6e}\n"
            f"- Independent NumPy maximum error: {numpy_max:.6e}\n"
            "- All actual-class topology and nonzero-assumption gates passed.\n\n"
            f"Limitation: {output['limitation']}\n"
        )
    print("CLAIMS_3_6_VERDICT=" + json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
