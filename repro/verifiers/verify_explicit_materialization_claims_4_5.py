#!/usr/bin/env python3
"""Verify integrity and classify the explicit materialization audit."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW = (
    ROOT
    / ".openresearch"
    / "artifacts"
    / "claim_4"
    / "explicit_materialization_results.json"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"BLOCKED: {message}")


def main() -> None:
    data = json.loads(RAW.read_text())
    rows = data["rows"]
    require(data["model"]["layers"] == 26, "not all model layers are present")
    require(data["model"]["dtype"] == "torch.bfloat16", "not bfloat16")
    require(len(data["prompts"]) == 5, "not all five prompts are present")
    require(len(rows) == 10, "expected five prompts times two methods")
    require(
        {(row["prompt_index"], row["method"]) for row in rows}
        == {(index, method) for index in range(5) for method in ("naive", "stable")},
        "prompt/method grid is incomplete",
    )
    require(
        all(row["layers_materialized"] == 26 for row in rows),
        "a route did not materialize all layers",
    )
    require(
        min(data["negative_controls"]) >= 0.01,
        "unpatched negative control did not separate",
    )

    explicit_matches = sum(
        row["baseline_token_id"] == row["explicit_token_id"] for row in rows
    )
    action_matches = sum(
        row["action_token_id"] == row["explicit_token_id"] for row in rows
    )
    max_action_explicit = max(
        row["action_to_explicit_logit_linf"] for row in rows
    )
    concordant = (
        explicit_matches == len(rows)
        and action_matches == len(rows)
        and max_action_explicit <= 1.0
    )
    output = {
        "route_assessment": "CONCORDANT" if concordant else "DIVERGENT",
        "rows": len(rows),
        "explicit_contextual_token_matches": explicit_matches,
        "action_explicit_token_matches": action_matches,
        "max_action_to_explicit_logit_linf": max_action_explicit,
        "max_baseline_to_explicit_logit_linf": max(
            row["baseline_to_explicit_logit_linf"] for row in rows
        ),
        "max_gate_target_linf": max(
            row["max_gate_target_linf"] for row in rows
        ),
        "max_up_target_linf": max(row["max_up_target_linf"] for row in rows),
        "max_down_target_linf": max(
            row["max_down_target_linf"] for row in rows
        ),
        "min_negative_unpatched_linf": min(data["negative_controls"]),
        "scope": (
            "Fidelity audit only; five tokens do not estimate a 100-token "
            "agreement percentage."
        ),
    }
    for claim in (4, 5):
        path = ROOT / ".openresearch" / "artifacts" / f"claim_{claim}"
        (path / "explicit_materialization_checker_output.json").write_text(
            json.dumps(output, indent=2) + "\n"
        )
    print("EXPLICIT_MATERIALIZATION_VERDICT=" + json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
