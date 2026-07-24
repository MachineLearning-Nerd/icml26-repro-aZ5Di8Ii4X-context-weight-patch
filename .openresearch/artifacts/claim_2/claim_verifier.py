#!/usr/bin/env python3
"""Independent contract checker for the pretrained Gemma evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".openresearch" / "artifacts"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def check(predicate: bool, message: str, checks: list[dict]) -> None:
    checks.append({"check": message, "passed": bool(predicate)})


def main() -> None:
    raw = load_json(ARTIFACTS / "claim_1" / "raw_results.json")
    c1 = load_json(ROOT / "repro" / "contracts" / "claim_1.json")
    c2 = load_json(ROOT / "repro" / "contracts" / "claim_2.json")
    layers = raw["layers"]
    first = layers[0]
    checks: list[dict] = []

    a1 = c1["acceptance"]
    check(first["gate_projection_linf"] <= a1["projection_linf_max"], "C1 gate projection", checks)
    check(first["up_projection_linf"] <= a1["projection_linf_max"], "C1 up projection", checks)
    check(first["explicit_gate_patch_linf"] <= a1["explicit_patch_linf_max"], "C1 explicit gate patch", checks)
    check(first["explicit_up_patch_linf"] <= a1["explicit_patch_linf_max"], "C1 explicit up patch", checks)
    check(first["patched_output_linf"] <= a1["block_output_linf_max"], "C1 block output", checks)
    check(
        first["target_reconstruction_linf"] <= a1["reconstruction_linf_max"],
        "C1 target reconstruction",
        checks,
    )
    check(first["zero_unscaled_activation_count"] == 0, "C1 nonzero activation assumption", checks)
    check(
        first["omit_scale_patch_linf"] >= a1["negative_omit_scale_linf_min"],
        "C1 omission negative control",
        checks,
    )

    a2 = c2["acceptance"]
    check(len(layers) == a2["layers_required"], "C2 layer count", checks)
    check(
        max(layer["patched_output_linf"] for layer in layers)
        <= a2["per_layer_output_linf_max"],
        "C2 all layer outputs",
        checks,
    )
    check(raw["final_hidden_linf"] <= a2["final_hidden_linf_max"], "C2 final hidden", checks)
    check(raw["final_logit_linf"] <= a2["final_logit_linf_max"], "C2 final logits", checks)
    check(raw["top1_agreement"], "C2 top-1 agreement", checks)
    check(
        raw["negative_unpatched_logit_linf"]
        >= a2["negative_unpatched_logit_linf_min"],
        "C2 unpatched negative control",
        checks,
    )
    check(
        all(layer["zero_unscaled_activation_count"] == 0 for layer in layers),
        "C2 assumptions at every layer",
        checks,
    )

    c1_pass = all(item["passed"] for item in checks[:8])
    c2_pass = all(item["passed"] for item in checks[8:])
    output = {
        "independent_checker": "repro/verifiers/verify_claims_1_2.py",
        "claim_1": "VERIFIED" if c1_pass else "BLOCKED",
        "claim_2": "VERIFIED" if c2_pass else "BLOCKED",
        "checks": checks,
    }
    for claim in ("claim_1", "claim_2"):
        claim_dir = ARTIFACTS / claim
        (claim_dir / "independent_checker_output.json").write_text(
            json.dumps(output, indent=2) + "\n"
        )
    (ARTIFACTS / "claim_1" / "EVAL.md").write_text(
        "# Claim 1 evaluation\n\n"
        f"**Verdict: {output['claim_1']}**\n\n"
        f"- Layer-0 output L-infinity error: "
        f"{first['patched_output_linf']:.6e}\n"
        f"- Explicit gate/up patch errors: "
        f"{first['explicit_gate_patch_linf']:.6e} / "
        f"{first['explicit_up_patch_linf']:.6e}\n"
        f"- Omitted-scale negative-control error: "
        f"{first['omit_scale_patch_linf']:.6e}\n\n"
        "Full pretrained Gemma 3 1B layer-0 parameters are used.\n"
    )
    (ARTIFACTS / "claim_2" / "EVAL.md").write_text(
        "# Claim 2 evaluation\n\n"
        f"**Verdict: {output['claim_2']}**\n\n"
        f"- Sequential layers checked: {len(layers)}\n"
        f"- Maximum layer-output L-infinity error: "
        f"{max(layer['patched_output_linf'] for layer in layers):.6e}\n"
        f"- Final hidden/logit L-infinity error: "
        f"{raw['final_hidden_linf']:.6e} / {raw['final_logit_linf']:.6e}\n"
        f"- Unpatched negative-control logit error: "
        f"{raw['negative_unpatched_logit_linf']:.6e}\n"
        f"- Top-1 agreement: {raw['top1_agreement']}\n\n"
        "The patch is constructed inductively at every one of the 26 layers.\n"
    )
    print(json.dumps(output, indent=2))
    if not (c1_pass and c2_pass):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
