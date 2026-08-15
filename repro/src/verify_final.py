"""Fail-closed verification of the committed ICML 2026 audit snapshot."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    patch = read_json("outputs/patch_summary.json")
    require(len(patch["trials"]) == 5, "toy trial count changed")
    require(patch["C1"]["all_exact"] is True, "toy exact identity failed")
    require(patch["C1"]["all_rank1"] is True, "toy rank-one gate failed")
    require(patch["control"]["mismatch_confirmed"] is True, "toy negative control failed")

    claim_1 = read_json(".openresearch/artifacts/claim_1/raw_results.json")
    require(claim_1["model"]["layers"] == 26, "Gemma layer count changed")
    require(claim_1["max_layer_output_linf"] < 0.001, "C1 output error changed")
    require(claim_1["negative_unpatched_logit_linf"] > 1.0, "C1 negative control changed")
    claim_2 = read_json(".openresearch/artifacts/claim_2/raw_results.json")
    require(len(claim_2["layers"]) == 26, "C2 layer evidence changed")
    require(claim_2["top1_agreement"] is True, "C2 top-1 agreement failed")
    require(claim_2["final_logit_linf"] < 0.1, "C2 logit error changed")

    architecture = read_json(".openresearch/artifacts/claim_3/raw_results.json")
    require(len(architecture["rows"]) == 40, "architecture trial count changed")
    required_classes = {
        "Gemma3MLP",
        "LlamaMLP",
        "MistralMLP",
        "FalconMLP",
        "Qwen2MLP",
        "GPT2MLP",
        "MixtralSparseMoeBlock",
        "GPTJBlock",
    }
    require(required_classes <= set(architecture["class_sources"]), "architecture route missing")
    require(all(architecture["topology_checks"].values()), "topology check failed")
    require(architecture["zero_assumption_control"]["rejected"] is True, "zero assumption control changed")
    require(max(row["block_linf"] for row in architecture["rows"]) < 1e-10, "architecture error changed")

    precision = read_json(".openresearch/artifacts/claim_4/raw_results.json")
    summary = precision["summary"]
    require(summary["float32_naive"]["matches"] == 100, "float32 token count changed")
    require(summary["bfloat16_naive"]["matches"] == 100, "naive bfloat16 result changed")
    require(summary["bfloat16_stable"]["matches"] == 100, "stable bfloat16 result changed")
    require(summary["bfloat16_naive"]["max_logit_linf"] > 0.1, "naive precision control changed")
    require(summary["bfloat16_stable"]["max_logit_linf"] < summary["bfloat16_naive"]["max_logit_linf"], "stable error reduction disappeared")

    for claim in (4, 5):
        explicit = read_json(
            f".openresearch/artifacts/claim_{claim}/explicit_materialization_results.json"
        )
        require(len(explicit["rows"]) == 10, f"C{claim} explicit audit size changed")
        require(min(explicit["negative_controls"]) > 1.0, f"C{claim} explicit control changed")

    summary_manifest = read_json("evidence/claim_summary.json")
    expected_statuses = {
        "C1": "VERIFIED_SCOPED",
        "C2": "VERIFIED_SCOPED",
        "C3": "VERIFIED_ARCHITECTURE_SCOPED",
        "C4": "FALSIFIED_SCOPED",
        "C5": "FALSIFIED_SCOPED",
        "C6": "VERIFIED_ARCHITECTURE_SCOPED",
        "toy_baseline": "VERIFIED_TOY",
        "full_paper_experiments": "NOT_REPRODUCED",
    }
    for claim, status in expected_statuses.items():
        require(summary_manifest["claims"][claim]["status"] == status, f"scope status changed: {claim}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for required in (
        "icml26-context-parameter-equivalence",
        "evidence/baseline-toy",
        "evidence/release-candidate",
        "FALSIFIED_SCOPED",
        "CITATION.cff",
        "Thank you",
    ):
        require(required in readme, f"README requirement missing: {required}")
    require("tree/orx/" not in readme, "README contains an old branch URL")

    logbook = read_json("logbook.json")
    for page in logbook["root"]["children"]:
        require((ROOT / page["file"]).is_file(), f"logbook page is missing: {page['file']}")
    require((ROOT / "pages/index.md").is_file(), "logbook index page is missing")

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    require('name = "icml26-context-parameter-equivalence"' in pyproject, "final package name missing")
    test_file = (ROOT / "repro/tests/test_rmsnorm_inverse.py").read_text(encoding="utf-8")
    require("parents[2]" in test_file, "clean-checkout test import fix missing")
    require(not (ROOT / ".trackio").exists(), "generated Trackio cache remains")

    tracked = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True
    ).splitlines()
    require(not any(path.startswith(".trackio/") for path in tracked), "tracked Trackio file remains")
    require(not any(path.endswith(".log") for path in tracked), "tracked execution log remains")

    print("PASS: scoped claims, negative precision result, source evidence, branch docs, and cleanup invariants")


if __name__ == "__main__":
    main()
