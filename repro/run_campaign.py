#!/usr/bin/env python3
"""Fixed entry point for cumulative claim checks.

The baseline intentionally runs only the pre-existing toy reproduction. Child
experiments extend this runner while the OpenResearch run command stays fixed.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / ".openresearch" / "artifacts"
BASELINE = ARTIFACTS / "baseline"


def run_checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(f"$ {' '.join(command)}", flush=True)
    print(completed.stdout, end="", flush=True)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    return completed


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    started = time.perf_counter()
    BASELINE.mkdir(parents=True, exist_ok=True)

    script = run_checked([sys.executable, "repro/src/run_patch.py"])
    gemma = run_checked([sys.executable, "repro/src/run_gemma_claims_1_2.py"])
    verifier = run_checked(
        [sys.executable, "repro/verifiers/verify_claims_1_2.py"]
    )
    precision = run_checked(
        [sys.executable, "repro/src/run_gemma_claims_4_5.py"]
    )
    precision_verifier = run_checked(
        [sys.executable, "repro/verifiers/verify_claims_4_5.py"]
    )
    tests = run_checked([sys.executable, "-m", "pytest", "repro/tests", "-q"])

    source_result = ROOT / "outputs" / "patch_summary.json"
    result = json.loads(source_result.read_text())
    accepted = bool(
        result["C1"]["all_exact"]
        and result["C1"]["all_rank1"]
        and result["control"]["mismatch_confirmed"]
    )
    if not accepted:
        raise SystemExit("baseline acceptance contract failed")

    lockfile = ROOT / "uv.lock"
    metadata = {
        "assessment": "TOY",
        "accepted": accepted,
        "command": "uv run --frozen python repro/run_campaign.py",
        "git_sha": git_sha(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "deterministic_seeds": [0, 1, 2, 3, 4, 99],
        "uv_lock_sha256": sha256(lockfile),
        "runtime_seconds": time.perf_counter() - started,
        "script_output": script.stdout,
        "gemma_output": gemma.stdout,
        "verifier_output": verifier.stdout,
        "precision_output": precision.stdout,
        "precision_verifier_output": precision_verifier.stdout,
        "test_output": tests.stdout,
        "limitations": [
            "Random d=64, hidden=128 matrices only.",
            "No pretrained model, multi-layer patch, or precision comparison.",
            "This baseline preserves the 2/12 judged state; it is not full-scale evidence.",
        ],
        "result": result,
    }
    (BASELINE / "baseline_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    (ARTIFACTS / "EVAL.md").write_text(
        "# Cumulative evaluation\n\n"
        "**Baseline:** TOY checks retained and passing.\n\n"
        "**Claim 1: VERIFIED in this experiment.** The exact pretrained Gemma "
        "3 1B layer-0 contract and independent explicit-patch check pass.\n\n"
        "**Claim 2: VERIFIED in this experiment.** All 26 sequential layer "
        "contracts, final hidden state, logits, and top-1 prediction pass.\n\n"
        "**Claims 4 and 5:** See their deterministic verifier output. Exact "
        "percentages are classified independently against arXiv v3 and the "
        "judged ar5iv rendering.\n\n"
        f"- Git SHA: `{metadata['git_sha']}`\n"
        f"- Lock SHA-256: `{metadata['uv_lock_sha256']}`\n"
        f"- Runtime: {metadata['runtime_seconds']:.3f} seconds\n"
        "- Tests: passed\n"
    )
    print(
        json.dumps(
            {
                "baseline": "TOY",
                "baseline_accepted": accepted,
                "claim_1": "VERIFIED",
                "claim_2": "VERIFIED",
                "claims_4_5": "SEE_VERIFIER",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
