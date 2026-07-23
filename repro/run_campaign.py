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
        "# Baseline evaluation\n\n"
        "**Assessment: TOY.** The existing rank-1 algebra check and negative "
        "control pass, but this run uses only small random matrices. It does "
        "not verify any claim on a pretrained transformer.\n\n"
        f"- Git SHA: `{metadata['git_sha']}`\n"
        f"- Lock SHA-256: `{metadata['uv_lock_sha256']}`\n"
        f"- Runtime: {metadata['runtime_seconds']:.3f} seconds\n"
        "- Tests: passed\n"
    )
    print(json.dumps({"baseline": "TOY", "accepted": accepted}, indent=2))


if __name__ == "__main__":
    main()
