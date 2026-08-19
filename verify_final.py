#!/usr/bin/env python3
"""Verify the context-parameter dossier and public branch state."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY = "icml26-context-parameter-equivalence"
CANONICAL = ("MachineLearning-Nerd", "MachineLearning-Nerd@users.noreply.github.com")
EXPECTED_BRANCHES = {
    "main",
    "evidence/architecture-matrix",
    "evidence/baseline-toy",
    "evidence/precision-sweep",
    "evidence/precision-verdict-architecture",
    "evidence/explicit-bfloat16-audit",
    "evidence/stable-inversion",
    "evidence/precision-cross-check",
    "evidence/gemma-block-multilayer",
    "evidence/release-candidate",
}
EXPECTED_STATUSES = {
    "C1": "VERIFIED_SCOPED",
    "C2": "VERIFIED_SCOPED",
    "C3": "VERIFIED_ARCHITECTURE_SCOPED",
    "C4": "FALSIFIED_SCOPED",
    "C5": "FALSIFIED_SCOPED",
    "C6": "VERIFIED_ARCHITECTURE_SCOPED",
}
REQUIRED_PATHS = [
    "README.md", "STATUS.md", "CLAIM_EVIDENCE.md", "SOURCE_MANIFEST.md",
    "ENVIRONMENT.md", "REPORT.md", "BRANCH_AUDIT.md", "CITATION.cff",
    "AUTHOR_THANK_YOU.md", "claims.json", "reproduction_verdicts.json",
    "AUTONOMOUS_STATE.json", "EVIDENCE_MANIFEST.json", "verify_final.py",
    "repro/src/verify_final.py", "evidence/claim_summary.json", "outputs/patch_summary.json",
    ".openresearch/artifacts/claim_1/raw_results.json", ".openresearch/artifacts/claim_2/raw_results.json",
    ".openresearch/artifacts/claim_3/raw_results.json", ".openresearch/artifacts/claim_4/raw_results.json",
    ".openresearch/artifacts/claim_5/explicit_materialization_results.json", ".openresearch/artifacts/claim_6/raw_results.json",
    "publication_gate.json", "logbook.json",
]


def fail(message: str) -> None:
    print(f"FINAL_AUDIT=FAILED {message}", file=sys.stderr)
    raise SystemExit(1)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def run(*args: str) -> str:
    result = subprocess.run(args, cwd=ROOT, check=False, capture_output=True, text=True)
    if result.returncode:
        fail(f"command failed: {' '.join(args)}\n{result.stderr.strip()}")
    return result.stdout.strip()


def current_json(path: str) -> object:
    try:
        return json.loads((ROOT / path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_git() -> int:
    origin = run("git", "config", "--get", "remote.origin.url").removesuffix(".git")
    require(origin == f"https://github.com/MachineLearning-Nerd/{REPOSITORY}", f"unexpected origin: {origin}")
    require(run("git", "symbolic-ref", "--short", "HEAD") == "main", "current branch is not main")
    require(run("git", "symbolic-ref", "refs/remotes/origin/HEAD") == "refs/remotes/origin/main", "origin HEAD is not main")
    remote = {
        line.removeprefix("origin/")
        for line in run("git", "for-each-ref", "--format=%(refname:short)", "refs/remotes/origin").splitlines()
        if line.startswith("origin/") and line != "origin/HEAD"
    }
    require(remote == EXPECTED_BRANCHES, "remote branch set differs from branch audit")
    refs = run("git", "for-each-ref", "--format=%(refname)", "refs").splitlines()
    require(not any("orx/" in ref or ref.endswith("/master") or "refs/original/" in ref for ref in refs), "legacy refs remain")
    identities = {
        tuple(line.split("\t"))
        for line in run("git", "log", "--all", "--format=%an\t%ae\t%cn\t%ce").splitlines()
        if line.strip()
    }
    require(identities == {(CANONICAL[0], CANONICAL[1], CANONICAL[0], CANONICAL[1])}, f"non-canonical identity: {sorted(identities)}")
    require("co-authored-by:" not in run("git", "log", "--all", "--format=%B").lower(), "coauthor trailer remains")
    require(run("git", "status", "--porcelain") == "", "worktree is not clean")
    return len(remote)


def verify_artifacts() -> None:
    for path in REQUIRED_PATHS:
        require((ROOT / path).is_file(), f"required path missing: {path}")
    claims = current_json("claims.json")
    require(claims.get("repository") == f"MachineLearning-Nerd/{REPOSITORY}", "claims repository differs")
    require(claims.get("overall_verdict") == "VERIFIED_SCOPED_WITH_NARROW_PRECISION_BOUNDARY" and claims.get("publication_allowed") is False, "claims publication boundary differs")
    require({row.get("id"): row.get("status") for row in claims.get("claims", [])} == EXPECTED_STATUSES, "claim statuses differ")
    reproduction = current_json("reproduction_verdicts.json")
    require(reproduction.get("overall_verdict") == "VERIFIED_SCOPED_WITH_NARROW_PRECISION_BOUNDARY" and reproduction.get("publication_allowed") is False, "reproduction header differs")
    require({row.get("id"): row.get("status") for row in reproduction.get("claims", [])} == EXPECTED_STATUSES, "reproduction statuses differ")
    state = current_json("AUTONOMOUS_STATE.json")
    require(state.get("phase") == "published_and_verified" and state.get("overall_verdict") == "VERIFIED_SCOPED_WITH_NARROW_PRECISION_BOUNDARY", "state is not final")
    require(state.get("publication_allowed") is False and state.get("branch_count") == len(EXPECTED_BRANCHES), "state publication or branch count differs")
    summary = current_json("evidence/claim_summary.json")
    require({key: value.get("status") for key, value in summary.get("claims", {}).items() if key in EXPECTED_STATUSES} == EXPECTED_STATUSES, "source claim summary differs")
    require(summary.get("claims", {}).get("full_paper_experiments", {}).get("status") == "NOT_REPRODUCED", "full-paper boundary changed")
    gate = current_json("publication_gate.json")
    require(gate.get("claims_verified") == 4 and gate.get("claims_falsified") == 2 and gate.get("claims_total") == 6, "historical publication gate changed")
    inner = subprocess.run([sys.executable, str(ROOT / "repro/src/verify_final.py")], cwd=ROOT, check=False, capture_output=True, text=True)
    require(inner.returncode == 0, f"inner verifier failed: {inner.stdout.strip()} {inner.stderr.strip()}")


def verify_manifest() -> None:
    manifest = current_json("EVIDENCE_MANIFEST.json")
    require(manifest.get("schema_version") == 1 and manifest.get("hash_algorithm") == "sha256", "manifest header changed")
    entries = manifest.get("files")
    require(isinstance(entries, dict) and entries, "evidence manifest is empty")
    for path, expected in entries.items():
        target = ROOT / path
        require(target.is_file(), f"manifest path missing: {path}")
        require(sha256(target) == expected, f"manifest hash mismatch: {path}")
    require("AUTONOMOUS_STATE.json" not in entries and "EVIDENCE_MANIFEST.json" not in entries, "mutable manifest cycle detected")


def main() -> None:
    branches = verify_git()
    verify_artifacts()
    verify_manifest()
    commits = int(run("git", "rev-list", "--count", "--all"))
    require(commits >= 20, "reachable history is unexpectedly short")
    print(f"FINAL_AUDIT=VERIFIED branches={branches} commits={commits} C1:verified C2:verified C3:architecture C4:falsified_scoped C5:falsified_scoped C6:architecture publication_allowed=false")


if __name__ == "__main__":
    main()
