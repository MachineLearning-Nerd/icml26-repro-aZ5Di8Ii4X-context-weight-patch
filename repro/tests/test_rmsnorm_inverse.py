from __future__ import annotations

import torch
import pytest

from repro.src.run_gemma_claims_4_5 import invert_rmsnorm


def test_invert_rmsnorm_interior_root() -> None:
    goal = torch.tensor([2.0, -1.0, 0.5, 3.0], dtype=torch.float64)
    scale = torch.tensor([1.0, 1.5, 0.75, 2.0], dtype=torch.float64)
    target, diagnostics = invert_rmsnorm(goal, scale, 2.5)
    assert diagnostics["solver_case"] == "paper_interior_root"
    assert diagnostics["constraint_error"] < 1e-12
    assert torch.sqrt(torch.mean(target.square())).item() == pytest.approx(2.5)


def test_invert_rmsnorm_hard_case_missing_from_paper_proof() -> None:
    # The minimum-scale coordinate has zero numerator. The secular function
    # stays below zero at min(m^2), so the paper's claimed interior root does
    # not exist. The boundary eigenspace supplies the remaining sphere norm.
    goal = torch.tensor([4.0, 0.1, -0.2, 0.3], dtype=torch.float64)
    scale = torch.tensor([0.0, 2.0, 2.0, 2.0], dtype=torch.float64)
    target, diagnostics = invert_rmsnorm(goal, scale, 1.75)
    assert diagnostics["solver_case"] == "hard_case_repair_no_paper_root"
    assert diagnostics["paper_interval_high_value"] < 0.0
    assert diagnostics["constraint_error"] < 1e-12
    assert torch.sqrt(torch.mean(target.square())).item() == pytest.approx(1.75)
