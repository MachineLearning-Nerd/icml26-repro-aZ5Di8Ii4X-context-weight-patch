import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import numpy as np
from run_patch import trial
def test_exact_match(): assert trial(seed=0)["exact"]
def test_rank1_gate(): assert trial(seed=1)["rank1_gate"]
def test_rank1_up(): assert trial(seed=2)["rank1_up"]
