# Negative controls & falsification


---
<!-- trackio-cell
{"type": "code", "id": "cell_6d595fe5ae9e", "created_at": "2026-07-17T12:37:45+00:00", "title": "Unit tests (3/3)", "command": ["python", "-m", "pytest", "repro/tests/test_patch.py", "-q"], "exit_code": 0, "duration_s": 0.289}
-->
````bash
$ python -m pytest repro/tests/test_patch.py -q
````

exit 0 · 0.3s


````python title=test_patch.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import numpy as np
from run_patch import trial
def test_exact_match(): assert trial(seed=0)["exact"]
def test_rank1_gate(): assert trial(seed=1)["rank1_gate"]
def test_rank1_up(): assert trial(seed=2)["rank1_up"]

````


````output
...                                                                      [100%]
3 passed in 0.07s

````
