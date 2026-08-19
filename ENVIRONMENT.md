# Environment and execution boundary

- Python: 3.12, with dependencies pinned by `uv.lock`.
- Core Gemma evidence: CPU-capable PyTorch and the pinned public Gemma 3 1B checkpoint.
- Gemma scope: one checkpoint, one target history/token, and 26 sequential layers.
- Other architecture routes: reduced-width deterministic random weights for architecture conformance, not pretrained language-quality evidence.
- Final verification: `python3 verify_final.py` checks committed evidence and runs the inner verifier without downloading model weights.

The historical full campaign can take hours. Its model-download and compute requirements are not
part of the lightweight publication-state gate.
