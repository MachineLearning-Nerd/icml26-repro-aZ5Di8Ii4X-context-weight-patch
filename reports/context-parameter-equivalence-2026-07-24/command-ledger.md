# Command ledger

This ledger records the commands that define or validate the scientific
campaign. It deliberately omits generated remote wrappers, credential
operations, and read-only presentation inspection commands. The experiment
command itself is identical on every node.

## Fixed environment and experiment command

```bash
uv sync --python 3.12
uv lock
uv run --frozen python repro/run_campaign.py
```

The last line is the command recorded by `orx exp status` for every experiment
node. Variants change committed code, never the command or environment.

## Startup and source audit

```bash
orx skill
orx skill orx-experiment-tree
orx skill orx-evidence
orx skill orx-git
orx skill orx-compute
orx projects --json
orx project view e0799332-ef3e-4e79-932d-2ada2344bceb
orx runs e0799332-ef3e-4e79-932d-2ada2344bceb
orx paper 2511.17864 --full
git status --short
git branch -a
git rev-parse HEAD
git rev-parse origin/master
df -h .
env | cut -d= -f1 | sort
```

Paper and Space retrieval used an explicit browser user agent and immutable
revision identifiers. SHA-256 manifests were generated immediately after each
download.

## Formal launches

Each launch below used the same image, CPU flavor, and four-hour safety
timeout. The experiment identifier selects committed code; it does not alter
the run command.

```bash
orx exp run 23b76d06-1bb6-4786-998e-4f36f482a903 --backend local
orx exp run 29d95256-61f3-49db-b4ef-dbdcc7202bed --backend hf --flavor cpu-upgrade --timeout 14400 --image ghcr.io/astral-sh/uv:python3.12-bookworm-slim
orx exp run d78ff588-1ad8-499f-a146-d5126f530ae0 --backend hf --flavor cpu-upgrade --timeout 14400 --image ghcr.io/astral-sh/uv:python3.12-bookworm-slim
orx exp run 38443bbd-d24f-43ad-afd5-4a783950a4f9 --backend hf --flavor cpu-upgrade --timeout 14400 --image ghcr.io/astral-sh/uv:python3.12-bookworm-slim
orx exp run 574c34ab-d558-4a35-98b8-853424537067 --backend hf --flavor cpu-upgrade --timeout 14400 --image ghcr.io/astral-sh/uv:python3.12-bookworm-slim
orx exp run 9975fcd6-0b98-4d38-bfca-674500063a6b --backend hf --flavor cpu-upgrade --timeout 14400 --image ghcr.io/astral-sh/uv:python3.12-bookworm-slim
orx exp run 72c5f81c-efe5-40f4-9457-36460c8e3dcf --backend hf --flavor cpu-upgrade --timeout 14400 --image ghcr.io/astral-sh/uv:python3.12-bookworm-slim
```

The release-candidate launch finished as run
`c73d8345-e83b-457b-872a-b4dc368cb2f7` in 58m12s.

## Monitoring and evidence

Runs were polled in bounded intervals:

```bash
orx exp wait <experiment-id> --timeout 480
orx runs e0799332-ef3e-4e79-932d-2ada2344bceb
orx logs <run-id>
orx exp desc <experiment-id> --set "<evidence summary>"
```

Cancelled routes used:

```bash
orx exp cancel <experiment-id>
```

No training or evaluation command was executed directly in a shell, and no
GPU backend was used.

## Local validation

```bash
uv run --frozen marimo check notebooks/context_weight_patch_reproduction.py
uv run --frozen python -m py_compile repro/run_campaign.py reports/context-parameter-equivalence-2026-07-24/build_figures.py notebooks/context_weight_patch_reproduction.py
uv run --frozen python reports/context-parameter-equivalence-2026-07-24/build_figures.py
uv run --frozen python -m pytest repro/tests -q
```

The cumulative remote command also executes the full verifier and test suite,
and each verifier exits nonzero when its claim contract is not satisfied.
