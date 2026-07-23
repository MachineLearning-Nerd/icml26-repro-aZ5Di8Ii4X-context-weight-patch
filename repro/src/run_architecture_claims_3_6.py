#!/usr/bin/env python3
"""Exercise Theorem 5 on actual Transformers architecture implementations."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import transformers
from transformers import (
    FalconConfig,
    GPT2Config,
    GPTJConfig,
    LlamaConfig,
    MistralConfig,
    MixtralConfig,
    Qwen2Config,
)
from transformers.models.falcon.modeling_falcon import FalconMLP
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3MLP
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2MLP
from transformers.models.gptj.modeling_gptj import GPTJBlock, GPTJMLP
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.mistral.modeling_mistral import MistralMLP
from transformers.models.mixtral.modeling_mixtral import (
    MixtralSparseMoeBlock,
)
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP


ROOT = Path(__file__).resolve().parents[2]
OUT3 = ROOT / ".openresearch" / "artifacts" / "claim_3"
OUT6 = ROOT / ".openresearch" / "artifacts" / "claim_6"
HIDDEN = 256
INTERMEDIATE = 512
SEEDS = list(range(20260723, 20260728))


def linf(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.max(torch.abs(a - b)).item())


def nonzero(vector: torch.Tensor, name: str) -> None:
    if float(torch.linalg.vector_norm(vector).item()) == 0.0:
        raise ValueError(f"{name} violates the theorem's nonzero assumption")


def patch_linear_input(
    layer: torch.nn.Linear, source: torch.Tensor, target: torch.Tensor
) -> int:
    nonzero(source, "input")
    difference = target - source
    delta = torch.outer(layer.weight @ difference, source) / source.dot(source)
    with torch.no_grad():
        layer.weight.add_(delta)
    return int(bool(torch.count_nonzero(delta)))


def patch_linear_output(
    layer: torch.nn.Linear, hidden: torch.Tensor, desired: torch.Tensor
) -> int:
    nonzero(hidden, "outer activation")
    delta = torch.outer(desired, hidden) / hidden.dot(hidden)
    with torch.no_grad():
        layer.weight.add_(delta)
    return int(bool(torch.count_nonzero(delta)))


def patch_conv1d_input(layer, source: torch.Tensor, target: torch.Tensor) -> int:
    """Patch a Transformers Conv1D input map, stored as [input, output]."""
    nonzero(source, "input")
    projected_change = (target - source) @ layer.weight
    delta = torch.outer(source, projected_change) / source.dot(source)
    with torch.no_grad():
        layer.weight.add_(delta)
    return int(bool(torch.count_nonzero(delta)))


def patch_conv1d_output(layer, hidden: torch.Tensor, desired: torch.Tensor) -> int:
    nonzero(hidden, "outer activation")
    delta = torch.outer(hidden, desired) / hidden.dot(hidden)
    with torch.no_grad():
        layer.weight.add_(delta)
    return int(bool(torch.count_nonzero(delta)))


def gated_parts(module, vector: torch.Tensor):
    if hasattr(module, "gate_proj"):
        hidden = module.act_fn(module.gate_proj(vector)) * module.up_proj(vector)
        return [module.gate_proj, module.up_proj], module.down_proj, hidden
    if hasattr(module, "w1"):
        hidden = module.act_fn(module.w1(vector)) * module.w3(vector)
        return [module.w1, module.w3], module.w2, hidden
    raise TypeError(type(module).__name__)


def standard_parts(name: str, module, vector: torch.Tensor):
    if name == "gpt2":
        hidden = module.act(module.c_fc(vector))
        return [module.c_fc], module.c_proj, hidden, "conv1d"
    if name == "gptj":
        hidden = module.act(module.fc_in(vector))
        return [module.fc_in], module.fc_out, hidden, "linear"
    if name == "falcon":
        hidden = module.act(module.dense_h_to_4h(vector))
        return (
            [module.dense_h_to_4h],
            module.dense_4h_to_h,
            hidden,
            "linear",
        )
    raise ValueError(name)


def sequential_trial(name: str, constructor, seed: int) -> dict:
    torch.manual_seed(seed)
    module = constructor().double().eval()
    source = torch.randn(HIDDEN, dtype=torch.float64)
    target = torch.randn(HIDDEN, dtype=torch.float64)
    nonzero(source, "source")
    nonzero(target, "target")
    desired = target - source
    contextual = target + module(target)

    patched = copy.deepcopy(module)
    negative = copy.deepcopy(module)
    ranks: list[int] = []

    if name in {"gemma", "llama", "mistral", "qwen"}:
        _, _, contextual_hidden = gated_parts(module, target)
        patched_inputs, patched_output, _ = gated_parts(patched, source)
        negative_inputs, _, _ = gated_parts(negative, source)
        for layer, neg_layer in zip(patched_inputs, negative_inputs):
            ranks.append(patch_linear_input(layer, source, target))
            patch_linear_input(neg_layer, source, target)
        _, _, hidden = gated_parts(patched, source)
        projection_error = linf(hidden, contextual_hidden)
        ranks.append(patch_linear_output(patched_output, hidden, desired))
    else:
        _, _, contextual_hidden, orientation = standard_parts(
            name, module, target
        )
        patched_inputs, patched_output, _, _ = standard_parts(
            name, patched, source
        )
        negative_inputs, _, _, _ = standard_parts(name, negative, source)
        for layer, neg_layer in zip(patched_inputs, negative_inputs):
            if orientation == "conv1d":
                ranks.append(patch_conv1d_input(layer, source, target))
                patch_conv1d_input(neg_layer, source, target)
            else:
                ranks.append(patch_linear_input(layer, source, target))
                patch_linear_input(neg_layer, source, target)
        _, _, hidden, _ = standard_parts(name, patched, source)
        projection_error = linf(hidden, contextual_hidden)
        if orientation == "conv1d":
            ranks.append(patch_conv1d_output(patched_output, hidden, desired))
        else:
            ranks.append(patch_linear_output(patched_output, hidden, desired))

    reproduced = source + patched(source)
    negative_output = source + negative(source)
    return {
        "architecture": name,
        "route": "sequential_residual",
        "seed": seed,
        "block_linf": linf(contextual, reproduced),
        "input_projection_linf": projection_error,
        "negative_control_linf": linf(contextual, negative_output),
        "patch_ranks": ranks,
        "source_norm": float(torch.linalg.vector_norm(source).item()),
        "target_norm": float(torch.linalg.vector_norm(target).item()),
        "outer_activation_norm": float(torch.linalg.vector_norm(hidden).item()),
    }


def mixtral_trial(constructor, seed: int) -> dict:
    torch.manual_seed(seed)
    module = constructor().double().eval()
    source = torch.randn(HIDDEN, dtype=torch.float64)
    target = torch.randn(HIDDEN, dtype=torch.float64)
    desired = target - source
    contextual_moe, contextual_logits = module(target[None, None, :])
    contextual = target + contextual_moe[0, 0]

    patched = copy.deepcopy(module)
    negative = copy.deepcopy(module)
    ranks = [patch_linear_input(patched.gate, source, target)]
    patch_linear_input(negative.gate, source, target)
    for patched_expert, negative_expert in zip(
        patched.experts, negative.experts
    ):
        ranks.append(patch_linear_input(patched_expert.w1, source, target))
        ranks.append(patch_linear_input(patched_expert.w3, source, target))
        patch_linear_input(negative_expert.w1, source, target)
        patch_linear_input(negative_expert.w3, source, target)

    # Transformers returns router logits flattened as [batch * sequence, experts].
    # Mirror the implementation's explicit float32 router calculation.  The
    # normalized top-k weights need not sum to exactly one after rounding, so
    # the theorem's S must be measured from these actual gates.
    router_weights = torch.softmax(contextual_logits, dim=1, dtype=torch.float)
    active_weights, active = torch.topk(
        router_weights[0], patched.top_k
    )
    active_weights = active_weights / active_weights.sum()
    active_weights = active_weights.to(source.dtype)
    gate_sum = float(active_weights.sum().item())
    max_hidden_error = 0.0
    min_hidden_norm = float("inf")
    for expert_index in active.tolist():
        original_expert = module.experts[expert_index]
        patched_expert = patched.experts[expert_index]
        original_hidden = (
            original_expert.act_fn(original_expert.w1(target))
            * original_expert.w3(target)
        )
        patched_hidden = (
            patched_expert.act_fn(patched_expert.w1(source))
            * patched_expert.w3(source)
        )
        max_hidden_error = max(
            max_hidden_error, linf(original_hidden, patched_hidden)
        )
        min_hidden_norm = min(
            min_hidden_norm,
            float(torch.linalg.vector_norm(patched_hidden).item()),
        )
        ranks.append(
            patch_linear_output(
                patched_expert.w2, patched_hidden, desired / gate_sum
            )
        )

    patched_moe, patched_logits = patched(source[None, None, :])
    negative_moe, _ = negative(source[None, None, :])
    reproduced = source + patched_moe[0, 0]
    negative_output = source + negative_moe[0, 0]
    return {
        "architecture": "mixtral",
        "route": "sparse_moe",
        "seed": seed,
        "block_linf": linf(contextual, reproduced),
        "input_projection_linf": max_hidden_error,
        "router_logit_linf": linf(contextual_logits, patched_logits),
        "negative_control_linf": linf(contextual, negative_output),
        "patch_ranks": ranks,
        "active_experts": active.tolist(),
        "active_gate_sum": gate_sum,
        "source_norm": float(torch.linalg.vector_norm(source).item()),
        "target_norm": float(torch.linalg.vector_norm(target).item()),
        "outer_activation_norm": min_hidden_norm,
    }


def gptj_parallel_trial(constructor, seed: int) -> dict:
    torch.manual_seed(seed)
    module = constructor().double().eval()
    source = torch.randn(HIDDEN, dtype=torch.float64)
    attention_without = torch.randn(HIDDEN, dtype=torch.float64)
    desired = torch.randn(HIDDEN, dtype=torch.float64)
    _, output_layer, hidden, _ = standard_parts("gptj", module, source)
    contextual = source + attention_without + desired + module(source)
    patched = copy.deepcopy(module)
    _, patched_output, patched_hidden, _ = standard_parts(
        "gptj", patched, source
    )
    rank = patch_linear_output(patched_output, patched_hidden, desired)
    reproduced = source + attention_without + patched(source)
    negative = source + attention_without + module(source)
    return {
        "architecture": "gptj",
        "route": "parallel_residual",
        "seed": seed,
        "block_linf": linf(contextual, reproduced),
        "input_projection_linf": 0.0,
        "negative_control_linf": linf(contextual, negative),
        "patch_ranks": [rank],
        "source_norm": float(torch.linalg.vector_norm(source).item()),
        "target_norm": float(torch.linalg.vector_norm(attention_without).item()),
        "outer_activation_norm": float(
            torch.linalg.vector_norm(hidden).item()
        ),
    }


def file_sha256(cls) -> tuple[str, str]:
    source = Path(inspect.getsourcefile(cls) or "")
    return str(source), hashlib.sha256(source.read_bytes()).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numpy_independent_cases() -> list[dict]:
    """Independent NumPy implementation of both stored weight orientations."""
    cases = []
    for orientation in ("linear", "conv1d"):
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            source = rng.standard_normal(37)
            target = rng.standard_normal(37)
            weight = rng.standard_normal((53, 37))
            if orientation == "linear":
                delta = np.outer(weight @ (target - source), source)
                delta /= source @ source
                error = np.max(np.abs((weight + delta) @ source - weight @ target))
            else:
                stored = weight.T
                delta = np.outer(source, (target - source) @ stored)
                delta /= source @ source
                error = np.max(np.abs(source @ (stored + delta) - target @ stored))
            cases.append(
                {
                    "orientation": orientation,
                    "seed": seed,
                    "projection_linf": float(error),
                    "negative_control_linf": float(
                        np.max(np.abs(source @ stored - target @ stored))
                        if orientation == "conv1d"
                        else np.max(np.abs(weight @ source - weight @ target))
                    ),
                }
            )
    return cases


def main() -> None:
    started = time.perf_counter()
    OUT3.mkdir(parents=True, exist_ok=True)
    OUT6.mkdir(parents=True, exist_ok=True)

    configs = {
        "gemma": lambda: Gemma3MLP(
            Gemma3TextConfig(
                hidden_size=HIDDEN,
                intermediate_size=INTERMEDIATE,
                hidden_activation="gelu_pytorch_tanh",
            )
        ),
        "llama": lambda: LlamaMLP(
            LlamaConfig(
                hidden_size=HIDDEN,
                intermediate_size=INTERMEDIATE,
                hidden_act="silu",
                mlp_bias=False,
            )
        ),
        "mistral": lambda: MistralMLP(
            MistralConfig(
                hidden_size=HIDDEN,
                intermediate_size=INTERMEDIATE,
                hidden_act="silu",
            )
        ),
        "qwen": lambda: Qwen2MLP(
            Qwen2Config(
                hidden_size=HIDDEN,
                intermediate_size=INTERMEDIATE,
                hidden_act="silu",
            )
        ),
        "gpt2": lambda: GPT2MLP(
            INTERMEDIATE,
            GPT2Config(
                n_embd=HIDDEN,
                n_inner=INTERMEDIATE,
                activation_function="gelu_new",
                resid_pdrop=0.0,
            ),
        ),
        "falcon": lambda: FalconMLP(
            FalconConfig(
                hidden_size=HIDDEN,
                ffn_hidden_size=INTERMEDIATE,
                activation="gelu",
                bias=True,
                hidden_dropout=0.0,
            )
        ),
    }
    mixtral_constructor = lambda: MixtralSparseMoeBlock(
        MixtralConfig(
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
            num_local_experts=4,
            num_experts_per_tok=2,
            hidden_act="silu",
            router_jitter_noise=0.0,
        )
    )
    gptj_constructor = lambda: GPTJMLP(
        INTERMEDIATE,
        GPTJConfig(
            n_embd=HIDDEN,
            n_inner=INTERMEDIATE,
            activation_function="gelu_new",
            resid_pdrop=0.0,
        ),
    )

    rows = []
    for name, constructor in configs.items():
        for seed in SEEDS:
            row = sequential_trial(name, constructor, seed)
            rows.append(row)
            print("ARCH_PROGRESS=" + json.dumps(row, sort_keys=True), flush=True)
    for seed in SEEDS:
        row = mixtral_trial(mixtral_constructor, seed)
        rows.append(row)
        print("ARCH_PROGRESS=" + json.dumps(row, sort_keys=True), flush=True)
    for seed in SEEDS:
        row = gptj_parallel_trial(gptj_constructor, seed)
        rows.append(row)
        print("ARCH_PROGRESS=" + json.dumps(row, sort_keys=True), flush=True)

    classes = [
        Gemma3MLP,
        LlamaMLP,
        MistralMLP,
        FalconMLP,
        Qwen2MLP,
        GPT2MLP,
        GPT2Block,
        MixtralSparseMoeBlock,
        GPTJMLP,
        GPTJBlock,
    ]
    class_sources = {}
    for cls in classes:
        source_path, digest = file_sha256(cls)
        class_sources[cls.__name__] = {
            "module": cls.__module__,
            "source_path": source_path,
            "source_sha256": digest,
        }

    gptj_forward = inspect.getsource(GPTJBlock.forward)
    topology_mixtral = mixtral_constructor()
    topology_checks = {
        "gptj_parallel_sum": all(
            term in gptj_forward
            for term in (
                "attn_outputs",
                "feed_forward_hidden_states",
                "+ residual",
            )
        ),
        "mixtral_router_and_experts": all(
            hasattr(topology_mixtral, name) for name in ("gate", "experts", "top_k")
        ),
        "gpt2_conv1d_orientation": configs["gpt2"]().c_fc.weight.shape
        == (HIDDEN, INTERMEDIATE),
    }
    zero_assumption_control = {"rejected": False, "message": ""}
    try:
        patch_linear_input(
            torch.nn.Linear(4, 4, bias=False).double(),
            torch.zeros(4, dtype=torch.float64),
            torch.ones(4, dtype=torch.float64),
        )
    except ValueError as error:
        zero_assumption_control = {"rejected": True, "message": str(error)}

    independent_cases = numpy_independent_cases()
    data = {
        "claims": [3, 6],
        "command": "uv run --frozen python repro/run_campaign.py",
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "uv_lock_sha256": sha256(ROOT / "uv.lock"),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "python_platform": platform.platform(),
        "cpu": {
            "logical_count": os.cpu_count(),
            "physical_count": psutil.cpu_count(logical=False),
            "torch_threads": torch.get_num_threads(),
            "memory_bytes": psutil.virtual_memory().total,
            "processor": platform.processor(),
        },
        "dtype": "float64",
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "seeds": SEEDS,
        "rows": rows,
        "class_sources": class_sources,
        "topology_checks": topology_checks,
        "zero_assumption_control": zero_assumption_control,
        "independent_numpy_cases": independent_cases,
        "runtime_seconds": time.perf_counter() - started,
        "limitations": [
            "Dynamic non-Gemma modules use width 256/intermediate 512.",
            "Weights are deterministic random initializations of actual Transformers classes.",
            "Full pretrained Gemma 3 1B evidence is supplied by cumulative Claims 1/2.",
            "This verifies architecture-level algebra, not language-model quality for every family.",
        ],
    }
    raw_text = json.dumps(data, indent=2) + "\n"
    (OUT3 / "raw_results.json").write_text(raw_text)
    (OUT6 / "raw_results.json").write_text(raw_text)
    for claim, output_path in ((3, OUT3), (6, OUT6)):
        shutil.copyfile(
            ROOT / "repro" / "contracts" / f"claim_{claim}.json",
            output_path / "claim_contract.json",
        )
        shutil.copyfile(
            ROOT / "repro" / "source_audits" / "claims_3_6.md",
            output_path / "source_audit.md",
        )
        shutil.copyfile(
            ROOT / "repro" / "methods" / "architecture_claims_3_6.md",
            output_path / "method.md",
        )
        (output_path / "environment.json").write_text(
            json.dumps(
                {
                    "command": data["command"],
                    "git_sha": data["git_sha"],
                    "uv_lock_sha256": data["uv_lock_sha256"],
                    "transformers_version": data["transformers_version"],
                    "torch_version": data["torch_version"],
                    "python_platform": data["python_platform"],
                    "cpu": data["cpu"],
                    "seeds": data["seeds"],
                    "runtime_seconds": data["runtime_seconds"],
                },
                indent=2,
            )
            + "\n"
        )
        (output_path / "limitations.md").write_text(
            "# Limitations and deviations\n\n"
            + "\n".join(f"- {item}" for item in data["limitations"])
            + "\n"
        )
    negative = {
        row["architecture"] + ":" + row["route"] + ":" + str(row["seed"]):
        row["negative_control_linf"]
        for row in rows
    }
    (OUT3 / "negative_control_output.json").write_text(
        json.dumps(negative, indent=2) + "\n"
    )
    (OUT6 / "negative_control_output.json").write_text(
        json.dumps(negative, indent=2) + "\n"
    )
    print(
        "CLAIMS_3_6_SUMMARY="
        + json.dumps(
            {
                "routes": len(rows),
                "architectures": sorted(
                    {row["architecture"] for row in rows}
                ),
                "max_block_linf": max(row["block_linf"] for row in rows),
                "min_negative_control_linf": min(
                    row["negative_control_linf"] for row in rows
                ),
                "topology_checks": topology_checks,
                "zero_assumption_control": zero_assumption_control,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
