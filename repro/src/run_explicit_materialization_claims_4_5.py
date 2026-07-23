#!/usr/bin/env python3
"""Audit full bfloat16 matrix materialization for Claims 4 and 5."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import psutil
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_gemma_claims_4_5 import (
    MODEL_BLOB,
    MODEL_ID,
    MODEL_REVISION,
    PROMPTS,
    invert_rmsnorm,
    register_target_hooks,
    run_reduced_layers,
    safe_component_divide,
    safe_scalar_denominator,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".openresearch" / "artifacts"
SEED = 20260723


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def materialized_input_projection(
    weight: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    denominator = safe_scalar_denominator(torch.dot(source, source))
    left = F.linear(target - source, weight)
    delta = torch.outer(left, source) / denominator
    actual = F.linear(source, weight + delta.to(weight.dtype))
    expected = F.linear(target, weight)
    error = float(torch.max(torch.abs(actual.float() - expected.float())).item())
    return actual, error


def materialized_output_projection(
    weight: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    denominator = safe_scalar_denominator(torch.dot(source, source))
    base = F.linear(source, weight)
    delta = torch.outer(target - base, source) / denominator
    actual = F.linear(source, weight + delta.to(weight.dtype))
    error = float(torch.max(torch.abs(actual.float() - target.float())).item())
    return actual, error


def run_explicit(
    model,
    captures,
    reduced_ids: torch.Tensor,
    position_id: torch.Tensor,
    method: str,
) -> tuple[torch.Tensor, dict]:
    hidden = model.model.embed_tokens(reduced_ids)
    position_global = model.model.rotary_emb(hidden, position_id)
    position_local = model.model.rotary_emb_local(hidden, position_id)
    diagnostics = {
        "layers_materialized": 0,
        "max_gate_target_linf": 0.0,
        "max_up_target_linf": 0.0,
        "max_down_target_linf": 0.0,
        "max_inversion_constraint_error": 0.0,
        "division_by_zero_count": 0,
        "hard_case_repair_count": 0,
        "max_abs_scale_patch": 0.0,
    }

    for index, layer in enumerate(model.model.layers):
        residual = hidden
        attention_input = layer.input_layernorm(hidden)
        positions = (
            position_local if layer.self_attn.is_sliding else position_global
        )
        attention_output, _ = layer.self_attn(
            hidden_states=attention_input,
            position_embeddings=positions,
            attention_mask=None,
            position_ids=position_id,
            past_key_values=None,
            use_cache=False,
            cache_position=position_id[0],
        )
        attention_output = layer.post_attention_layernorm(attention_output)
        v = (residual + attention_output)[:, -1]
        z = layer.pre_feedforward_layernorm(v)
        v_c = captures[index]["v_c"]
        z_c = captures[index]["z_c"]

        gate, gate_error = materialized_input_projection(
            layer.mlp.gate_proj.weight, z[0], z_c[0]
        )
        up, up_error = materialized_input_projection(
            layer.mlp.up_proj.weight, z[0], z_c[0]
        )
        gated = layer.mlp.act_fn(gate) * up
        diagnostics["max_gate_target_linf"] = max(
            diagnostics["max_gate_target_linf"], gate_error
        )
        diagnostics["max_up_target_linf"] = max(
            diagnostics["max_up_target_linf"], up_error
        )

        scale = 1.0 + layer.post_feedforward_layernorm.weight
        if method == "naive":
            h_down = F.linear(gated, layer.mlp.down_proj.weight)
            normalized = layer.post_feedforward_layernorm._norm(h_down)
            delta_scale, zero_count = safe_component_divide(
                v_c.to(v.dtype) - v, normalized
            )
        elif method == "stable":
            goal = (
                v_c.to(v.dtype)
                - v
                + captures[index]["h_out_c"].to(v.dtype)
            )
            requested_rms = float(
                torch.sqrt(
                    torch.mean(
                        captures[index]["h_down_c"][0].double().square()
                    )
                ).item()
            )
            h_target, inversion = invert_rmsnorm(
                goal[0], scale, requested_rms
            )
            h_down, down_error = materialized_output_projection(
                layer.mlp.down_proj.weight, gated, h_target
            )
            diagnostics["max_down_target_linf"] = max(
                diagnostics["max_down_target_linf"], down_error
            )
            diagnostics["max_inversion_constraint_error"] = max(
                diagnostics["max_inversion_constraint_error"],
                inversion["constraint_error"],
            )
            if inversion["solver_case"] != "paper_interior_root":
                diagnostics["hard_case_repair_count"] += 1
            normalized = layer.post_feedforward_layernorm._norm(h_down)
            remainder = goal[0] - scale * normalized
            delta_scale, zero_count = safe_component_divide(
                remainder, normalized
            )
            delta_scale = delta_scale[None, :]
        else:
            raise ValueError(method)

        diagnostics["division_by_zero_count"] += zero_count
        diagnostics["max_abs_scale_patch"] = max(
            diagnostics["max_abs_scale_patch"],
            float(torch.max(torch.abs(delta_scale.float())).item()),
        )
        hidden = (v + (scale + delta_scale) * normalized)[:, None, :]
        diagnostics["layers_materialized"] += 1

    final_hidden = model.model.norm(hidden)
    return model.lm_head(final_hidden)[:, -1].float(), diagnostics


def main() -> None:
    started = time.perf_counter()
    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        local_files_only=False,
    )
    model.eval()
    rows = []
    negatives = []

    for prompt_index, prompt in enumerate(PROMPTS):
        full_ids = tokenizer(prompt, return_tensors="pt").input_ids
        positions = torch.arange(full_ids.shape[1], dtype=torch.long)[None, :]
        captures, handles = register_target_hooks(model)
        with torch.inference_mode():
            contextual = model(
                input_ids=full_ids,
                position_ids=positions,
                use_cache=False,
            )
        for handle in handles:
            handle.remove()
        baseline_logits = contextual.logits[:, -1].float()
        reduced_ids = full_ids[:, -1:]
        reduced_position = positions[:, -1:]
        with torch.inference_mode():
            unpatched = model(
                input_ids=reduced_ids,
                position_ids=reduced_position,
                use_cache=False,
            ).logits[:, -1].float()
        negative_linf = float(
            torch.max(torch.abs(unpatched - baseline_logits)).item()
        )
        negatives.append(negative_linf)

        for method in ("naive", "stable"):
            with torch.inference_mode():
                action_logits, _ = run_reduced_layers(
                    model,
                    captures,
                    reduced_ids,
                    reduced_position,
                    method,
                    False,
                )
                explicit_logits, diagnostics = run_explicit(
                    model,
                    captures,
                    reduced_ids,
                    reduced_position,
                    method,
                )
            baseline_token = int(torch.argmax(baseline_logits).item())
            action_token = int(torch.argmax(action_logits).item())
            explicit_token = int(torch.argmax(explicit_logits).item())
            row = {
                "prompt_index": prompt_index,
                "method": method,
                "baseline_token_id": baseline_token,
                "action_token_id": action_token,
                "explicit_token_id": explicit_token,
                "baseline_token": tokenizer.decode([baseline_token]),
                "action_to_explicit_logit_linf": float(
                    torch.max(
                        torch.abs(action_logits - explicit_logits)
                    ).item()
                ),
                "baseline_to_explicit_logit_linf": float(
                    torch.max(
                        torch.abs(baseline_logits - explicit_logits)
                    ).item()
                ),
                "negative_unpatched_logit_linf": negative_linf,
                **diagnostics,
            }
            rows.append(row)
            print(
                "EXPLICIT_PROGRESS=" + json.dumps(row, sort_keys=True),
                flush=True,
            )

    runtime = time.perf_counter() - started
    raw = {
        "claims": [4, 5],
        "purpose": "fidelity audit of target-action shortcut",
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "weight_blob": MODEL_BLOB,
            "layers": len(model.model.layers),
            "dtype": str(next(model.parameters()).dtype),
        },
        "prompts": PROMPTS,
        "rows": rows,
        "negative_controls": negatives,
        "seed": SEED,
        "runtime_seconds": runtime,
        "environment": {
            "command": "uv run --frozen python repro/run_campaign.py",
            "git_sha": git_sha(),
            "uv_lock_sha256": sha256(ROOT / "uv.lock"),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "peak_rss_bytes": psutil.Process().memory_info().rss,
            "gpu_used": False,
        },
        "limitation": (
            "One generated token per prompt audits materialization fidelity; "
            "it does not estimate the paper's 100-token percentage."
        ),
    }
    raw_text = json.dumps(raw, indent=2) + "\n"
    for claim in (4, 5):
        path = ARTIFACTS / f"claim_{claim}"
        (path / "explicit_materialization_results.json").write_text(raw_text)
        shutil.copy2(
            ROOT / "repro" / "contracts" / "claims_4_5_explicit_audit.json",
            path / "explicit_materialization_contract.json",
        )
        shutil.copy2(
            ROOT
            / "repro"
            / "methods"
            / "explicit_materialization_claims_4_5.md",
            path / "explicit_materialization_method.md",
        )
        shutil.copy2(
            ROOT
            / "repro"
            / "source_audits"
            / "claims_4_5_research_routes.md",
            path / "three_route_uncertainty_audit.md",
        )
    print(
        "EXPLICIT_MATERIALIZATION_SUMMARY="
        + json.dumps(
            {
                "rows": len(rows),
                "explicit_token_matches": sum(
                    row["baseline_token_id"] == row["explicit_token_id"]
                    for row in rows
                ),
                "action_explicit_token_matches": sum(
                    row["action_token_id"] == row["explicit_token_id"]
                    for row in rows
                ),
                "max_action_to_explicit_logit_linf": max(
                    row["action_to_explicit_logit_linf"] for row in rows
                ),
                "max_baseline_to_explicit_logit_linf": max(
                    row["baseline_to_explicit_logit_linf"] for row in rows
                ),
                "min_negative_unpatched_linf": min(negatives),
                "runtime_seconds": runtime,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
