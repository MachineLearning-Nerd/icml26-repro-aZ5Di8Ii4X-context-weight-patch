#!/usr/bin/env python3
"""Test Theorems 1 and 2 on the full pretrained Gemma 3 1B transformer."""

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


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".openresearch" / "artifacts"
MODEL_ID = "unsloth/gemma-3-1b-it"
MODEL_REVISION = "5b11413a10db4e486ef16a20101fd028f8f2499c"
MODEL_BLOB = "cf8782aa8cd58000ba309c9f5ac84a810c3bf7bf"
PROMPT = (
    "Write a single-sentence weather forecast for Mars, from the perspective "
    "of a slightly annoyed robot:"
)
SEED = 20260723


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def last_vector(value: torch.Tensor) -> torch.Tensor:
    return value.detach()[0, -1].clone()


def register_target_hooks(model):
    captures = [{} for _ in model.model.layers]
    handles = []

    for index, layer in enumerate(model.model.layers):
        def pre_hook(_module, args, output, index=index):
            captures[index]["v_c"] = last_vector(args[0])
            captures[index]["z_c"] = last_vector(output)

        def post_hook(_module, args, output, index=index):
            captures[index]["h_down_c"] = last_vector(args[0])
            captures[index]["mlp_branch_c"] = last_vector(output)

        def layer_hook(_module, _args, output, index=index):
            captures[index]["layer_output_c"] = last_vector(output[0])

        handles.append(layer.pre_feedforward_layernorm.register_forward_hook(pre_hook))
        handles.append(layer.post_feedforward_layernorm.register_forward_hook(post_hook))
        handles.append(layer.register_forward_hook(layer_hook))
    return captures, handles


def rank1_projection_action(
    weight: torch.Tensor, z: torch.Tensor, z_c: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dz = z_c - z
    denominator = torch.dot(z, z)
    left = F.linear(dz, weight)
    patched = F.linear(z, weight) + left * (torch.dot(z, z) / denominator)
    target = F.linear(z_c, weight)
    return patched, target, left


def explicit_patch_error(
    weight: torch.Tensor, z: torch.Tensor, z_c: torch.Tensor
) -> float:
    dz = z_c - z
    left = F.linear(dz, weight)
    delta = torch.outer(left, z) / torch.dot(z, z)
    explicit = F.linear(z, weight + delta)
    target = F.linear(z_c, weight)
    return float(torch.max(torch.abs(explicit - target)).item())


def copy_evidence_scaffold(claim: int) -> Path:
    claim_dir = ARTIFACTS / f"claim_{claim}"
    claim_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        ROOT / "repro" / "contracts" / f"claim_{claim}.json",
        claim_dir / "claim_contract.json",
    )
    shutil.copy2(
        ROOT / "repro" / "source_audits" / "claims_1_2.md",
        claim_dir / "source_audit.md",
    )
    shutil.copy2(
        ROOT / "repro" / "methods" / "gemma_claims_1_2.md",
        claim_dir / "method.md",
    )
    shutil.copy2(
        ROOT / "repro" / "verifiers" / "verify_claims_1_2.py",
        claim_dir / "claim_verifier.py",
    )
    return claim_dir


def main() -> None:
    torch.manual_seed(SEED)
    torch.set_grad_enabled(False)
    started = time.perf_counter()
    process = psutil.Process(os.getpid())
    claim_1_dir = copy_evidence_scaffold(1)
    claim_2_dir = copy_evidence_scaffold(2)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.float32,
        attn_implementation="eager",
        use_safetensors=True,
    )
    model.eval()

    full_ids = tokenizer(PROMPT, return_tensors="pt", add_special_tokens=True).input_ids
    if full_ids.shape[1] < 2:
        raise SystemExit("prompt tokenization did not produce context plus test token")
    reduced_ids = full_ids[:, -1:]
    target_position = torch.tensor([[full_ids.shape[1] - 1]], dtype=torch.long)

    captures, handles = register_target_hooks(model)
    with torch.inference_mode():
        contextual = model(input_ids=full_ids, use_cache=False)
    for handle in handles:
        handle.remove()

    with torch.inference_mode():
        unpatched = model(
            input_ids=reduced_ids,
            position_ids=target_position,
            use_cache=False,
        )

    hidden = model.model.embed_tokens(reduced_ids)
    position_global = model.model.rotary_emb(hidden, target_position)
    position_local = model.model.rotary_emb_local(hidden, target_position)
    layer_rows = []

    with torch.inference_mode():
        for index, layer in enumerate(model.model.layers):
            residual = hidden
            attention_input = layer.input_layernorm(hidden)
            position_embeddings = (
                position_local if layer.self_attn.is_sliding else position_global
            )
            attention_output, _ = layer.self_attn(
                hidden_states=attention_input,
                position_embeddings=position_embeddings,
                attention_mask=None,
                position_ids=target_position,
                past_key_values=None,
                use_cache=False,
                cache_position=target_position[0],
            )
            attention_output = layer.post_attention_layernorm(attention_output)
            v = (residual + attention_output)[0, -1]
            z = layer.pre_feedforward_layernorm(v)
            v_c = captures[index]["v_c"]
            z_c = captures[index]["z_c"]

            gate, gate_target, _ = rank1_projection_action(
                layer.mlp.gate_proj.weight, z, z_c
            )
            up, up_target, _ = rank1_projection_action(
                layer.mlp.up_proj.weight, z, z_c
            )
            gated = layer.mlp.act_fn(gate) * up
            h_down = layer.mlp.down_proj(gated)
            unscaled = layer.post_feedforward_layernorm._norm(h_down.float())
            scale = 1.0 + layer.post_feedforward_layernorm.weight.float()
            zero_count = int(torch.count_nonzero(unscaled == 0).item())
            if zero_count:
                raise SystemExit(
                    f"Theorem 1 nonzero assumption failed at layer {index}"
                )
            delta_scale = (v_c.float() - v.float()) / unscaled
            patched = v.float() + (scale + delta_scale) * unscaled
            target_reconstructed = v_c.float() + scale * unscaled
            target_captured = captures[index]["layer_output_c"].float()
            omit_scale = v.float() + scale * unscaled

            row = {
                "layer": index,
                "z_norm": float(torch.linalg.vector_norm(z.float()).item()),
                "minimum_abs_unscaled_activation": float(
                    torch.min(torch.abs(unscaled)).item()
                ),
                "zero_unscaled_activation_count": zero_count,
                "gate_projection_linf": float(
                    torch.max(torch.abs(gate - gate_target)).item()
                ),
                "up_projection_linf": float(
                    torch.max(torch.abs(up - up_target)).item()
                ),
                "patched_output_linf": float(
                    torch.max(torch.abs(patched - target_reconstructed)).item()
                ),
                "target_reconstruction_linf": float(
                    torch.max(torch.abs(target_reconstructed - target_captured)).item()
                ),
                "omit_scale_patch_linf": float(
                    torch.max(torch.abs(omit_scale - target_reconstructed)).item()
                ),
                "maximum_abs_delta_scale": float(
                    torch.max(torch.abs(delta_scale)).item()
                ),
            }
            if index == 0:
                row["explicit_gate_patch_linf"] = explicit_patch_error(
                    layer.mlp.gate_proj.weight, z, z_c
                )
                row["explicit_up_patch_linf"] = explicit_patch_error(
                    layer.mlp.up_proj.weight, z, z_c
                )
            else:
                row["explicit_gate_patch_linf"] = 0.0
                row["explicit_up_patch_linf"] = 0.0
            layer_rows.append(row)
            hidden = patched.to(hidden.dtype).view(1, 1, -1)

        patched_final_hidden = model.model.norm(hidden)
        patched_logits = model.lm_head(patched_final_hidden)[0, -1]

    contextual_logits = contextual.logits[0, -1].float()
    unpatched_logits = unpatched.logits[0, -1].float()
    contextual_hidden = model.model.norm(
        captures[-1]["layer_output_c"].view(1, 1, -1)
    )[0, -1].float()
    final_hidden_linf = float(
        torch.max(torch.abs(patched_final_hidden[0, -1].float() - contextual_hidden)).item()
    )
    final_logit_linf = float(
        torch.max(torch.abs(patched_logits.float() - contextual_logits)).item()
    )
    negative_linf = float(
        torch.max(torch.abs(unpatched_logits - contextual_logits)).item()
    )
    contextual_top1 = int(torch.argmax(contextual_logits).item())
    patched_top1 = int(torch.argmax(patched_logits).item())

    runtime = time.perf_counter() - started
    raw = {
        "paper": "arXiv:2511.17864v3",
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "weight_blob": MODEL_BLOB,
            "architecture": model.__class__.__name__,
            "layers": len(model.model.layers),
            "hidden_size": model.config.hidden_size,
            "intermediate_size": model.config.intermediate_size,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "dtype": str(next(model.parameters()).dtype),
        },
        "prompt": PROMPT,
        "prompt_token_count": int(full_ids.shape[1]),
        "test_token_id": int(reduced_ids.item()),
        "test_token": tokenizer.decode(reduced_ids[0]),
        "position_id": int(target_position.item()),
        "seed": SEED,
        "layers": layer_rows,
        "max_layer_output_linf": max(row["patched_output_linf"] for row in layer_rows),
        "max_target_reconstruction_linf": max(
            row["target_reconstruction_linf"] for row in layer_rows
        ),
        "final_hidden_linf": final_hidden_linf,
        "final_logit_linf": final_logit_linf,
        "contextual_top1_token_id": contextual_top1,
        "patched_top1_token_id": patched_top1,
        "top1_agreement": contextual_top1 == patched_top1,
        "negative_unpatched_logit_linf": negative_linf,
        "runtime_seconds": runtime,
        "peak_rss_bytes": process.memory_info().rss,
        "environment": {
            "command": "uv run --frozen python repro/run_campaign.py",
            "git_sha": git_sha(),
            "uv_lock_sha256": sha256(ROOT / "uv.lock"),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "gpu_used": false,
        },
    }
    raw_text = json.dumps(raw, indent=2) + "\n"
    (claim_1_dir / "raw_results.json").write_text(raw_text)
    (claim_2_dir / "raw_results.json").write_text(raw_text)
    negative = {
        "omit_scale_patch_linf_layer_0": layer_rows[0]["omit_scale_patch_linf"],
        "unpatched_final_logit_linf": negative_linf,
        "expected": "Both mismatches must exceed their preregistered minima.",
    }
    for claim_dir in (claim_1_dir, claim_2_dir):
        (claim_dir / "negative_control_output.json").write_text(
            json.dumps(negative, indent=2) + "\n"
        )
        (claim_dir / "environment.json").write_text(
            json.dumps(raw["environment"], indent=2) + "\n"
        )
        (claim_dir / "limitations.md").write_text(
            "# Limitations and deviations\n\n"
            "- One pretrained Gemma 3 1B checkpoint and one prompt/token are tested.\n"
            "- The public ungated mirror is used; its weight blob ID and byte size "
            "match the gated official checkpoint exactly.\n"
            "- The experiment tests float32 numerical realization of exact "
            "real-arithmetic theorems.\n"
            "- It does not establish a reusable patch for arbitrary future tokens.\n"
        )

    summary = {
        "model": f"{MODEL_ID}@{MODEL_REVISION}",
        "layers": len(layer_rows),
        "claim_1_layer_0_output_linf": layer_rows[0]["patched_output_linf"],
        "claim_2_max_layer_output_linf": raw["max_layer_output_linf"],
        "final_logit_linf": final_logit_linf,
        "top1_agreement": raw["top1_agreement"],
        "negative_unpatched_logit_linf": negative_linf,
        "runtime_seconds": runtime,
    }
    print("GEMMA_CLAIMS_1_2_SUMMARY=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
