#!/usr/bin/env python3
"""Reproduce Section 4 precision and stable-inversion claims on Gemma 3 1B."""

from __future__ import annotations

import gc
import hashlib
import json
import math
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
TOKENS_PER_PROMPT = 20
SEED = 20260723
PROMPTS = [
    (
        "Write a single-sentence weather forecast for Mars, from the "
        "perspective of a slightly annoyed robot:"
    ),
    (
        "Constraint: No 'e's. Topic: The meaning of life. Length: Exactly "
        "20 words. Write:"
    ),
    (
        "Explain what a 'bug' is in computer programming, but explain it to "
        "a 16th-century medieval peasant. Use 20 words or less:"
    ),
    (
        "Topic: Three siblings named Jack, Fred, and Alice. Task: Write one "
        "5-7-5 haiku about them:"
    ),
    (
        "We can absorb the impact of a prompt for a modern language model "
        "such as Gemma into the MLP weights. Write a two-line rhyming "
        "couplet about this fact:"
    ),
]


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
            captures[index]["h_out_c"] = last_vector(output)

        def layer_hook(_module, _args, output, index=index):
            captures[index]["layer_output_c"] = last_vector(output[0])

        handles.append(layer.pre_feedforward_layernorm.register_forward_hook(pre_hook))
        handles.append(layer.post_feedforward_layernorm.register_forward_hook(post_hook))
        handles.append(layer.register_forward_hook(layer_hook))
    return captures, handles


def safe_scalar_denominator(value: torch.Tensor) -> torch.Tensor:
    if float(value.float().item()) == 0.0:
        return torch.ones_like(value)
    return value


def safe_component_divide(
    numerator: torch.Tensor, denominator: torch.Tensor
) -> tuple[torch.Tensor, int]:
    zero = denominator == 0
    safe = torch.where(zero, torch.ones_like(denominator), denominator)
    return numerator / safe, int(torch.count_nonzero(zero).item())


def materialized_rank1_action(
    weight: torch.Tensor, source: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    denominator = safe_scalar_denominator(torch.dot(source, source))
    left = F.linear(target - source, weight)
    delta = torch.outer(left, source) / denominator
    patched = F.linear(source, weight + delta.to(weight.dtype))
    return patched, {
        "delta_frobenius": float(
            (
                torch.linalg.vector_norm(left.float())
                * torch.linalg.vector_norm(source.float())
                / denominator.float().abs()
            ).item()
        ),
        "target_linf": float(
            torch.max(torch.abs(patched.float() - F.linear(target, weight).float())).item()
        ),
    }


def invert_rmsnorm(
    goal: torch.Tensor, scale: torch.Tensor, target_rms: float
) -> tuple[torch.Tensor, dict[str, float | str]]:
    """Solve the Appendix-B constrained inverse on its stated root interval."""
    goal64 = goal.double()
    scale64 = scale.double()
    scale_sq = scale64.square()
    numerator_sq = (goal64 * scale64).square()
    minimum = float(torch.min(scale_sq).item())
    high = minimum - max(1e-12, abs(minimum) * 1e-12)

    def root_function(mu: float) -> float:
        value = torch.mean(numerator_sq / (scale_sq - mu).square()) - 1.0
        return float(value.item())

    high_value = root_function(high)
    if math.isfinite(high_value) and high_value > 0.0:
        solver_case = "paper_interior_root"
        step = max(1.0, abs(minimum))
        low = minimum - step
        for _ in range(256):
            if root_function(low) < 0.0:
                break
            step *= 2.0
            low = minimum - step
        else:
            raise RuntimeError("failed to bracket Appendix-B RMSNorm inverse")

        for _ in range(128):
            midpoint = (low + high) / 2.0
            if root_function(midpoint) < 0.0:
                low = midpoint
            else:
                high = midpoint
        mu = (low + high) / 2.0
        y = goal64 * scale64 / (scale_sq - mu)
    else:
        # The paper's existence proof overlooks the trust-region "hard case":
        # when every numerator at min(m^2) is zero, F need not diverge there.
        # The global constrained minimizer then has mu=min(m^2), uses the
        # regular secular solution off that eigenspace, and fills the remaining
        # sphere norm inside the minimum eigenspace.
        solver_case = "hard_case_repair_no_paper_root"
        mu = minimum
        tolerance = max(1e-12, abs(minimum) * 1e-12)
        minimum_mask = torch.abs(scale_sq - minimum) <= tolerance
        y = torch.zeros_like(goal64)
        regular = ~minimum_mask
        y[regular] = (
            goal64[regular]
            * scale64[regular]
            / (scale_sq[regular] - minimum)
        )
        remaining = goal64.numel() - float(torch.sum(y.square()).item())
        if remaining < -1e-8:
            raise RuntimeError(
                "hard-case RMSNorm inverse has negative remaining norm"
            )
        indices = torch.nonzero(minimum_mask, as_tuple=False).flatten()
        if indices.numel() == 0:
            raise RuntimeError("hard-case minimum eigenspace is empty")
        y[indices[0]] = math.sqrt(max(0.0, remaining))
    constraint_error = abs(float(torch.mean(y.square()).item()) - 1.0)
    target = (target_rms * y).to(goal.dtype)
    return target, {
        "mu": mu,
        "constraint_error": constraint_error,
        "target_rms": float(torch.sqrt(torch.mean(target.double().square())).item()),
        "requested_rms": target_rms,
        "solver_case": solver_case,
        "paper_interval_high_value": high_value,
    }


def run_reduced_layers(model, captures, reduced_ids, position_id, method: str):
    hidden = model.model.embed_tokens(reduced_ids)
    position_global = model.model.rotary_emb(hidden, position_id)
    position_local = model.model.rotary_emb_local(hidden, position_id)
    diagnostics = {
        "max_projection_target_linf": 0.0,
        "max_inversion_constraint_error": 0.0,
        "division_by_zero_count": 0,
        "max_abs_scale_patch": 0.0,
        "matrix_patch_frobenius_sum": 0.0,
        "hard_case_repair_count": 0,
    }

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
            position_ids=position_id,
            past_key_values=None,
            use_cache=False,
            cache_position=position_id[0],
        )
        attention_output = layer.post_attention_layernorm(attention_output)
        v = (residual + attention_output)[0, -1]
        z = layer.pre_feedforward_layernorm(v)
        v_c = captures[index]["v_c"]
        z_c = captures[index]["z_c"]

        gate, gate_diag = materialized_rank1_action(
            layer.mlp.gate_proj.weight, z, z_c
        )
        up, up_diag = materialized_rank1_action(
            layer.mlp.up_proj.weight, z, z_c
        )
        diagnostics["max_projection_target_linf"] = max(
            diagnostics["max_projection_target_linf"],
            gate_diag["target_linf"],
            up_diag["target_linf"],
        )
        diagnostics["matrix_patch_frobenius_sum"] += (
            gate_diag["delta_frobenius"] + up_diag["delta_frobenius"]
        )
        gated = layer.mlp.act_fn(gate) * up
        scale = 1.0 + layer.post_feedforward_layernorm.weight

        if method == "naive":
            h_down = layer.mlp.down_proj(gated)
            normalized = layer.post_feedforward_layernorm._norm(h_down)
            delta_scale, zero_count = safe_component_divide(
                v_c.to(v.dtype) - v, normalized
            )
        elif method == "stable":
            h_down_c = captures[index]["h_down_c"]
            goal = (
                v_c.to(v.dtype)
                - v
                + captures[index]["h_out_c"].to(v.dtype)
            )
            requested_rms = float(
                torch.sqrt(torch.mean(h_down_c.double().square())).item()
            )
            h_target, inversion = invert_rmsnorm(
                goal, scale, requested_rms
            )
            diagnostics["max_inversion_constraint_error"] = max(
                diagnostics["max_inversion_constraint_error"],
                inversion["constraint_error"],
            )
            if inversion["solver_case"] != "paper_interior_root":
                diagnostics["hard_case_repair_count"] += 1
            denominator = safe_scalar_denominator(torch.dot(gated, gated))
            down_delta = torch.outer(h_target - h_down_c.to(h_target.dtype), gated)
            down_delta = down_delta / denominator
            diagnostics["matrix_patch_frobenius_sum"] += float(
                torch.linalg.vector_norm(down_delta.float()).item()
            )
            h_down = F.linear(
                gated,
                layer.mlp.down_proj.weight
                + down_delta.to(layer.mlp.down_proj.weight.dtype),
            )
            normalized = layer.post_feedforward_layernorm._norm(h_down)
            remainder = goal - scale * normalized
            delta_scale, zero_count = safe_component_divide(
                remainder, normalized
            )
        else:
            raise ValueError(method)

        diagnostics["division_by_zero_count"] += zero_count
        diagnostics["max_abs_scale_patch"] = max(
            diagnostics["max_abs_scale_patch"],
            float(torch.max(torch.abs(delta_scale.float())).item()),
        )
        patched = v + (scale + delta_scale) * normalized
        hidden = patched.view(1, 1, -1)

    final_hidden = model.model.norm(hidden)
    logits = model.lm_head(final_hidden)[0, -1].float()
    return logits, diagnostics


def run_precision(model, tokenizer, dtype_name: str) -> tuple[list[dict], list[float]]:
    rows = []
    negative_controls = []
    for prompt_index, prompt in enumerate(PROMPTS):
        history = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).input_ids
        for step in range(TOKENS_PER_PROMPT):
            captures, handles = register_target_hooks(model)
            with torch.inference_mode():
                contextual = model(input_ids=history, use_cache=False)
            for handle in handles:
                handle.remove()
            baseline_logits = contextual.logits[0, -1].float()
            baseline_token = int(torch.argmax(baseline_logits).item())
            reduced_ids = history[:, -1:]
            position_id = torch.tensor(
                [[history.shape[1] - 1]], dtype=torch.long
            )

            with torch.inference_mode():
                naive_logits, naive_diag = run_reduced_layers(
                    model, captures, reduced_ids, position_id, "naive"
                )
                stable_logits, stable_diag = run_reduced_layers(
                    model, captures, reduced_ids, position_id, "stable"
                )
                if step == 0:
                    unpatched = model(
                        input_ids=reduced_ids,
                        position_ids=position_id,
                        use_cache=False,
                    ).logits[0, -1].float()
                    negative_controls.append(
                        float(
                            torch.max(
                                torch.abs(unpatched - baseline_logits)
                            ).item()
                        )
                    )

            baseline_probability = torch.softmax(baseline_logits, dim=-1)
            naive_probability = torch.softmax(naive_logits, dim=-1)
            stable_probability = torch.softmax(stable_logits, dim=-1)
            row = {
                "precision": dtype_name,
                "prompt_index": prompt_index,
                "step": step,
                "absolute_position": int(position_id.item()),
                "baseline_token_id": baseline_token,
                "baseline_token": tokenizer.decode([baseline_token]),
                "naive_token_id": int(torch.argmax(naive_logits).item()),
                "stable_token_id": int(torch.argmax(stable_logits).item()),
                "naive_logit_linf": float(
                    torch.max(torch.abs(naive_logits - baseline_logits)).item()
                ),
                "stable_logit_linf": float(
                    torch.max(torch.abs(stable_logits - baseline_logits)).item()
                ),
                "naive_tvd": float(
                    (0.5 * torch.sum(torch.abs(
                        naive_probability - baseline_probability
                    ))).item()
                ),
                "stable_tvd": float(
                    (0.5 * torch.sum(torch.abs(
                        stable_probability - baseline_probability
                    ))).item()
                ),
                "naive_max_projection_target_linf": naive_diag[
                    "max_projection_target_linf"
                ],
                "stable_max_projection_target_linf": stable_diag[
                    "max_projection_target_linf"
                ],
                "stable_max_inversion_constraint_error": stable_diag[
                    "max_inversion_constraint_error"
                ],
                "naive_max_abs_scale_patch": naive_diag[
                    "max_abs_scale_patch"
                ],
                "stable_max_abs_scale_patch": stable_diag[
                    "max_abs_scale_patch"
                ],
                "naive_matrix_patch_frobenius_sum": naive_diag[
                    "matrix_patch_frobenius_sum"
                ],
                "stable_matrix_patch_frobenius_sum": stable_diag[
                    "matrix_patch_frobenius_sum"
                ],
                "stable_hard_case_repair_count": stable_diag[
                    "hard_case_repair_count"
                ],
                "division_by_zero_count": (
                    naive_diag["division_by_zero_count"]
                    + stable_diag["division_by_zero_count"]
                ),
            }
            rows.append(row)
            print(
                "PRECISION_PROGRESS="
                + json.dumps(
                    {
                        "precision": dtype_name,
                        "prompt": prompt_index,
                        "step": step,
                        "baseline": baseline_token,
                        "naive": row["naive_token_id"],
                        "stable": row["stable_token_id"],
                        "naive_linf": row["naive_logit_linf"],
                        "stable_linf": row["stable_logit_linf"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            next_token = torch.tensor([[baseline_token]], dtype=torch.long)
            history = torch.cat([history, next_token], dim=1)
    return rows, negative_controls


def aggregate(rows: list[dict], precision: str, method: str) -> dict:
    selected = [row for row in rows if row["precision"] == precision]
    matches = sum(
        row["baseline_token_id"] == row[f"{method}_token_id"]
        for row in selected
    )
    return {
        "token_count": len(selected),
        "matches": matches,
        "agreement": matches / len(selected),
        "max_logit_linf": max(row[f"{method}_logit_linf"] for row in selected),
        "max_tvd": max(row[f"{method}_tvd"] for row in selected),
    }


def copy_scaffold(claim: int) -> Path:
    path = ARTIFACTS / f"claim_{claim}"
    path.mkdir(parents=True, exist_ok=True)
    for source, target in (
        (ROOT / "repro" / "contracts" / f"claim_{claim}.json",
         path / "claim_contract.json"),
        (ROOT / "repro" / "source_audits" / "claims_4_5.md",
         path / "source_audit.md"),
        (ROOT / "repro" / "methods" / "gemma_claims_4_5.md",
         path / "method.md"),
        (ROOT / "repro" / "verifiers" / "verify_claims_4_5.py",
         path / "claim_verifier.py"),
    ):
        shutil.copy2(source, target)
    return path


def main() -> None:
    torch.manual_seed(SEED)
    torch.set_grad_enabled(False)
    started = time.perf_counter()
    process = psutil.Process(os.getpid())
    claim_dirs = [copy_scaffold(4), copy_scaffold(5)]
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    )
    rows = []
    negatives = []
    model_metadata = None

    for dtype, dtype_name in (
        (torch.float32, "float32"),
        (torch.bfloat16, "bfloat16"),
    ):
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            dtype=dtype,
            attn_implementation="eager",
            use_safetensors=True,
        )
        model.eval()
        if model_metadata is None:
            model_metadata = {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "weight_blob": MODEL_BLOB,
                "architecture": model.__class__.__name__,
                "layers": len(model.model.layers),
                "hidden_size": model.config.hidden_size,
                "intermediate_size": model.config.intermediate_size,
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
            }
        precision_rows, precision_negatives = run_precision(
            model, tokenizer, dtype_name
        )
        rows.extend(precision_rows)
        negatives.extend(precision_negatives)
        del model
        gc.collect()

    summary = {
        f"{precision}_{method}": aggregate(rows, precision, method)
        for precision in ("float32", "bfloat16")
        for method in ("naive", "stable")
    }
    environment = {
        "command": "uv run --frozen python repro/run_campaign.py",
        "git_sha": git_sha(),
        "uv_lock_sha256": sha256(ROOT / "uv.lock"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "gpu_used": False,
    }
    raw = {
        "paper": "arXiv:2511.17864v3",
        "model": model_metadata,
        "prompts": PROMPTS,
        "tokens_per_prompt": TOKENS_PER_PROMPT,
        "seed": SEED,
        "rows": rows,
        "summary": summary,
        "negative_controls": {
            "unpatched_reduced_context_logit_linf": negatives,
            "minimum_logit_linf": min(negatives),
        },
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_bytes": process.memory_info().rss,
        "environment": environment,
    }
    raw_text = json.dumps(raw, indent=2) + "\n"
    negative_text = json.dumps(raw["negative_controls"], indent=2) + "\n"
    for path in claim_dirs:
        (path / "raw_results.json").write_text(raw_text)
        (path / "negative_control_output.json").write_text(negative_text)
        (path / "environment.json").write_text(
            json.dumps(environment, indent=2) + "\n"
        )
        (path / "limitations.md").write_text(
            "# Limitations and deviations\n\n"
            "- The full Gemma 3 1B model is tested; the paper also mentions 4B.\n"
            "- Exact prompt strings are recovered from the paper source PDFs.\n"
            "- The 20-token horizon is recovered from the plotted Mars panel; "
            "the prose does not state it.\n"
            "- The stable scalar root is solved in float64 before its target is "
            "cast to the tested dtype; the model and materialized patches use "
            "the declared float32 or bfloat16 precision.\n"
            "- The authoritative v3 and ar5iv/judge percentages disagree; both "
            "are classified separately.\n"
        )
    print("CLAIMS_4_5_RAW_JSON=" + json.dumps(raw, sort_keys=True), flush=True)
    print("CLAIMS_4_5_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
