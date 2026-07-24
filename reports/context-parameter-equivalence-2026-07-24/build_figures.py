#!/usr/bin/env python3
"""Build the report's evidence figures from its checked-in summary data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "data" / "summary.json").read_text())
OUT = HERE / "images"
OUT.mkdir(exist_ok=True)

COLORS = {
    "paper": "#8a94a6",
    "observed": "#176b87",
    "stable": "#28a18b",
    "negative": "#d95050",
    "done": "#28a18b",
    "failed": "#d95050",
    "cancelled": "#e2a33a",
}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def precision_agreement() -> None:
    labels = ["float32\nnaive", "bfloat16\nnaive", "bfloat16\nstable"]
    paper = [
        DATA["paper"]["float32_agreement"],
        DATA["paper"]["bfloat16_naive_agreement"],
        DATA["paper"]["bfloat16_stable_judged_agreement"],
    ]
    observed = [
        DATA["observed"]["float32_agreement"],
        DATA["observed"]["bfloat16_naive_agreement"],
        DATA["observed"]["bfloat16_stable_agreement"],
    ]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bars_a = ax.bar(x - 0.19, paper, 0.38, label="Paper / judged rendering", color=COLORS["paper"])
    bars_b = ax.bar(x + 0.19, observed, 0.38, label="Observed (100 tokens)", color=COLORS["observed"])
    ax.axhline(100, color="#233044", lw=1, alpha=0.35)
    ax.set_ylim(80, 102.5)
    ax.set_ylabel("Greedy token agreement (%)")
    ax.set_xticks(x, labels)
    ax.set_title("The reported bfloat16 agreement gap was absent")
    ax.legend(frameon=False, loc="lower left")
    ax.bar_label(bars_a, fmt="%.1f%%", padding=3, fontsize=9)
    ax.bar_label(bars_b, fmt="%.1f%%", padding=3, fontsize=9)
    ax.text(
        2,
        100.8,
        "arXiv v3 prose: 100%",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#4d596b",
    )
    save(fig, "01-headline-precision-agreement.png")


def precision_errors() -> None:
    labels = ["float32\nnaive", "bfloat16\nnaive", "bfloat16\nstable", "explicit audit\naction ↔ matrix"]
    values = [
        DATA["observed"]["float32_logit_linf"],
        DATA["observed"]["bfloat16_naive_logit_linf"],
        DATA["observed"]["bfloat16_stable_logit_linf"],
        DATA["observed"]["explicit_action_logit_linf"],
    ]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bars = ax.bar(labels, values, color=[COLORS["observed"], COLORS["paper"], COLORS["stable"], "#6e57a5"])
    ax.set_yscale("log")
    ax.set_ylabel("Maximum logit L∞ error (log scale)")
    ax.set_title("Stable inversion reduced error, even though tokens already matched")
    ax.bar_label(bars, labels=[f"{v:.3g}" for v in values], padding=4, fontsize=9)
    ax.grid(axis="y", which="both", alpha=0.18)
    save(fig, "02-precision-logit-errors.png")


def full_model_errors() -> None:
    labels = ["Layer 0\npatched", "Worst of\n26 layers", "Final\nlogits", "Unpatched\ncontrol"]
    values = [
        DATA["gemma"]["layer0_output_linf"],
        DATA["gemma"]["max_26_layer_output_linf"],
        DATA["gemma"]["final_logit_linf"],
        DATA["gemma"]["unpatched_final_logit_linf"],
    ]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bars = ax.bar(labels, values, color=[COLORS["observed"]] * 3 + [COLORS["negative"]])
    ax.set_yscale("log")
    ax.set_ylabel("L∞ error (log scale)")
    ax.set_title("The full 26-layer patch stays close; the unpatched control does not")
    ax.bar_label(bars, labels=[f"{v:.3g}" for v in values], padding=4, fontsize=9)
    ax.grid(axis="y", which="both", alpha=0.18)
    save(fig, "03-full-model-vs-control.png")


def architecture_coverage() -> None:
    names = list(DATA["architectures"])
    exact = [DATA["architectures"][name]["max_block_linf"] for name in names]
    negative = [DATA["architectures"][name]["min_negative_linf"] for name in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    ax.bar(x - 0.19, exact, 0.38, label="Patched maximum error", color=COLORS["observed"])
    ax.bar(x + 0.19, negative, 0.38, label="Omitted-patch minimum error", color=COLORS["negative"])
    ax.set_yscale("log")
    ax.set_xticks(x, [name.upper() if name.startswith("gpt") else name.title() for name in names])
    ax.set_ylabel("Block L∞ error (log scale)")
    ax.set_title("Actual architecture classes satisfy the construction across five seeds")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", which="both", alpha=0.18)
    save(fig, "04-architecture-coverage.png")


def experiment_lineage() -> None:
    runs = DATA["formal_runs"]
    labels = [run["label"] for run in runs]
    minutes = [run["seconds"] / 60 for run in runs]
    colors = [COLORS[run["status"]] for run in runs]
    fig, ax = plt.subplots(figsize=(11.5, 7))
    y = np.arange(len(runs))
    bars = ax.barh(y, minutes, color=colors)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Wall-clock minutes")
    ax.set_title("The final route followed measured dead ends, not a single clean run")
    ax.bar_label(bars, labels=[f"{value:.1f}m" for value in minutes], padding=4, fontsize=8)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS[name])
        for name in ("done", "failed", "cancelled")
    ]
    ax.legend(handles, ["Done", "Failed", "Cancelled"], frameon=False, ncol=3, loc="lower right")
    ax.grid(axis="x", alpha=0.18)
    save(fig, "05-experiment-lineage.png")


if __name__ == "__main__":
    precision_agreement()
    precision_errors()
    full_model_errors()
    architecture_coverage()
    experiment_lineage()
