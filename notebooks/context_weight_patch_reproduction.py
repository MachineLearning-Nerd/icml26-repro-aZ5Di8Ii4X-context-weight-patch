import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    return mo, pd, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Context-equivalent weight patches on Gemma 3

    **Observed headline:** float32, naive bfloat16, and stable bfloat16 each
    matched **100/100** greedy tokens on the five paper prompts. The paper
    reports 100% / 87.5%, with a stable endpoint of 98% in the judged
    rendering (100% in arXiv v3).

    This notebook embeds the completed CPU evidence. It does **not** rerun
    the 1B-parameter model.
    """)
    return


@app.cell
def _():
    claim_rows = [
        {"claim": 1, "verdict": "VERIFIED", "confidence": "HIGH", "evidence": "Gemma layer-0 L∞ 7.629e-06"},
        {"claim": 2, "verdict": "VERIFIED", "confidence": "HIGH", "evidence": "26 layers; final-logit L∞ 2.480e-05"},
        {"claim": 3, "verdict": "VERIFIED", "confidence": "MEDIUM", "evidence": "40 actual-class trials; max L∞ 4.441e-15"},
        {"claim": 4, "verdict": "FALSIFIED", "confidence": "MEDIUM", "evidence": "naive BF16 100/100, not 87.5%"},
        {"claim": 5, "verdict": "FALSIFIED", "confidence": "MEDIUM", "evidence": "naive=stable=100/100; error 0.375→0.25"},
        {"claim": 6, "verdict": "VERIFIED (architecture)", "confidence": "MEDIUM", "evidence": "8 named families plus MoE/parallel routes"},
    ]
    precision_rows = [
        {"method": "float32 naive", "paper_percent": 100.0, "observed_percent": 100.0, "max_logit_linf": 2.6702880859375e-05},
        {"method": "bfloat16 naive", "paper_percent": 87.5, "observed_percent": 100.0, "max_logit_linf": 0.375},
        {"method": "bfloat16 stable", "paper_percent": 98.0, "observed_percent": 100.0, "max_logit_linf": 0.25},
    ]
    architecture_rows = [
        {"architecture": "Falcon", "max_block_linf": 3.1086244689504383e-15, "min_control_linf": 4.091164029582885},
        {"architecture": "Gemma", "max_block_linf": 3.9968028886505635e-15, "min_control_linf": 3.6050010585394254},
        {"architecture": "GPT-2", "max_block_linf": 4.440892098500626e-15, "min_control_linf": 3.633338677189962},
        {"architecture": "GPT-J", "max_block_linf": 1.7763568394002505e-15, "min_control_linf": 2.4963252329265053},
        {"architecture": "Llama", "max_block_linf": 2.6645352591003757e-15, "min_control_linf": 3.6050010585394254},
        {"architecture": "Mistral", "max_block_linf": 2.6645352591003757e-15, "min_control_linf": 3.6050010585394254},
        {"architecture": "Mixtral", "max_block_linf": 2.55351295663786e-15, "min_control_linf": 3.90132591424009},
        {"architecture": "Qwen", "max_block_linf": 2.6645352591003757e-15, "min_control_linf": 3.6050010585394254},
    ]
    return architecture_rows, claim_rows, precision_rows


@app.cell
def _(mo, pd, plt, precision_rows):
    precision_frame = pd.DataFrame(precision_rows)
    figure, axis = plt.subplots(figsize=(8, 4))
    precision_frame.plot(
        x="method",
        y=["paper_percent", "observed_percent"],
        kind="bar",
        color=["#8a94a6", "#176b87"],
        ax=axis,
    )
    axis.set_ylim(80, 102)
    axis.set_ylabel("Greedy token agreement (%)")
    axis.set_xlabel("")
    axis.set_title("Reported versus observed precision agreement")
    axis.legend(["Paper / judged", "Observed"], frameon=False)
    mo.vstack(
        [
            mo.md("## Strongest result"),
            figure,
            mo.md(
                "Stable inversion lowers worst logit error, but no token-agreement "
                "increase is possible here because naive bfloat16 already matches 100/100."
            ),
        ]
    )
    return


@app.cell
def _(claim_rows, mo):
    claim_picker = mo.ui.dropdown(
        options={f"Claim {row['claim']}": row["claim"] for row in claim_rows},
        value=1,
        label="Inspect a claim",
    )
    claim_picker
    return (claim_picker,)


@app.cell
def _(claim_picker, claim_rows, mo):
    selected_claim = next(row for row in claim_rows if row["claim"] == claim_picker.value)
    mo.callout(
        mo.md(
            f"**{selected_claim['verdict']} · {selected_claim['confidence']} confidence**  \n"
            f"{selected_claim['evidence']}"
        ),
        kind="info",
    )
    return


@app.cell
def _(architecture_rows, mo, pd):
    mo.vstack(
        [
            mo.md(
                """
                ## Named architecture coverage

                These are actual pinned Transformers classes at reduced width,
                evaluated over five deterministic seeds. Controls omit the output
                patch and must separate.
                """
            ),
            mo.ui.table(pd.DataFrame(architecture_rows), pagination=False),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## How the construction works

    For nonzero \(z\), define

    \[
    \Delta W = \frac{W(z_C-z)z^\top}{\|z\|^2}.
    \]

    Then

    \[
    (W+\Delta W)z
    = Wz + W(z_C-z)\frac{z^\top z}{\|z\|^2}
    = Wz_C.
    \]

    The campaign applies this idea to both gated input projections, then
    uses output controllability to absorb the residual difference. Mixtral
    uses the actual rounded router sum \(S=\sum_j s_j\); GPT-J uses a
    parallel-block output update.

    ## Reproduce formally

    ```bash
    uv sync --frozen
    uv run --frozen python repro/run_campaign.py
    ```

    Formal reproduction requires enough CPU RAM for Gemma 3 1B. The
    displayed results are already embedded, so Molab readers do not need to
    execute that expensive path.
    """)
    return


if __name__ == "__main__":
    app.run()
