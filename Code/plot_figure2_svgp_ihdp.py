"""Reproduce Figure 2 directly from the three saved IHDP summary files."""

from pathlib import Path

import matplotlib
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results/figure_data/figure2_svgp_ihdp"
OUTPUT_DIR = ROOT / "results/figures"


def add_tolerance_lines(ax, mean_tolerance, other_tolerance, other_label):
    mean_line = ax.axhline(
        mean_tolerance,
        color="#B43939",
        linewidth=1.3,
        label="Mean tolerance",
    )
    other_line = ax.axhline(
        other_tolerance,
        color="#B43939",
        linewidth=1.3,
        linestyle="--",
        label=other_label,
    )
    return [mean_line, other_line]


def plot_process_panel(ax, data, mean_column, q95_column, title, subtitle):
    mean_line = ax.plot(
        data["n_inducing"], data[mean_column], "o-", color="#3664B4", label="Mean"
    )[0]
    q95_line = ax.plot(
        data["n_inducing"], data[q95_column], "o-", color="#737B82", label="q95"
    )[0]
    tolerance_handles = add_tolerance_lines(ax, 0.10, 0.20, "q95 tolerance")
    ax.set_title(title, loc="left", pad=10)
    ax.set_ylabel(subtitle)
    return [mean_line, q95_line] + tolerance_handles


def plot_ate_panel(ax, data, title, selected_rank):
    series = [
        ("mean_std_mean_diff", "Mean difference", "#6852AC"),
        ("mean_abs_sd_ratio_error", "SD error", "#CC6600"),
        ("mean_abs_width_ratio_error", "Width error", "#00897B"),
    ]
    handles = []
    for column, label, color in series:
        line = ax.plot(data["n_inducing"], data[column], "o-", color=color, label=label)[0]
        handles.append(line)
    tolerance_handles = add_tolerance_lines(ax, 0.10, 0.05, "SD / width tolerance")
    ax.axvline(selected_rank, color="0.65", linewidth=1.2)
    ax.text(selected_rank, 0.36, f"m={selected_rank}", ha="center", color="0.4")
    ax.set_title(title, loc="left", pad=10)
    ax.set_ylabel("Mean diagnostic error")
    return handles + tolerance_handles


def main():
    process = pd.read_csv(DATA_DIR / "process_diagnostics.csv")
    raw = pd.read_csv(DATA_DIR / "raw_equivalence.csv")
    corrected = pd.read_csv(DATA_DIR / "corrected_equivalence.csv")

    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)

    process_handles = plot_process_panel(
        axes[0, 0],
        process,
        "mean_proc_mean",
        "q95_proc_mean",
        "Process mean error",
        "Standardized RMSE",
    )
    plot_process_panel(
        axes[0, 1],
        process,
        "mean_proc_cov",
        "q95_proc_cov",
        "Process covariance error",
        "Relative Frobenius error",
    )
    ate_handles = plot_ate_panel(axes[1, 0], raw, "Raw ATE equivalence errors", 200)
    plot_ate_panel(axes[1, 1], corrected, "Corrected ATE equivalence errors", 160)

    axes[0, 0].set_ylim(0, 0.22)
    axes[0, 1].set_ylim(0, 1.05)
    axes[1, 0].set_ylim(0, 0.38)
    axes[1, 1].set_ylim(0, 0.38)
    for ax in axes.flat:
        ax.set_xlabel("Inducing rank m")
        ax.set_xticks([80, 160, 240, 320, 400, 480])
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.legend(
        process_handles,
        ["Mean", "q95", "Mean tolerance", "q95 tolerance"],
        loc="center left",
        bbox_to_anchor=(0.10, 0.535),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    fig.legend(
        ate_handles,
        [
            "Mean difference",
            "SD error",
            "Width error",
            "Mean tolerance",
            "SD / width tolerance",
        ],
        loc="lower left",
        bbox_to_anchor=(0.10, 0.015),
        ncol=5,
        frameon=False,
        fontsize=8,
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.16, hspace=0.95, wspace=0.26)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / "svgp_ihdp_error_vs_m"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Saved {stem}.png and {stem}.pdf")


if __name__ == "__main__":
    main()
