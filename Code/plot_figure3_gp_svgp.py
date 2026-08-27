"""Reproduce Figure 3 directly from the saved simulation results."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "results/figure_data/figure3_exact_svgp/posterior_metrics.csv"
OUTPUT_DIR = ROOT / "results/figures"

PANELS = [
    ("Exact GP", "exact_gp", "exact_gp_onestep_logistic_pi"),
    ("SVGP (m=160)", "svgp", "svgp_onestep_logistic_pi"),
]


def metric_label(rows):
    return (
        f"Bias {rows['bias'].mean():+.3f}\n"
        f"Cov {100 * rows['covered'].mean():.1f}%\n"
        f"Width {rows['ci_width'].mean():.3f}"
    )


def draw_distribution(ax, values, color, label, grid, bins):
    ax.hist(values, bins=bins, density=True, color=color, alpha=0.20)
    ax.plot(grid, gaussian_kde(values)(grid), color=color, linewidth=2.5, label=label)


def main():
    data = pd.read_csv(DATA_FILE)
    data = data.loc[
        (data["scenario"] == "strong_homogeneous")
        & (data["estimand"] == "population_ate_bb")
        & (data["n_inducing"] == 160)
    ].copy()

    methods = [method for _, raw, corrected in PANELS for method in (raw, corrected)]
    data = data.loc[data["method"].isin(methods)]

    x_min, x_max = 0.55, 1.40
    grid = np.linspace(x_min, x_max, 400)
    bins = np.linspace(x_min, x_max, 29)
    true_ate = data["true_value"].iloc[0]

    plt.rcParams.update({"font.family": "serif", "font.size": 12})
    fig, axes = plt.subplots(1, 2, figsize=(12.82, 5), sharex=True, sharey=True)

    for ax, (title, raw_method, corrected_method) in zip(axes, PANELS):
        ax.set_box_aspect(3 / 5)
        raw = data.loc[data["method"] == raw_method]
        corrected = data.loc[data["method"] == corrected_method]

        draw_distribution(
            ax, raw["posterior_mean"], "#0000B3", "Raw posterior", grid, bins
        )
        draw_distribution(
            ax,
            corrected["posterior_mean"],
            "#CC8400",
            "One-step corrected",
            grid,
            bins,
        )
        ax.axvline(true_ate, color="black", linestyle="--", linewidth=1.8, label="True ATE")
        ax.text(0.03, 0.96, metric_label(raw), color="#0000B3", va="top", transform=ax.transAxes)
        ax.text(
            0.97,
            0.96,
            metric_label(corrected),
            color="#8A5600",
            ha="right",
            va="top",
            transform=ax.transAxes,
        )
        ax.set(title=title, xlabel="Posterior mean of ATE", xlim=(x_min, x_max))
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Density across 200 simulations")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Posterior mean distributions", y=0.97)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.20, top=0.82, wspace=0.14)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / "synthetic_exact_gp_svgp_posterior_mean_1x2_strong_homogeneous_m160"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Saved {stem}.png and {stem}.pdf")


if __name__ == "__main__":
    main()
