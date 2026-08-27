"""Run sanity checks for Module 1 synthetic DGPs.

This script does not fit GP models yet. It verifies that the DGP behaves as
expected and that the evaluation table format is ready for later methods.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dgp import SyntheticCausalDGP, available_scenarios, summarize_dataset
from src.metrics import aggregate_metric_rows, point_estimate_metrics


def naive_difference_in_means(data: dict[str, np.ndarray]) -> float:
    treated = data["y"][data["a"] == 1]
    control = data["y"][data["a"] == 0]
    return float(treated.mean() - control.mean())


def oracle_ipw(data: dict[str, np.ndarray]) -> float:
    a = data["a"]
    y = data["y"]
    pi = data["pi"]
    return float(np.mean(a * y / pi - (1 - a) * y / (1 - pi)))


def oracle_plugin(data: dict[str, np.ndarray]) -> float:
    return float(np.mean(data["mu1"] - data["mu0"]))


def run(args: argparse.Namespace) -> None:
    output_dir = ROOT / "outputs" / "module1"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows = []
    metric_rows = []

    for overlap, effect in available_scenarios():
        scenario = f"{overlap}_{effect}"
        dgp = SyntheticCausalDGP(
            n_features=args.n_features,
            overlap=overlap,
            effect=effect,
            noise_sd=args.noise_sd,
        )
        for rep in range(args.reps):
            seed = args.seed + rep + 10_000 * available_scenarios().index((overlap, effect))
            data = dgp.sample(args.n, seed=seed)
            summary = summarize_dataset(data)
            dataset_rows.append({"scenario": scenario, "rep": rep, **summary})

            estimates = {
                "naive_diff_means": naive_difference_in_means(data),
                "oracle_ipw_true_pi": oracle_ipw(data),
                "oracle_plugin_true_mu": oracle_plugin(data),
            }
            for method, estimate in estimates.items():
                metric_rows.append(
                    {
                        "scenario": scenario,
                        "rep": rep,
                        "method": method,
                        **point_estimate_metrics(estimate, dgp.true_ate),
                    }
                )

    dataset_df = pd.DataFrame(dataset_rows)
    metric_df = pd.DataFrame(metric_rows)

    aggregate_rows = []
    for (scenario, method), group in metric_df.groupby(["scenario", "method"], sort=True):
        aggregate_rows.append(aggregate_metric_rows(group.to_dict("records")))
    aggregate_df = pd.DataFrame(aggregate_rows).sort_values(["scenario", "method"])

    dataset_path = output_dir / "dgp_dataset_summaries.csv"
    metric_path = output_dir / "baseline_point_metrics.csv"
    aggregate_path = output_dir / "baseline_point_metrics_aggregated.csv"
    dataset_df.to_csv(dataset_path, index=False)
    metric_df.to_csv(metric_path, index=False)
    aggregate_df.to_csv(aggregate_path, index=False)

    print(f"Wrote {dataset_path}")
    print(f"Wrote {metric_path}")
    print(f"Wrote {aggregate_path}")
    print()
    print(aggregate_df[["scenario", "method", "bias_mean", "abs_error_mean", "squared_error_mean"]].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--n-features", type=int, default=5)
    parser.add_argument("--noise-sd", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260520)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

