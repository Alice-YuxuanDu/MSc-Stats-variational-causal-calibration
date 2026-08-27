"""Run Module 4 RFF full posterior and mean-field VI experiments."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_module4_experiment_grid import correction_suffix, propensity_for_model
from src.dgp import SyntheticCausalDGP, available_scenarios, summarize_dataset
from src.gp_exact import (
    ExactGPConfig,
    fit_gp_lengthscales_empirical_bayes,
    make_counterfactual_inputs,
    make_gp_inputs,
    population_ate_bb_draws,
    sample_cate_draws,
)
from src.metrics import aggregate_metric_rows, posterior_draw_metrics
from src.posterior_correction import (
    one_step_population_ate_draws,
    one_step_sample_cate_draws,
    outcome_eif_draws,
)
from src.vi_rff import (
    RFFConfig,
    coefficient_variance_ratios,
    counterfactual_draws_from_beta,
    fit_rff_bayesian_regression,
    linear_functional_variance,
    meanfield_kl_to_full,
    sample_full_beta,
    sample_meanfield_beta,
)


def add_draw_metrics(
    metric_rows: list[dict[str, float | str | int]],
    scenario: str,
    rep: int,
    method: str,
    effect_draws: np.ndarray,
    corrected_sample_draws: np.ndarray,
    corrected_population_draws: np.ndarray,
    raw_population_draws: np.ndarray,
    sample_true: float,
    population_true: float,
    level: float,
    corrected_suffix: str,
) -> None:
    draw_map = {
        ("sample_cate", method): (sample_cate_draws(effect_draws), sample_true),
        ("population_ate_bb", method): (raw_population_draws, population_true),
        ("sample_cate", f"{method}_onestep_{corrected_suffix}"): (
            corrected_sample_draws,
            sample_true,
        ),
        ("population_ate_bb", f"{method}_onestep_{corrected_suffix}"): (
            corrected_population_draws,
            population_true,
        ),
    }
    for (estimand, row_method), (draws, true_value) in draw_map.items():
        metric_rows.append(
            {
                "scenario": scenario,
                "rep": rep,
                "method": row_method,
                "estimand": estimand,
                **posterior_draw_metrics(draws, true_value, level=level),
            }
        )


def run(args: argparse.Namespace) -> None:
    output_dir = ROOT / "outputs" / "module4"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows = []
    diagnostic_rows = []
    metric_rows = []

    scenarios = available_scenarios()
    for scenario_idx, (overlap, effect) in enumerate(scenarios):
        scenario = f"{overlap}_{effect}"
        dgp = SyntheticCausalDGP(
            n_features=args.n_features,
            overlap=overlap,
            effect=effect,
            noise_sd=args.noise_sd,
        )
        for rep in range(args.reps):
            seed = args.seed + rep + 10_000 * scenario_idx
            posterior_seed = args.seed + 1_000_000 + rep + 10_000 * scenario_idx
            rng = np.random.default_rng(posterior_seed)

            data = dgp.sample(args.n, seed=seed)
            pi_used = propensity_for_model(
                data,
                model=args.pi_model,
                ridge=args.pi_logistic_ridge,
                max_iter=args.pi_logistic_max_iter,
                tolerance=args.pi_logistic_tolerance,
                clip=args.pi_clip,
            )

            train_z = make_gp_inputs(data["x"], data["a"])
            initial_gp_config = ExactGPConfig(
                signal_sd=args.signal_sd,
                lengthscale=args.lengthscale,
                treatment_lengthscale=args.treatment_lengthscale,
                noise_sd=args.noise_sd,
                jitter=args.jitter,
                center_y=not args.no_center_y,
            )
            mml_result = None
            selected_gp_config = initial_gp_config
            if args.lengthscale_mode in {"mml", "map"}:
                mml_result = fit_gp_lengthscales_empirical_bayes(
                    train_z,
                    data["y"],
                    initial_gp_config,
                    selection_method=args.lengthscale_mode,
                    prior_medians=(
                        args.map_prior_lx_median
                        if args.map_prior_lx_median is not None
                        else args.lengthscale,
                        args.map_prior_la_median
                        if args.map_prior_la_median is not None
                        else args.treatment_lengthscale,
                    ),
                    prior_log_sds=(
                        args.map_prior_log_sd_x,
                        args.map_prior_log_sd_a,
                    ),
                    bounds=(
                        args.mml_lengthscale_min,
                        args.mml_lengthscale_max,
                    ),
                    n_restarts=args.mml_restarts,
                    maxiter=args.mml_maxiter,
                    seed=seed,
                )
                selected_gp_config = mml_result.config
            config = RFFConfig(
                n_rff=args.n_rff,
                signal_sd=args.signal_sd,
                lengthscale=selected_gp_config.lengthscale,
                treatment_lengthscale=(
                    selected_gp_config.treatment_lengthscale
                ),
                beta_prior_sd=args.beta_prior_sd,
                noise_sd=args.noise_sd,
                jitter=args.jitter,
                center_y=not args.no_center_y,
            )
            dataset_rows.append(
                {
                    "scenario": scenario,
                    "rep": rep,
                    "lengthscale_mode": args.lengthscale_mode,
                    "lengthscale": selected_gp_config.lengthscale,
                    "treatment_lengthscale": (
                        selected_gp_config.treatment_lengthscale
                    ),
                    **summarize_dataset(data),
                }
            )
            control_z, treated_z = make_counterfactual_inputs(data["x"])
            posterior = fit_rff_bayesian_regression(train_z, data["y"], config, rng)

            phi0 = posterior.feature_map.transform(control_z)
            phi1 = posterior.feature_map.transform(treated_z)
            sample_cate_coefficient = (phi1 - phi0).mean(axis=0)
            full_cate_var = linear_functional_variance(
                posterior,
                sample_cate_coefficient,
            )
            meanfield_cate_var = linear_functional_variance(
                posterior,
                sample_cate_coefficient,
                meanfield=True,
            )
            coefficient_ratios = coefficient_variance_ratios(posterior)
            diagnostic_rows.append(
                {
                    "scenario": scenario,
                    "rep": rep,
                    "lengthscale_mode": args.lengthscale_mode,
                    "lengthscale": selected_gp_config.lengthscale,
                    "treatment_lengthscale": (
                        selected_gp_config.treatment_lengthscale
                    ),
                    "mml_log_marginal_likelihood": (
                        mml_result.log_marginal_likelihood
                        if mml_result is not None
                        else np.nan
                    ),
                    "selection_log_prior": (
                        mml_result.log_prior
                        if mml_result is not None
                        else np.nan
                    ),
                    "selection_log_posterior": (
                        mml_result.log_posterior
                        if mml_result is not None
                        else np.nan
                    ),
                    "mml_success": (
                        int(mml_result.success)
                        if mml_result is not None
                        else np.nan
                    ),
                    "mml_at_boundary": (
                        int(mml_result.at_boundary)
                        if mml_result is not None
                        else np.nan
                    ),
                    "meanfield_kl_to_full": meanfield_kl_to_full(posterior),
                    "coefficient_var_ratio_min": float(coefficient_ratios.min()),
                    "coefficient_var_ratio_median": float(np.median(coefficient_ratios)),
                    "coefficient_var_ratio_max": float(coefficient_ratios.max()),
                    "sample_cate_full_sd_analytic": float(np.sqrt(full_cate_var)),
                    "sample_cate_meanfield_sd_analytic": float(
                        np.sqrt(meanfield_cate_var)
                    ),
                    "sample_cate_sd_ratio_analytic": float(
                        np.sqrt(meanfield_cate_var / full_cate_var)
                    ),
                }
            )

            posterior_types = {
                "rff_full_gaussian": sample_full_beta(posterior, args.draws, rng, args.jitter),
                "rff_meanfield_vi": sample_meanfield_beta(posterior, args.draws, rng),
            }

            for method, beta_draws in posterior_types.items():
                mu0_draws, mu1_draws = counterfactual_draws_from_beta(
                    beta_draws,
                    control_z,
                    treated_z,
                    posterior,
                )
                effect_draws = mu1_draws - mu0_draws
                eif_y = outcome_eif_draws(
                    mu0_draws=mu0_draws,
                    mu1_draws=mu1_draws,
                    a=data["a"],
                    y=data["y"],
                    pi=pi_used,
                    pi_clip=args.pi_clip,
                )
                raw_population = population_ate_bb_draws(effect_draws, rng)
                corrected_sample = one_step_sample_cate_draws(effect_draws, eif_y, rng)
                corrected_population = one_step_population_ate_draws(effect_draws, eif_y, rng)

                add_draw_metrics(
                    metric_rows=metric_rows,
                    scenario=scenario,
                    rep=rep,
                    method=method,
                    effect_draws=effect_draws,
                    corrected_sample_draws=corrected_sample,
                    corrected_population_draws=corrected_population,
                    raw_population_draws=raw_population,
                    sample_true=float(data["tau"].mean()),
                    population_true=dgp.true_ate,
                    level=args.level,
                    corrected_suffix=correction_suffix(args.pi_model),
                )

            print(f"finished {scenario} rep {rep + 1}/{args.reps}", flush=True)

    dataset_df = pd.DataFrame(dataset_rows)
    diagnostic_df = pd.DataFrame(diagnostic_rows)
    metric_df = pd.DataFrame(metric_rows)

    aggregate_rows = []
    for (_, _, _), group in metric_df.groupby(["scenario", "estimand", "method"], sort=True):
        aggregate_rows.append(aggregate_metric_rows(group.to_dict("records")))
    aggregate_df = pd.DataFrame(aggregate_rows).sort_values(["scenario", "estimand", "method"])

    dataset_path = output_dir / "meanfield_vi_dataset_summaries.csv"
    diagnostic_path = output_dir / "meanfield_vi_diagnostics.csv"
    metric_path = output_dir / "meanfield_vi_posterior_metrics.csv"
    aggregate_path = output_dir / "meanfield_vi_posterior_metrics_aggregated.csv"
    dataset_df.to_csv(dataset_path, index=False)
    diagnostic_df.to_csv(diagnostic_path, index=False)
    metric_df.to_csv(metric_path, index=False)
    aggregate_df.to_csv(aggregate_path, index=False)

    print(f"Wrote {dataset_path}")
    print(f"Wrote {diagnostic_path}")
    print(f"Wrote {metric_path}")
    print(f"Wrote {aggregate_path}")
    print()
    columns = [
        "scenario",
        "estimand",
        "method",
        "bias_mean",
        "abs_error_mean",
        "ci_width_mean",
        "covered_mean",
        "posterior_sd_mean",
    ]
    print(aggregate_df[columns].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=250)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--n-features", type=int, default=5)
    parser.add_argument("--n-rff", type=int, default=200)
    parser.add_argument("--noise-sd", type=float, default=1.0)
    parser.add_argument("--signal-sd", type=float, default=1.0)
    parser.add_argument("--lengthscale", type=float, default=0.8)
    parser.add_argument("--treatment-lengthscale", type=float, default=0.5)
    parser.add_argument(
        "--lengthscale-mode", choices=["mml", "map", "fixed"], default="map"
    )
    parser.add_argument("--mml-lengthscale-min", type=float, default=0.05)
    parser.add_argument("--mml-lengthscale-max", type=float, default=50.0)
    parser.add_argument("--mml-restarts", type=int, default=3)
    parser.add_argument("--mml-maxiter", type=int, default=100)
    parser.add_argument("--map-prior-lx-median", type=float)
    parser.add_argument("--map-prior-la-median", type=float)
    parser.add_argument("--map-prior-log-sd-x", type=float, default=0.5)
    parser.add_argument("--map-prior-log-sd-a", type=float, default=0.5)
    parser.add_argument("--beta-prior-sd", type=float, default=1.0)
    parser.add_argument("--jitter", type=float, default=1e-8)
    parser.add_argument("--pi-clip", type=float, default=0.02)
    parser.add_argument("--level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--pi-model", choices=["oracle", "logistic"], default="oracle")
    parser.add_argument("--pi-logistic-ridge", type=float, default=1.0)
    parser.add_argument("--pi-logistic-max-iter", type=int, default=100)
    parser.add_argument("--pi-logistic-tolerance", type=float, default=1e-8)
    parser.add_argument("--no-center-y", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
