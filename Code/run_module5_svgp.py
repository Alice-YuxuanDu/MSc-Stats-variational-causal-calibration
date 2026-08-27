"""Run paired exact-GP and sparse variational GP calibration experiments."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from run_module4_experiment_grid import (
    bootstrap_weights,
    correction_suffix,
    overlap_diagnostics,
    parse_int_grid,
    parse_pi_clip_grid,
    posterior_draw_sets,
    propensity_for_model,
    scenario_map,
    wilson_interval,
)
from src.dgp import SyntheticCausalDGP, summarize_dataset
from src.gp_exact import (
    ExactGPConfig,
    fit_exact_gp_predictive,
    fit_gp_lengthscales_empirical_bayes,
    make_counterfactual_inputs,
    make_gp_inputs,
    sample_counterfactual_functions,
)
from src.metrics import posterior_draw_metrics
from src.posterior_correction import outcome_eif_draws
from src.svgp import (
    SVGPConfig,
    exact_gp_log_marginal_likelihood,
    fit_svgp,
    fit_svgp_elbo_restart,
    linear_functional_variance,
    predict_svgp,
    sample_svgp_counterfactuals,
)


def file_sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_rng(
    seed: int,
    scenario_idx: int,
    rep: int,
    label: int
):
    return np.random.default_rng(
        np.random.SeedSequence([seed, scenario_idx, rep, label])
    )


def add_metric_rows(
    rows: list[dict[str, float | int | str]],
    scenario: str,
    rep: int,
    n_inducing: int,
    pi_clip: float,
    method: str,
    effect_draws: np.ndarray,
    eif_draws: np.ndarray,
    weights: tuple[np.ndarray, np.ndarray, np.ndarray],
    sample_true: float,
    population_true: float,
    level: float,
    corrected_suffix: str
):
    draw_sets = posterior_draw_sets(
        effect_draws,
        eif_draws,
        *weights,
    )
    entries = (
        (
            "sample_ate",
            method,
            draw_sets["sample_ate_raw"],
            sample_true,
        ),
        (
            "population_ate_bb",
            method,
            draw_sets["population_ate_bb_raw"],
            population_true,
        ),
        (
            "sample_ate",
            f"{method}_onestep_{corrected_suffix}",
            draw_sets["sample_ate_corrected"],
            sample_true,
        ),
        (
            "population_ate_bb",
            f"{method}_onestep_{corrected_suffix}",
            draw_sets["population_ate_bb_corrected"],
            population_true,
        ),
    )
    for estimand, row_method, draws, truth in entries:
        rows.append(
            {
                "scenario": scenario,
                "rep": rep,
                "n_inducing": n_inducing,
                "pi_clip": pi_clip,
                "method": row_method,
                "estimand": estimand,
                "true_value": truth,
                **posterior_draw_metrics(draws, truth, level),
            }
        )


def aggregate_metrics(metric_df: pd.DataFrame, level: float):
    keys = ["scenario", "n_inducing", "pi_clip", "estimand", "method"]
    rows = []
    for key, group in metric_df.groupby(keys, sort=True):
        n_reps = int(group.shape[0])
        coverage = float(group["covered"].mean())
        empirical_sd = float(group["posterior_mean"].std(ddof=1))
        posterior_sd = float(group["posterior_sd"].mean())
        lower, upper = wilson_interval(
            int(group["covered"].sum()),
            n_reps,
        )
        rows.append(
            {
                **dict(zip(keys, key)),
                "n_reps": n_reps,
                "bias": float(group["bias"].mean()),
                "bias_mcse": float(
                    group["bias"].std(ddof=1) / np.sqrt(n_reps)
                ),
                "mean_abs_error": float(group["abs_error"].mean()),
                "rmse": float(np.sqrt(group["squared_error"].mean())),
                "coverage": coverage,
                "coverage_mcse": float(
                    np.sqrt(coverage * (1.0 - coverage) / n_reps)
                ),
                "coverage_wilson_lower": lower,
                "coverage_wilson_upper": upper,
                "nominal_coverage": level,
                "mean_ci_width": float(group["ci_width"].mean()),
                "empirical_sd_posterior_mean": empirical_sd,
                "mean_posterior_sd": posterior_sd,
                "empirical_to_posterior_sd_ratio": (
                    empirical_sd / posterior_sd
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_comparisons(metric_df: pd.DataFrame):
    index = ["scenario", "rep", "n_inducing", "pi_clip", "estimand"]
    wide = metric_df.pivot(
        index=index,
        columns="method",
        values=["posterior_mean", "abs_error", "posterior_sd", "ci_width"],
    )
    methods = set(metric_df["method"])
    corrected_suffixes = sorted(
        method.removeprefix("exact_gp_onestep_")
        for method in methods
        if method.startswith("exact_gp_onestep_")
    )
    comparisons = [
        ("svgp_minus_exact_raw", "svgp", "exact_gp"),
    ]
    for suffix in corrected_suffixes:
        exact_corrected = f"exact_gp_onestep_{suffix}"
        svgp_corrected = f"svgp_onestep_{suffix}"
        comparisons.extend(
            [
                (
                    f"svgp_minus_exact_onestep_{suffix}",
                    svgp_corrected,
                    exact_corrected,
                ),
                (
                    f"onestep_{suffix}_minus_raw_svgp",
                    svgp_corrected,
                    "svgp",
                ),
            ]
        )
    group_keys = ["scenario", "n_inducing", "pi_clip", "estimand"]
    rows = []
    for label, left, right in comparisons:
        frame = pd.DataFrame(
            {
                "posterior_mean_difference": (
                    wide[("posterior_mean", left)]
                    - wide[("posterior_mean", right)]
                ),
                "abs_error_difference": (
                    wide[("abs_error", left)] - wide[("abs_error", right)]
                ),
                "posterior_sd_ratio": (
                    wide[("posterior_sd", left)]
                    / wide[("posterior_sd", right)]
                ),
                "ci_width_ratio": (
                    wide[("ci_width", left)] / wide[("ci_width", right)]
                ),
            }
        ).reset_index()
        for key, group in frame.groupby(group_keys, sort=True):
            rows.append(
                {
                    **dict(zip(group_keys, key)),
                    "comparison": label,
                    "n_reps": int(group.shape[0]),
                    "posterior_mean_difference": float(
                        group["posterior_mean_difference"].mean()
                    ),
                    "abs_error_difference": float(
                        group["abs_error_difference"].mean()
                    ),
                    "posterior_sd_ratio": float(
                        group["posterior_sd_ratio"].mean()
                    ),
                    "ci_width_ratio": float(group["ci_width_ratio"].mean()),
                }
            )
    return pd.DataFrame(rows)


def save_outputs(
    output_dir: Path,
    dataset_rows: list[dict[str, float | int | str]],
    diagnostic_rows: list[dict[str, float | int | str]],
    metric_rows: list[dict[str, float | int | str]],
    level: float
):
    dataset_df = pd.DataFrame(dataset_rows).drop_duplicates(
        ["scenario", "rep"],
        keep="last",
    )
    diagnostic_df = pd.DataFrame(diagnostic_rows).drop_duplicates(
        ["scenario", "rep", "n_inducing"],
        keep="last",
    )
    metric_df = pd.DataFrame(metric_rows).drop_duplicates(
        [
            "scenario",
            "rep",
            "n_inducing",
            "pi_clip",
            "method",
            "estimand",
        ],
        keep="last",
    )
    dataset_df.sort_values(["scenario", "rep"]).to_csv(
        output_dir / "dataset_summaries.csv",
        index=False,
    )
    diagnostic_df.sort_values(
        ["scenario", "n_inducing", "rep"]
    ).to_csv(output_dir / "diagnostics.csv", index=False)
    metric_df.sort_values(
        [
            "scenario",
            "n_inducing",
            "pi_clip",
            "estimand",
            "method",
            "rep",
        ]
    ).to_csv(output_dir / "posterior_metrics.csv", index=False)
    aggregate_metrics(metric_df, level).to_csv(
        output_dir / "posterior_metrics_aggregated.csv",
        index=False,
    )
    paired_comparisons(metric_df).to_csv(
        output_dir / "paired_comparisons.csv",
        index=False,
    )


def load_rows(path: Path):
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict("records")


def run(args: argparse.Namespace):
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root).expanduser().resolve() / run_name
    config_path = output_dir / "config.json"
    selected_scenarios = args.scenarios or list(scenario_map())
    selection_path = (
        Path(args.lengthscale_selection_csv).expanduser().resolve()
        if args.lengthscale_selection_csv
        else None
    )
    lengthscale_selections = {}
    if selection_path is not None:
        selection_df = pd.read_csv(selection_path)
        for _, row in selection_df.iterrows():
            key = (str(row["scenario"]), int(row["rep"]))
            values = (
                float(row["selected_lengthscale"]),
                float(row["selected_treatment_lengthscale"]),
                float(row.get("mml_initial_log_marginal_likelihood", np.nan)),
                float(row.get("mml_log_marginal_likelihood", np.nan)),
                float(row.get("mml_success", np.nan)),
                float(row.get("mml_at_boundary", np.nan)),
                float(row.get("selection_initial_log_prior", np.nan)),
                float(row.get("selection_log_prior", np.nan)),
                float(row.get("selection_initial_log_posterior", np.nan)),
                float(row.get("selection_log_posterior", np.nan)),
            )
            lengthscale_selections[key] = values
    source_paths = (
        Path(__file__),
        CODE_DIR / "src" / "svgp.py",
        CODE_DIR / "src" / "gp_exact.py",
        CODE_DIR / "src" / "posterior_correction.py",
    )
    config_record = {
        "run_name": run_name,
        "n": args.n,
        "reps": args.reps,
        "draws": args.draws,
        "n_features": args.n_features,
        "n_inducing_grid": args.n_inducing_grid,
        "pi_clip_grid": args.pi_clip_grid,
        "pi_model": args.pi_model,
        "pi_logistic_ridge": args.pi_logistic_ridge,
        "pi_logistic_max_iter": args.pi_logistic_max_iter,
        "pi_logistic_tolerance": args.pi_logistic_tolerance,
        "scenarios": selected_scenarios,
        "noise_sd": args.noise_sd,
        "signal_sd": args.signal_sd,
        "lengthscale": args.lengthscale,
        "treatment_lengthscale": args.treatment_lengthscale,
        "lengthscale_mode": args.lengthscale_mode,
        "mml_lengthscale_min": args.mml_lengthscale_min,
        "mml_lengthscale_max": args.mml_lengthscale_max,
        "mml_restarts": args.mml_restarts,
        "mml_maxiter": args.mml_maxiter,
        "map_prior_lx_median": (
            args.map_prior_lx_median
            if args.map_prior_lx_median is not None
            else args.lengthscale
        ),
        "map_prior_la_median": (
            args.map_prior_la_median
            if args.map_prior_la_median is not None
            else args.treatment_lengthscale
        ),
        "map_prior_log_sd_x": args.map_prior_log_sd_x,
        "map_prior_log_sd_a": args.map_prior_log_sd_a,
        "lengthscale_selection_csv": (
            str(selection_path) if selection_path is not None else None
        ),
        "lengthscale_selection_sha256": (
            file_sha256(selection_path)
            if selection_path is not None
            else None
        ),
        "inducing_location_mode": args.inducing_location_mode,
        "inducing_location_restarts": args.inducing_location_restarts,
        "jitter": args.jitter,
        "level": args.level,
        "seed": args.seed,
        "center_y": not args.no_center_y,
        "runtime_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "python": sys.version.split()[0],
        },
        "source_hashes": {
            path.name: file_sha256(path) for path in source_paths
        },
    }

    if output_dir.exists() and not args.resume:
        raise FileExistsError(
            f"{output_dir} already exists. Use --resume or a new --run-name."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and config_path.exists():
        if json.loads(config_path.read_text()) != config_record:
            raise ValueError("The existing run configuration does not match.")
    config_path.write_text(
        json.dumps(config_record, indent=2, sort_keys=True) + "\n"
    )

    dataset_rows = load_rows(output_dir / "dataset_summaries.csv")
    diagnostic_rows = load_rows(output_dir / "diagnostics.csv")
    metric_rows = load_rows(output_dir / "posterior_metrics.csv")
    completed = {
        (str(row["scenario"]), int(row["rep"]), int(row["n_inducing"]))
        for row in diagnostic_rows
    }
    existing_datasets = {
        (str(row["scenario"]), int(row["rep"])) for row in dataset_rows
    }

    scenarios = scenario_map()
    scenario_indices = {
        name: index for index, name in enumerate(scenarios)
    }
    completed_since_save = 0
    for scenario in selected_scenarios:
        scenario_idx = scenario_indices[scenario]
        overlap, effect = scenarios[scenario]
        dgp = SyntheticCausalDGP(
            n_features=args.n_features,
            overlap=overlap,
            effect=effect,
            noise_sd=args.noise_sd,
        )
        for rep in range(args.reps):
            data_seed = args.seed + rep + 10_000 * scenario_idx
            data = dgp.sample(args.n, seed=data_seed)
            if (scenario, rep) not in existing_datasets:
                dataset_rows.append(
                    {
                        "scenario": scenario,
                        "rep": rep,
                        "data_seed": data_seed,
                        **summarize_dataset(data),
                    }
                )
                existing_datasets.add((scenario, rep))

            train_z = make_gp_inputs(data["x"], data["a"])
            control_z, treated_z = make_counterfactual_inputs(data["x"])
            test_z = np.vstack([control_z, treated_z])
            exact_config = ExactGPConfig(
                signal_sd=args.signal_sd,
                lengthscale=args.lengthscale,
                treatment_lengthscale=args.treatment_lengthscale,
                noise_sd=args.noise_sd,
                jitter=args.jitter,
                center_y=not args.no_center_y,
            )
            mml_fit_seconds = 0.0
            mml_result = None
            selection = lengthscale_selections.get((scenario, rep))
            mml_initial_lml = np.nan
            mml_selected_lml = np.nan
            mml_success = np.nan
            mml_at_boundary = np.nan
            selection_initial_log_prior = np.nan
            selection_log_prior = np.nan
            selection_initial_log_posterior = np.nan
            selection_log_posterior = np.nan
            if selection_path is not None:
                exact_config = ExactGPConfig(
                    signal_sd=args.signal_sd,
                    lengthscale=selection[0],
                    treatment_lengthscale=selection[1],
                    noise_sd=args.noise_sd,
                    jitter=args.jitter,
                    center_y=not args.no_center_y,
                )
                (
                    mml_initial_lml,
                    mml_selected_lml,
                    mml_success,
                    mml_at_boundary,
                    selection_initial_log_prior,
                    selection_log_prior,
                    selection_initial_log_posterior,
                    selection_log_posterior,
                ) = selection[2:]
                lengthscale_selection_source = "csv"
            elif args.lengthscale_mode in {"mml", "map"}:
                mml_start = perf_counter()
                mml_result = fit_gp_lengthscales_empirical_bayes(
                    train_z,
                    data["y"],
                    exact_config,
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
                    seed=args.seed + rep + 10_000 * scenario_idx,
                )
                mml_fit_seconds = perf_counter() - mml_start
                exact_config = mml_result.config
                mml_initial_lml = mml_result.initial_log_marginal_likelihood
                mml_selected_lml = mml_result.log_marginal_likelihood
                mml_success = int(mml_result.success)
                mml_at_boundary = int(mml_result.at_boundary)
                selection_initial_log_prior = mml_result.initial_log_prior
                selection_log_prior = mml_result.log_prior
                selection_initial_log_posterior = (
                    mml_result.initial_log_posterior
                )
                selection_log_posterior = mml_result.log_posterior
                lengthscale_selection_source = "optimized"
                objective_label = (
                    "logPost"
                    if args.lengthscale_mode == "map"
                    else "logML"
                )
                objective_value = (
                    mml_result.log_posterior
                    if args.lengthscale_mode == "map"
                    else mml_result.log_marginal_likelihood
                )
                print(
                    f"{args.lengthscale_mode.upper()} "
                    f"{scenario} rep {rep + 1}: "
                    f"l_x={exact_config.lengthscale:.6g} "
                    f"l_a={exact_config.treatment_lengthscale:.6g} "
                    f"{objective_label}={objective_value:.3f}",
                    flush=True,
                )
            else:
                lengthscale_selection_source = "fixed"
            exact_start = perf_counter()
            exact = fit_exact_gp_predictive(
                train_z,
                data["y"],
                test_z,
                exact_config,
            )
            exact_fit_seconds = perf_counter() - exact_start
            exact_mu0, exact_mu1 = sample_counterfactual_functions(
                exact,
                n_units=args.n,
                n_draws=args.draws,
                rng=make_rng(args.seed, scenario_idx, rep, 71),
                jitter=args.jitter,
            )
            exact_effect = exact_mu1 - exact_mu0
            coefficient = np.concatenate(
                [
                    -np.ones(args.n) / args.n,
                    np.ones(args.n) / args.n,
                ]
            )
            exact_ate_var = float(
                max(coefficient @ exact.cov @ coefficient, 0.0)
            )
            exact_counterfactual_rms_sd = float(
                np.sqrt(np.mean(np.maximum(np.diag(exact.cov), 0.0)))
            )
            weights = (
                bootstrap_weights(
                    args.draws,
                    args.n,
                    make_rng(args.seed, scenario_idx, rep, 81),
                ),
                bootstrap_weights(
                    args.draws,
                    args.n,
                    make_rng(args.seed, scenario_idx, rep, 82),
                ),
                bootstrap_weights(
                    args.draws,
                    args.n,
                    make_rng(args.seed, scenario_idx, rep, 83),
                ),
            )

            pi_by_clip = {
                pi_clip: propensity_for_model(
                    data,
                    model=args.pi_model,
                    ridge=args.pi_logistic_ridge,
                    max_iter=args.pi_logistic_max_iter,
                    tolerance=args.pi_logistic_tolerance,
                    clip=pi_clip,
                )
                for pi_clip in args.pi_clip_grid
            }
            diagnostic_clip = args.pi_clip_grid[0]
            diagnostic_pi = pi_by_clip[diagnostic_clip]
            pi_error = diagnostic_pi - np.clip(
                data["pi"],
                diagnostic_clip,
                1.0 - diagnostic_clip,
            )

            exact_eif = {
                pi_clip: outcome_eif_draws(
                    exact_mu0,
                    exact_mu1,
                    data["a"],
                    data["y"],
                    pi_by_clip[pi_clip],
                    pi_clip,
                )
                for pi_clip in args.pi_clip_grid
            }

            for n_inducing in args.n_inducing_grid:
                if (scenario, rep, n_inducing) in completed:
                    continue
                svgp_config = SVGPConfig(
                    n_inducing=n_inducing,
                    signal_sd=args.signal_sd,
                    lengthscale=exact_config.lengthscale,
                    treatment_lengthscale=exact_config.treatment_lengthscale,
                    noise_sd=args.noise_sd,
                    jitter=args.jitter,
                    center_y=not args.no_center_y,
                )
                sparse_start = perf_counter()
                if args.inducing_location_mode == "balanced":
                    sparse = fit_svgp(train_z, data["y"], svgp_config)
                else:
                    sparse = fit_svgp_elbo_restart(
                        train_z,
                        data["y"],
                        svgp_config,
                        n_restarts=args.inducing_location_restarts,
                        seed=args.seed
                        + 1_000_000 * scenario_idx
                        + 10_000 * rep
                        + n_inducing,
                    )
                sparse_predictive = predict_svgp(sparse, test_z)
                sparse_fit_seconds = perf_counter() - sparse_start
                sparse_mu0, sparse_mu1 = sample_svgp_counterfactuals(
                    sparse_predictive,
                    n_units=args.n,
                    n_draws=args.draws,
                    rng=make_rng(
                        args.seed,
                        scenario_idx,
                        rep,
                        10_000 + n_inducing,
                    ),
                    jitter=args.jitter,
                )
                sparse_effect = sparse_mu1 - sparse_mu0
                sparse_ate_var = linear_functional_variance(
                    sparse_predictive,
                    coefficient,
                )
                exact_lml = exact_gp_log_marginal_likelihood(
                    train_z,
                    data["y"],
                    svgp_config,
                )
                predictive_mean_rmse = float(
                    np.sqrt(
                        np.mean(
                            (sparse_predictive.mean - exact.mean) ** 2
                        )
                    )
                )
                diagnostic_rows.append(
                    {
                        "scenario": scenario,
                        "rep": rep,
                        "n_inducing": n_inducing,
                        "inducing_location_mode": args.inducing_location_mode,
                        "inducing_location_restarts": (
                            args.inducing_location_restarts
                        ),
                        "pi_model": args.pi_model,
                        "lengthscale_mode": args.lengthscale_mode,
                        "lengthscale_selection_source": (
                            lengthscale_selection_source
                        ),
                        "initial_lengthscale": args.lengthscale,
                        "initial_treatment_lengthscale": (
                            args.treatment_lengthscale
                        ),
                        "selected_lengthscale": exact_config.lengthscale,
                        "selected_treatment_lengthscale": (
                            exact_config.treatment_lengthscale
                        ),
                        "mml_fit_seconds": mml_fit_seconds,
                        "mml_initial_log_marginal_likelihood": (
                            mml_initial_lml
                        ),
                        "mml_log_marginal_likelihood": (
                            mml_selected_lml
                        ),
                        "mml_success": mml_success,
                        "mml_at_boundary": mml_at_boundary,
                        "selection_initial_log_prior": (
                            selection_initial_log_prior
                        ),
                        "selection_log_prior": selection_log_prior,
                        "selection_initial_log_posterior": (
                            selection_initial_log_posterior
                        ),
                        "selection_log_posterior": (
                            selection_log_posterior
                        ),
                        "pi_diagnostic_clip": diagnostic_clip,
                        "pi_estimate_mean": float(diagnostic_pi.mean()),
                        "pi_estimate_min": float(diagnostic_pi.min()),
                        "pi_estimate_p05": float(
                            np.quantile(diagnostic_pi, 0.05)
                        ),
                        "pi_estimate_p95": float(
                            np.quantile(diagnostic_pi, 0.95)
                        ),
                        "pi_estimate_max": float(diagnostic_pi.max()),
                        "pi_estimate_true_rmse": float(
                            np.sqrt(np.mean(pi_error**2))
                        ),
                        "pi_estimate_true_mean_abs_error": float(
                            np.mean(np.abs(pi_error))
                        ),
                        **overlap_diagnostics(
                            data["a"],
                            diagnostic_pi,
                            diagnostic_clip,
                        ),
                        "exact_fit_seconds": exact_fit_seconds,
                        "svgp_fit_seconds": sparse_fit_seconds,
                        "speed_ratio_exact_over_svgp": (
                            exact_fit_seconds / sparse_fit_seconds
                        ),
                        "elbo": sparse.elbo,
                        "exact_log_marginal_likelihood": exact_lml,
                        "elbo_gap": exact_lml - sparse.elbo,
                        "trace_gap": sparse.trace_gap,
                        "relative_trace_gap": sparse.relative_trace_gap,
                        "exact_counterfactual_rms_posterior_sd": (
                            exact_counterfactual_rms_sd
                        ),
                        "predictive_mean_rmse_vs_exact": predictive_mean_rmse,
                        "predictive_mean_standardized_rmse_vs_exact": (
                            predictive_mean_rmse
                            / exact_counterfactual_rms_sd
                        ),
                        "predictive_cov_relative_frobenius": float(
                            np.linalg.norm(
                                sparse_predictive.cov - exact.cov,
                                ord="fro",
                            )
                            / np.linalg.norm(exact.cov, ord="fro")
                        ),
                        "sample_ate_mean_difference": float(
                            coefficient
                            @ (sparse_predictive.mean - exact.mean)
                        ),
                        "sample_ate_exact_sd_analytic": float(
                            np.sqrt(exact_ate_var)
                        ),
                        "sample_ate_svgp_sd_analytic": float(
                            np.sqrt(sparse_ate_var)
                        ),
                        "sample_ate_svgp_to_exact_sd_ratio": float(
                            np.sqrt(sparse_ate_var / exact_ate_var)
                        ),
                    }
                )

                for pi_clip in args.pi_clip_grid:
                    add_metric_rows(
                        metric_rows,
                        scenario,
                        rep,
                        n_inducing,
                        pi_clip,
                        "exact_gp",
                        exact_effect,
                        exact_eif[pi_clip],
                        weights,
                        float(data["tau"].mean()),
                        dgp.true_ate,
                        args.level,
                        correction_suffix(args.pi_model),
                    )
                    sparse_eif = outcome_eif_draws(
                        sparse_mu0,
                        sparse_mu1,
                        data["a"],
                        data["y"],
                        pi_by_clip[pi_clip],
                        pi_clip,
                    )
                    add_metric_rows(
                        metric_rows,
                        scenario,
                        rep,
                        n_inducing,
                        pi_clip,
                        "svgp",
                        sparse_effect,
                        sparse_eif,
                        weights,
                        float(data["tau"].mean()),
                        dgp.true_ate,
                        args.level,
                        correction_suffix(args.pi_model),
                    )

                completed.add((scenario, rep, n_inducing))
                completed_since_save += 1
                print(
                    f"finished {scenario} rep {rep + 1}/{args.reps} "
                    f"m={n_inducing} ({sparse_fit_seconds:.2f}s)",
                    flush=True,
                )
                if completed_since_save >= args.checkpoint_every:
                    save_outputs(
                        output_dir,
                        dataset_rows,
                        diagnostic_rows,
                        metric_rows,
                        args.level,
                    )
                    completed_since_save = 0

    save_outputs(
        output_dir,
        dataset_rows,
        diagnostic_rows,
        metric_rows,
        args.level,
    )
    return output_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name")
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "outputs" / "module5_svgp"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--n", type=int, default=250)
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--n-features", type=int, default=5)
    parser.add_argument(
        "--n-inducing-grid",
        type=parse_int_grid,
        default=[20, 40, 80, 120, 160],
    )
    parser.add_argument(
        "--pi-clip-grid",
        type=parse_pi_clip_grid,
        default=[0.02],
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=list(scenario_map()),
    )
    parser.add_argument("--noise-sd", type=float, default=1.0)
    parser.add_argument("--signal-sd", type=float, default=1.0)
    parser.add_argument("--lengthscale", type=float, default=0.8)
    parser.add_argument("--treatment-lengthscale", type=float, default=0.5)
    parser.add_argument(
        "--lengthscale-mode",
        choices=["mml", "map", "fixed"],
        default="map",
    )
    parser.add_argument("--mml-lengthscale-min", type=float, default=0.05)
    parser.add_argument("--mml-lengthscale-max", type=float, default=50.0)
    parser.add_argument("--mml-restarts", type=int, default=3)
    parser.add_argument("--mml-maxiter", type=int, default=100)
    parser.add_argument("--map-prior-lx-median", type=float)
    parser.add_argument("--map-prior-la-median", type=float)
    parser.add_argument("--map-prior-log-sd-x", type=float, default=0.5)
    parser.add_argument("--map-prior-log-sd-a", type=float, default=0.5)
    parser.add_argument(
        "--lengthscale-selection-csv",
        help=(
            "Reuse per-scenario/rep MML/MAP lengthscales from a Module 4 "
            "diagnostics.csv instead of optimizing them again."
        ),
    )
    parser.add_argument(
        "--inducing-location-mode",
        choices=["balanced", "elbo-restart"],
        default="balanced",
        help=(
            "balanced uses deterministic balanced maximin locations. "
            "elbo-restart chooses the best ELBO among balanced random restarts."
        ),
    )
    parser.add_argument("--inducing-location-restarts", type=int, default=8)
    parser.add_argument("--jitter", type=float, default=1e-8)
    parser.add_argument("--level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--pi-model", choices=["oracle", "logistic"], default="oracle")
    parser.add_argument("--pi-logistic-ridge", type=float, default=1.0)
    parser.add_argument("--pi-logistic-max-iter", type=int, default=100)
    parser.add_argument("--pi-logistic-tolerance", type=float, default=1e-8)
    parser.add_argument("--no-center-y", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    output = run(parse_args())
    print(f"Wrote experiment outputs to {output}")
