"""Run reproducible paired RFF mean-field VI calibration experiments."""

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

from src.dgp import SyntheticCausalDGP, available_scenarios, summarize_dataset
from src.gp_exact import (
    ExactGPConfig,
    fit_gp_lengthscales_empirical_bayes,
    make_counterfactual_inputs,
    make_gp_inputs,
)
from src.metrics import posterior_draw_metrics
from src.posterior_correction import outcome_eif_draws
from src.vi_rff import (
    RFFConfig,
    coefficient_variance_ratios,
    counterfactual_draws_from_beta,
    fit_rff_bayesian_regression,
    linear_functional_variance,
    linear_functional_variance_decomposition,
    meanfield_kl_to_full,
    sample_full_beta,
    sample_meanfield_beta,
)


METHODS = (
    "rff_full_gaussian",
    "rff_meanfield_vi",
    "rff_full_gaussian_onestep_oracle_pi",
    "rff_meanfield_vi_onestep_oracle_pi",
)
ESTIMANDS = ("sample_ate", "population_ate_bb")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_pi_clip_grid(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_grid(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def fit_logistic_propensity(
    x: np.ndarray,
    a: np.ndarray,
    ridge: float,
    max_iter: int,
    tolerance: float,
    clip: float,
) -> np.ndarray:
    """Estimate pi(X) with ridge-regularized logistic regression."""

    x = np.asarray(x, dtype=float)
    a = np.asarray(a, dtype=float)
    design = np.column_stack([np.ones(x.shape[0]), x])
    beta = np.zeros(design.shape[1])
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0

    for _ in range(max_iter):
        pi = sigmoid(design @ beta)
        weights = np.maximum(pi * (1.0 - pi), 1e-8)
        gradient = design.T @ (pi - a) + penalty @ beta
        hessian = (design.T * weights) @ design + penalty
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if np.max(np.abs(step)) < tolerance:
            break

    return np.clip(sigmoid(design @ beta), clip, 1.0 - clip)


def propensity_for_model(
    data: dict[str, np.ndarray],
    model: str,
    ridge: float,
    max_iter: int,
    tolerance: float,
    clip: float,
) -> np.ndarray:
    if model == "oracle":
        return np.clip(data["pi"], clip, 1.0 - clip)
    if model == "logistic":
        return fit_logistic_propensity(
            data["x"],
            data["a"],
            ridge=ridge,
            max_iter=max_iter,
            tolerance=tolerance,
            clip=clip,
        )
    raise ValueError(f"unknown propensity model: {model}")


def correction_suffix(pi_model: str) -> str:
    return "oracle_pi" if pi_model == "oracle" else f"{pi_model}_pi"


def scenario_map() -> dict[str, tuple[str, str]]:
    return {
        f"{overlap}_{effect}": (overlap, effect)
        for overlap, effect in available_scenarios()
    }


def make_rngs(
    base_seed: int,
    scenario_idx: int,
    rep: int,
    n_rff: int,
) -> tuple[np.random.Generator, ...]:
    seed_sequence = np.random.SeedSequence(
        [base_seed, scenario_idx, rep, n_rff]
    )
    return tuple(
        np.random.default_rng(child)
        for child in seed_sequence.spawn(3)
    )


def make_bootstrap_rngs(
    base_seed: int,
    scenario_idx: int,
    rep: int,
) -> tuple[np.random.Generator, ...]:
    seed_sequence = np.random.SeedSequence(
        [base_seed, scenario_idx, rep, 99173]
    )
    return tuple(
        np.random.default_rng(child)
        for child in seed_sequence.spawn(3)
    )


def bootstrap_weights(
    n_draws: int,
    n_units: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return rng.dirichlet(np.ones(n_units), size=n_draws)


def posterior_draw_sets(
    effect_draws: np.ndarray,
    eif_draws: np.ndarray,
    raw_population_weights: np.ndarray,
    corrected_sample_weights: np.ndarray,
    corrected_population_weights: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "sample_ate_raw": effect_draws.mean(axis=1),
        "population_ate_bb_raw": np.sum(
            raw_population_weights * effect_draws,
            axis=1,
        ),
        "sample_ate_corrected": (
            effect_draws.mean(axis=1)
            + np.sum(corrected_sample_weights * eif_draws, axis=1)
        ),
        "population_ate_bb_corrected": np.sum(
            corrected_population_weights * (effect_draws + eif_draws),
            axis=1,
        ),
    }


def add_metric_rows(
    rows: list[dict[str, float | int | str]],
    scenario: str,
    rep: int,
    n_rff: int,
    pi_clip: float,
    posterior_type: str,
    draw_sets: dict[str, np.ndarray],
    sample_true: float,
    population_true: float,
    level: float,
    corrected_suffix: str,
) -> None:
    corrected_method = f"{posterior_type}_onestep_{corrected_suffix}"
    entries = (
        ("sample_ate", posterior_type, draw_sets["sample_ate_raw"], sample_true),
        (
            "population_ate_bb",
            posterior_type,
            draw_sets["population_ate_bb_raw"],
            population_true,
        ),
        (
            "sample_ate",
            corrected_method,
            draw_sets["sample_ate_corrected"],
            sample_true,
        ),
        (
            "population_ate_bb",
            corrected_method,
            draw_sets["population_ate_bb_corrected"],
            population_true,
        ),
    )
    for estimand, method, draws, true_value in entries:
        rows.append(
            {
                "scenario": scenario,
                "rep": rep,
                "n_rff": n_rff,
                "pi_clip": pi_clip,
                "method": method,
                "estimand": estimand,
                "true_value": true_value,
                **posterior_draw_metrics(draws, true_value, level=level),
            }
        )


def overlap_diagnostics(
    a: np.ndarray,
    pi: np.ndarray,
    pi_clip: float,
) -> dict[str, float]:
    clipped_pi = np.clip(pi, pi_clip, 1.0 - pi_clip)
    observed_inverse_weight = (
        a / clipped_pi + (1.0 - a) / (1.0 - clipped_pi)
    )
    weight_sum = float(observed_inverse_weight.sum())
    weight_ess = weight_sum**2 / float(np.sum(observed_inverse_weight**2))
    return {
        "propensity_clip_fraction": float(
            np.mean((pi < pi_clip) | (pi > 1.0 - pi_clip))
        ),
        "observed_inverse_weight_max": float(observed_inverse_weight.max()),
        "observed_inverse_weight_p95": float(
            np.quantile(observed_inverse_weight, 0.95)
        ),
        "observed_inverse_weight_ess": weight_ess,
        "observed_inverse_weight_ess_fraction": weight_ess / a.size,
    }


def aggregate_metrics(metric_df: pd.DataFrame, level: float) -> pd.DataFrame:
    group_keys = ["scenario", "n_rff", "pi_clip", "estimand", "method"]
    rows = []
    for key, group in metric_df.groupby(group_keys, sort=True):
        coverage = float(group["covered"].mean())
        n_reps = int(group.shape[0])
        empirical_sd = float(group["posterior_mean"].std(ddof=1))
        mean_posterior_sd = float(group["posterior_sd"].mean())
        wilson_lower, wilson_upper = wilson_interval(
            int(group["covered"].sum()),
            n_reps,
        )
        rows.append(
            {
                **dict(zip(group_keys, key)),
                "n_reps": n_reps,
                "bias": float(group["bias"].mean()),
                "bias_mcse": float(group["bias"].std(ddof=1) / np.sqrt(n_reps)),
                "mean_abs_error": float(group["abs_error"].mean()),
                "median_abs_error": float(group["abs_error"].median()),
                "rmse": float(np.sqrt(group["squared_error"].mean())),
                "coverage": coverage,
                "coverage_mcse": float(
                    np.sqrt(coverage * (1.0 - coverage) / n_reps)
                ),
                "coverage_wilson_lower": wilson_lower,
                "coverage_wilson_upper": wilson_upper,
                "nominal_coverage": level,
                "mean_ci_width": float(group["ci_width"].mean()),
                "empirical_sd_posterior_mean": empirical_sd,
                "mean_posterior_sd": mean_posterior_sd,
                "empirical_to_posterior_sd_ratio": (
                    empirical_sd / mean_posterior_sd
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_comparisons(metric_df: pd.DataFrame) -> pd.DataFrame:
    index = ["scenario", "rep", "n_rff", "pi_clip", "estimand"]
    wide = metric_df.pivot(
        index=index,
        columns="method",
        values=["posterior_mean", "abs_error", "posterior_sd", "ci_width"],
    )
    methods = set(metric_df["method"])
    corrected_suffixes = sorted(
        method.removeprefix("rff_full_gaussian_onestep_")
        for method in methods
        if method.startswith("rff_full_gaussian_onestep_")
    )
    comparisons = [
        (
            "meanfield_minus_full_raw",
            "rff_meanfield_vi",
            "rff_full_gaussian",
        ),
    ]
    for suffix in corrected_suffixes:
        full_corrected = f"rff_full_gaussian_onestep_{suffix}"
        meanfield_corrected = f"rff_meanfield_vi_onestep_{suffix}"
        comparisons.extend(
            [
                (
                    f"meanfield_minus_full_onestep_{suffix}",
                    meanfield_corrected,
                    full_corrected,
                ),
                (
                    f"onestep_{suffix}_minus_raw_full",
                    full_corrected,
                    "rff_full_gaussian",
                ),
                (
                    f"onestep_{suffix}_minus_raw_meanfield",
                    meanfield_corrected,
                    "rff_meanfield_vi",
                ),
            ]
        )
    rows = []
    group_levels = ["scenario", "n_rff", "pi_clip", "estimand"]
    for comparison, left, right in comparisons:
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
        for key, group in frame.groupby(group_levels, sort=True):
            rows.append(
                {
                    **dict(zip(group_levels, key)),
                    "comparison": comparison,
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


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    half_width = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z**2 / (4.0 * total**2)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def save_outputs(
    output_dir: Path,
    dataset_rows: list[dict[str, float | int | str]],
    diagnostic_rows: list[dict[str, float | int | str]],
    metric_rows: list[dict[str, float | int | str]],
    level: float,
) -> None:
    dataset_df = pd.DataFrame(dataset_rows).drop_duplicates(
        ["scenario", "rep"],
        keep="last",
    )
    diagnostic_df = pd.DataFrame(diagnostic_rows).drop_duplicates(
        ["scenario", "rep", "n_rff", "pi_clip"],
        keep="last",
    )
    metric_df = pd.DataFrame(metric_rows).drop_duplicates(
        ["scenario", "rep", "n_rff", "pi_clip", "method", "estimand"],
        keep="last",
    )

    dataset_df.sort_values(["scenario", "rep"]).to_csv(
        output_dir / "dataset_summaries.csv",
        index=False,
    )
    diagnostic_df.sort_values(
        ["scenario", "n_rff", "pi_clip", "rep"]
    ).to_csv(output_dir / "diagnostics.csv", index=False)
    metric_df.sort_values(
        ["scenario", "n_rff", "pi_clip", "estimand", "method", "rep"]
    ).to_csv(output_dir / "posterior_metrics.csv", index=False)
    aggregate_metrics(metric_df, level).to_csv(
        output_dir / "posterior_metrics_aggregated.csv",
        index=False,
    )
    paired_comparisons(metric_df).to_csv(
        output_dir / "paired_comparisons.csv",
        index=False,
    )


def load_rows(path: Path) -> list[dict[str, float | int | str]]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict("records")


def run(args: argparse.Namespace) -> Path:
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root).expanduser().resolve()
    output_dir = output_root / run_name
    config_path = output_dir / "config.json"
    selected_scenarios = args.scenarios or list(scenario_map())

    config_record = {
        "run_name": run_name,
        "n": args.n,
        "reps": args.reps,
        "draws": args.draws,
        "n_features": args.n_features,
        "n_rff_grid": args.n_rff_grid,
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
        "beta_prior_sd": args.beta_prior_sd,
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
            path.name: file_sha256(path)
            for path in (
                Path(__file__),
                CODE_DIR / "src" / "dgp.py",
                CODE_DIR / "src" / "metrics.py",
                CODE_DIR / "src" / "posterior_correction.py",
                CODE_DIR / "src" / "gp_exact.py",
                CODE_DIR / "src" / "vi_rff.py",
            )
        },
    }

    if output_dir.exists() and not args.resume:
        raise FileExistsError(
            f"{output_dir} already exists. Use --resume or a new --run-name."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and config_path.exists():
        existing_config = json.loads(config_path.read_text())
        if existing_config != config_record:
            raise ValueError("The existing run configuration does not match.")
    config_path.write_text(json.dumps(config_record, indent=2, sort_keys=True) + "\n")

    dataset_rows = load_rows(output_dir / "dataset_summaries.csv")
    diagnostic_rows = load_rows(output_dir / "diagnostics.csv")
    metric_rows = load_rows(output_dir / "posterior_metrics.csv")
    completed = {
        (
            str(row["scenario"]),
            int(row["rep"]),
            int(row["n_rff"]),
            float(row["pi_clip"]),
        )
        for row in diagnostic_rows
    }
    existing_datasets = {
        (str(row["scenario"]), int(row["rep"]))
        for row in dataset_rows
    }

    scenarios = scenario_map()
    canonical_scenario_indices = {
        name: index for index, name in enumerate(scenarios)
    }
    completed_since_save = 0
    for scenario in selected_scenarios:
        scenario_idx = canonical_scenario_indices[scenario]
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
            initial_gp_config = ExactGPConfig(
                signal_sd=args.signal_sd,
                lengthscale=args.lengthscale,
                treatment_lengthscale=args.treatment_lengthscale,
                noise_sd=args.noise_sd,
                jitter=args.jitter,
                center_y=not args.no_center_y,
            )
            mml_fit_seconds = 0.0
            mml_result = None
            selected_gp_config = initial_gp_config
            if args.lengthscale_mode in {"mml", "map"}:
                mml_start = perf_counter()
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
                    seed=args.seed + rep + 10_000 * scenario_idx,
                )
                mml_fit_seconds = perf_counter() - mml_start
                selected_gp_config = mml_result.config
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
                    f"l_x={selected_gp_config.lengthscale:.6g} "
                    f"l_a={selected_gp_config.treatment_lengthscale:.6g} "
                    f"{objective_label}={objective_value:.3f}",
                    flush=True,
                )
            (
                raw_bb_rng,
                sample_bb_rng,
                population_bb_rng,
            ) = make_bootstrap_rngs(args.seed, scenario_idx, rep)
            raw_population_weights = bootstrap_weights(
                args.draws,
                args.n,
                raw_bb_rng,
            )
            corrected_sample_weights = bootstrap_weights(
                args.draws,
                args.n,
                sample_bb_rng,
            )
            corrected_population_weights = bootstrap_weights(
                args.draws,
                args.n,
                population_bb_rng,
            )
            for n_rff in args.n_rff_grid:
                pending_clips = [
                    pi_clip
                    for pi_clip in args.pi_clip_grid
                    if (scenario, rep, n_rff, pi_clip) not in completed
                ]
                if not pending_clips:
                    continue

                (
                    feature_rng,
                    full_rng,
                    meanfield_rng,
                ) = make_rngs(args.seed, scenario_idx, rep, n_rff)
                config = RFFConfig(
                    n_rff=n_rff,
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
                fit_start = perf_counter()
                posterior = fit_rff_bayesian_regression(
                    train_z,
                    data["y"],
                    config,
                    feature_rng,
                )
                fit_seconds = perf_counter() - fit_start

                phi0 = posterior.feature_map.transform(control_z)
                phi1 = posterior.feature_map.transform(treated_z)
                sample_ate_coefficient = (phi1 - phi0).mean(axis=0)
                full_ate_var = linear_functional_variance(
                    posterior,
                    sample_ate_coefficient,
                )
                meanfield_ate_var = linear_functional_variance(
                    posterior,
                    sample_ate_coefficient,
                    meanfield=True,
                )
                ate_var_decomposition = linear_functional_variance_decomposition(
                    posterior,
                    sample_ate_coefficient,
                )
                coefficient_ratios = coefficient_variance_ratios(posterior)

                beta_draws = {
                    "rff_full_gaussian": sample_full_beta(
                        posterior,
                        args.draws,
                        full_rng,
                        args.jitter,
                    ),
                    "rff_meanfield_vi": sample_meanfield_beta(
                        posterior,
                        args.draws,
                        meanfield_rng,
                    ),
                }
                counterfactual_draws = {
                    method: counterfactual_draws_from_beta(
                        draws,
                        control_z,
                        treated_z,
                        posterior,
                    )
                    for method, draws in beta_draws.items()
                }
                posterior_mu0 = (
                    posterior.feature_map.transform(control_z) @ posterior.mean
                    + posterior.y_mean
                )
                posterior_mu1 = (
                    posterior.feature_map.transform(treated_z) @ posterior.mean
                    + posterior.y_mean
                )
                posterior_effect = posterior_mu1 - posterior_mu0

                for pi_clip in pending_clips:
                    clip_start = perf_counter()
                    pi_used = propensity_for_model(
                        data,
                        model=args.pi_model,
                        ridge=args.pi_logistic_ridge,
                        max_iter=args.pi_logistic_max_iter,
                        tolerance=args.pi_logistic_tolerance,
                        clip=pi_clip,
                    )
                    pi_error = pi_used - np.clip(data["pi"], pi_clip, 1.0 - pi_clip)
                    center_eif = outcome_eif_draws(
                        posterior_mu0[None, :],
                        posterior_mu1[None, :],
                        data["a"],
                        data["y"],
                        pi_used,
                        pi_clip=pi_clip,
                    )[0]
                    diagnostic_rows.append(
                        {
                            "scenario": scenario,
                            "rep": rep,
                            "n_rff": n_rff,
                            "pi_clip": pi_clip,
                            "pi_model": args.pi_model,
                            "lengthscale_mode": args.lengthscale_mode,
                            "initial_lengthscale": args.lengthscale,
                            "initial_treatment_lengthscale": (
                                args.treatment_lengthscale
                            ),
                            "selected_lengthscale": (
                                selected_gp_config.lengthscale
                            ),
                            "selected_treatment_lengthscale": (
                                selected_gp_config.treatment_lengthscale
                            ),
                            "mml_fit_seconds": mml_fit_seconds,
                            "mml_initial_log_marginal_likelihood": (
                                mml_result.initial_log_marginal_likelihood
                                if mml_result is not None
                                else np.nan
                            ),
                            "mml_log_marginal_likelihood": (
                                mml_result.log_marginal_likelihood
                                if mml_result is not None
                                else np.nan
                            ),
                            "selection_initial_log_prior": (
                                mml_result.initial_log_prior
                                if mml_result is not None
                                else np.nan
                            ),
                            "selection_log_prior": (
                                mml_result.log_prior
                                if mml_result is not None
                                else np.nan
                            ),
                            "selection_initial_log_posterior": (
                                mml_result.initial_log_posterior
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
                            "pi_estimate_mean": float(pi_used.mean()),
                            "pi_estimate_min": float(pi_used.min()),
                            "pi_estimate_p05": float(np.quantile(pi_used, 0.05)),
                            "pi_estimate_p95": float(np.quantile(pi_used, 0.95)),
                            "pi_estimate_max": float(pi_used.max()),
                            "pi_estimate_true_rmse": float(
                                np.sqrt(np.mean(pi_error**2))
                            ),
                            "pi_estimate_true_mean_abs_error": float(
                                np.mean(np.abs(pi_error))
                            ),
                            "fit_seconds": fit_seconds,
                            "meanfield_kl_to_full": meanfield_kl_to_full(posterior),
                            "coefficient_var_ratio_min": float(
                                coefficient_ratios.min()
                            ),
                            "coefficient_var_ratio_median": float(
                                np.median(coefficient_ratios)
                            ),
                            "coefficient_var_ratio_max": float(
                                coefficient_ratios.max()
                            ),
                            "sample_ate_full_sd_analytic": float(
                                np.sqrt(full_ate_var)
                            ),
                            "sample_ate_full_var_total_analytic": float(
                                ate_var_decomposition["full_total"]
                            ),
                            "sample_ate_full_var_diag_analytic": float(
                                ate_var_decomposition["full_diagonal"]
                            ),
                            "sample_ate_full_var_cross_analytic": float(
                                ate_var_decomposition["full_cross"]
                            ),
                            "sample_ate_full_cross_to_diag_ratio_analytic": (
                                float(
                                    ate_var_decomposition["full_cross"]
                                    / ate_var_decomposition["full_diagonal"]
                                )
                            ),
                            "sample_ate_meanfield_sd_analytic": float(
                                np.sqrt(meanfield_ate_var)
                            ),
                            "sample_ate_meanfield_var_analytic": float(
                                ate_var_decomposition["meanfield_total"]
                            ),
                            "sample_ate_meanfield_to_full_var_ratio_analytic": (
                                float(
                                    ate_var_decomposition["meanfield_total"]
                                    / ate_var_decomposition["full_total"]
                                )
                            ),
                            "sample_ate_sd_ratio_analytic": float(
                                np.sqrt(meanfield_ate_var / full_ate_var)
                            ),
                            "center_eif_abs_max": float(
                                np.max(np.abs(center_eif))
                            ),
                            "center_eif_abs_p95": float(
                                np.quantile(np.abs(center_eif), 0.95)
                            ),
                            "center_aipw_signal_sd": float(
                                np.std(posterior_effect + center_eif, ddof=1)
                            ),
                            "center_correction_mean": float(center_eif.mean()),
                            **overlap_diagnostics(
                                data["a"],
                                pi_used,
                                pi_clip,
                            ),
                        }
                    )

                    for method, (mu0_draws, mu1_draws) in (
                        counterfactual_draws.items()
                    ):
                        effect_draws = mu1_draws - mu0_draws
                        eif_draws = outcome_eif_draws(
                            mu0_draws,
                            mu1_draws,
                            data["a"],
                            data["y"],
                            pi_used,
                            pi_clip=pi_clip,
                        )
                        draw_sets = posterior_draw_sets(
                            effect_draws,
                            eif_draws,
                            raw_population_weights,
                            corrected_sample_weights,
                            corrected_population_weights,
                        )
                        add_metric_rows(
                            metric_rows,
                            scenario,
                            rep,
                            n_rff,
                            pi_clip,
                            method,
                            draw_sets,
                            sample_true=float(data["tau"].mean()),
                            population_true=dgp.true_ate,
                            level=args.level,
                            corrected_suffix=correction_suffix(args.pi_model),
                        )

                    completed.add((scenario, rep, n_rff, pi_clip))
                    completed_since_save += 1
                    print(
                        f"finished {scenario} rep {rep + 1}/{args.reps} "
                        f"n_rff={n_rff} pi_clip={pi_clip:g} "
                        f"({perf_counter() - clip_start:.2f}s)",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name")
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "outputs" / "module4_experiments"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--n", type=int, default=250)
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--n-features", type=int, default=5)
    parser.add_argument("--n-rff-grid", type=parse_int_grid, default=[200])
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
    parser.add_argument("--beta-prior-sd", type=float, default=1.0)
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
