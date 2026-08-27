"""Run IHDP raw RFF full posterior versus mean-field VI diagnostics."""

import argparse
import csv
from datetime import datetime
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

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


RAW_METHODS = ("rff_full_gaussian", "rff_meanfield_vi")
CORRECTED_SUFFIX = "_onestep_logistic_pi"
DIAGNOSTIC_COLUMNS = (
    "realization",
    "n",
    "n_features",
    "n_rff",
    "signal_sd",
    "lengthscale_mode",
    "initial_lengthscale",
    "initial_treatment_lengthscale",
    "lengthscale",
    "treatment_lengthscale",
    "noise_sd",
    "mml_fit_seconds",
    "mml_initial_log_marginal_likelihood",
    "mml_log_marginal_likelihood",
    "mml_success",
    "mml_n_iterations",
    "mml_n_function_evaluations",
    "mml_at_boundary",
    "selection_initial_log_prior",
    "selection_log_prior",
    "selection_initial_log_posterior",
    "selection_log_posterior",
    "fit_seconds",
    "treated_share",
    "pi_model",
    "pi_clip",
    "pi_min",
    "pi_p05",
    "pi_mean",
    "pi_p95",
    "pi_max",
    "pi_clip_fraction",
    "x_standardized",
    "y_standardized",
    "y_mean",
    "y_sd",
    "sample_ate_true",
    "ihdp_ate_field",
    "meanfield_kl_to_full",
    "coefficient_var_ratio_min",
    "coefficient_var_ratio_median",
    "coefficient_var_ratio_max",
    "sample_ate_full_sd_analytic",
    "sample_ate_full_var_total_analytic",
    "sample_ate_full_var_diag_analytic",
    "sample_ate_full_var_cross_analytic",
    "sample_ate_full_cross_to_diag_ratio_analytic",
    "sample_ate_meanfield_sd_analytic",
    "sample_ate_meanfield_var_analytic",
    "sample_ate_meanfield_to_full_var_ratio_analytic",
    "sample_ate_sd_ratio_analytic",
)
METRIC_COLUMNS = (
    "realization",
    "n_rff",
    "signal_sd",
    "lengthscale_mode",
    "initial_lengthscale",
    "initial_treatment_lengthscale",
    "lengthscale",
    "treatment_lengthscale",
    "noise_sd",
    "method",
    "estimand",
    "true_value",
    "posterior_mean",
    "bias",
    "abs_error",
    "squared_error",
    "ci_lower",
    "ci_upper",
    "ci_width",
    "covered",
    "posterior_sd",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_int_grid(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_grid(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def make_rngs(
    base_seed: int,
    realization: int,
    n_rff: int,
    config_index: int,
) -> tuple[np.random.Generator, ...]:
    seed_sequence = np.random.SeedSequence(
        [base_seed, realization, n_rff, config_index]
    )
    return tuple(np.random.default_rng(child) for child in seed_sequence.spawn(4))


def bootstrap_weights(
    n_draws: int,
    n_units: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return rng.dirichlet(np.ones(n_units), size=n_draws)


def fit_logistic_propensity(
    x: np.ndarray,
    a: np.ndarray,
    clip: float,
    c: float,
) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        C=c,
        max_iter=2000,
        solver="lbfgs",
    )
    model.fit(x, a.astype(int))
    pi = model.predict_proba(x)[:, 1]
    return np.clip(pi, clip, 1.0 - clip)


def propensity_diagnostics(pi: np.ndarray, clip: float) -> dict[str, float]:
    return {
        "pi_min": float(pi.min()),
        "pi_p05": float(np.quantile(pi, 0.05)),
        "pi_mean": float(pi.mean()),
        "pi_p95": float(np.quantile(pi, 0.95)),
        "pi_max": float(pi.max()),
        "pi_clip_fraction": float(np.mean((pi <= clip) | (pi >= 1.0 - clip))),
    }


def load_ihdp_realization(
    data: np.lib.npyio.NpzFile,
    realization: int,
    standardize_x: bool,
    standardize_y: bool,
) -> dict[str, np.ndarray | float]:
    x = np.asarray(data["x"][:, :, realization], dtype=float)
    if standardize_x:
        scale = x.std(axis=0, ddof=1)
        scale = np.where(scale > 0.0, scale, 1.0)
        x = (x - x.mean(axis=0)) / scale
    a = np.asarray(data["t"][:, realization], dtype=float)
    y = np.asarray(data["yf"][:, realization], dtype=float)
    y_mean = float(y.mean())
    y_sd = float(y.std(ddof=1))
    y_model = (y - y_mean) / y_sd if standardize_y else y
    mu0 = np.asarray(data["mu0"][:, realization], dtype=float)
    mu1 = np.asarray(data["mu1"][:, realization], dtype=float)
    tau = mu1 - mu0
    return {
        "x": x,
        "a": a,
        "y": y,
        "y_model": y_model,
        "y_mean": y_mean,
        "y_sd": y_sd if standardize_y else 1.0,
        "mu0": mu0,
        "mu1": mu1,
        "tau": tau,
        "sample_ate_true": float(tau.mean()),
        "ihdp_ate_field": float(np.asarray(data["ate"])),
    }


def aggregate_metric_rows(rows: list[dict[str, object]], level: float) -> list[dict[str, object]]:
    group_keys = (
        "n_rff",
        "signal_sd",
        "noise_sd",
        "lengthscale_mode",
        "initial_lengthscale",
        "initial_treatment_lengthscale",
        "estimand",
        "method",
    )
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[item] for item in group_keys)
        groups.setdefault(key, []).append(row)

    aggregate_rows = []
    for key, group in sorted(groups.items()):
        n_reps = len(group)
        covered = np.array([float(row["covered"]) for row in group], dtype=float)
        posterior_mean = np.array(
            [float(row["posterior_mean"]) for row in group],
            dtype=float,
        )
        posterior_sd = np.array(
            [float(row["posterior_sd"]) for row in group],
            dtype=float,
        )
        bias = np.array([float(row["bias"]) for row in group], dtype=float)
        abs_error = np.array([float(row["abs_error"]) for row in group], dtype=float)
        squared_error = np.array(
            [float(row["squared_error"]) for row in group],
            dtype=float,
        )
        ci_width = np.array([float(row["ci_width"]) for row in group], dtype=float)
        selected_lengthscales = np.array(
            [float(row["lengthscale"]) for row in group], dtype=float
        )
        selected_treatment_lengthscales = np.array(
            [float(row["treatment_lengthscale"]) for row in group], dtype=float
        )
        coverage = float(covered.mean())
        aggregate_rows.append(
            {
                **dict(zip(group_keys, key)),
                "selected_lengthscale_mean": float(selected_lengthscales.mean()),
                "selected_lengthscale_min": float(selected_lengthscales.min()),
                "selected_lengthscale_max": float(selected_lengthscales.max()),
                "selected_treatment_lengthscale_mean": float(
                    selected_treatment_lengthscales.mean()
                ),
                "selected_treatment_lengthscale_min": float(
                    selected_treatment_lengthscales.min()
                ),
                "selected_treatment_lengthscale_max": float(
                    selected_treatment_lengthscales.max()
                ),
                "n_reps": n_reps,
                "bias": float(bias.mean()),
                "bias_mcse": float(bias.std(ddof=1) / np.sqrt(n_reps))
                if n_reps > 1
                else 0.0,
                "mean_abs_error": float(abs_error.mean()),
                "median_abs_error": float(np.median(abs_error)),
                "rmse": float(np.sqrt(squared_error.mean())),
                "coverage": coverage,
                "coverage_mcse": float(np.sqrt(coverage * (1.0 - coverage) / n_reps)),
                "nominal_coverage": level,
                "mean_ci_width": float(ci_width.mean()),
                "empirical_sd_posterior_mean": float(posterior_mean.std(ddof=1))
                if n_reps > 1
                else 0.0,
                "mean_posterior_sd": float(posterior_sd.mean()),
            }
        )
    return aggregate_rows


def paired_comparisons(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], dict[str, dict[str, object]]] = {}
    for row in metric_rows:
        key = (
            int(row["realization"]),
            int(row["n_rff"]),
            float(row["signal_sd"]),
            float(row["noise_sd"]),
            str(row["lengthscale_mode"]),
            float(row["initial_lengthscale"]),
            float(row["initial_treatment_lengthscale"]),
        )
        grouped.setdefault(key, {})[str(row["method"])] = row

    comparison_specs = (
        (
            "meanfield_minus_full_raw",
            "rff_meanfield_vi",
            "rff_full_gaussian",
        ),
        (
            "meanfield_minus_full_onestep_logistic_pi",
            f"rff_meanfield_vi{CORRECTED_SUFFIX}",
            f"rff_full_gaussian{CORRECTED_SUFFIX}",
        ),
        (
            "onestep_minus_raw_full",
            f"rff_full_gaussian{CORRECTED_SUFFIX}",
            "rff_full_gaussian",
        ),
        (
            "onestep_minus_raw_meanfield",
            f"rff_meanfield_vi{CORRECTED_SUFFIX}",
            "rff_meanfield_vi",
        ),
    )
    by_config: dict[tuple[object, ...], dict[str, list[dict[str, float]]]] = {}
    for key, methods in grouped.items():
        (
            _,
            n_rff,
            signal_sd,
            noise_sd,
            lengthscale_mode,
            initial_lengthscale,
            initial_treatment_lengthscale,
        ) = key
        config_key = (
            n_rff,
            signal_sd,
            noise_sd,
            lengthscale_mode,
            initial_lengthscale,
            initial_treatment_lengthscale,
        )
        for comparison, left_name, right_name in comparison_specs:
            left = methods[left_name]
            right = methods[right_name]
            by_config.setdefault(config_key, {}).setdefault(comparison, []).append(
                {
                    "posterior_mean_difference": float(left["posterior_mean"])
                    - float(right["posterior_mean"]),
                    "abs_error_difference": float(left["abs_error"])
                    - float(right["abs_error"]),
                    "posterior_sd_ratio": float(left["posterior_sd"])
                    / float(right["posterior_sd"]),
                    "ci_width_ratio": float(left["ci_width"])
                    / float(right["ci_width"]),
                    "selected_lengthscale": float(left["lengthscale"]),
                    "selected_treatment_lengthscale": float(
                        left["treatment_lengthscale"]
                    ),
                }
            )

    rows = []
    for config_key, comparison_groups in sorted(by_config.items()):
        (
            n_rff,
            signal_sd,
            noise_sd,
            lengthscale_mode,
            initial_lengthscale,
            initial_treatment_lengthscale,
        ) = config_key
        for comparison, group in sorted(comparison_groups.items()):
            rows.append(
                {
                    "n_rff": n_rff,
                    "signal_sd": signal_sd,
                    "noise_sd": noise_sd,
                    "lengthscale_mode": lengthscale_mode,
                    "initial_lengthscale": initial_lengthscale,
                    "initial_treatment_lengthscale": (
                        initial_treatment_lengthscale
                    ),
                    "selected_lengthscale_mean": float(
                        np.mean([row["selected_lengthscale"] for row in group])
                    ),
                    "selected_treatment_lengthscale_mean": float(
                        np.mean(
                            [
                                row["selected_treatment_lengthscale"]
                                for row in group
                            ]
                        )
                    ),
                    "estimand": "sample_ate",
                    "comparison": comparison,
                    "n_reps": len(group),
                    "posterior_mean_difference": float(
                        np.mean(
                            [row["posterior_mean_difference"] for row in group]
                        )
                    ),
                    "abs_error_difference": float(
                        np.mean([row["abs_error_difference"] for row in group])
                    ),
                    "posterior_sd_ratio": float(
                        np.mean([row["posterior_sd_ratio"] for row in group])
                    ),
                    "ci_width_ratio": float(
                        np.mean([row["ci_width_ratio"] for row in group])
                    ),
                }
            )
    return rows


def summarize_diagnostics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    columns = (
        "sample_ate_full_var_total_analytic",
        "sample_ate_full_var_diag_analytic",
        "sample_ate_full_var_cross_analytic",
        "sample_ate_full_cross_to_diag_ratio_analytic",
        "sample_ate_meanfield_var_analytic",
        "sample_ate_meanfield_to_full_var_ratio_analytic",
        "sample_ate_sd_ratio_analytic",
    )
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            int(row["n_rff"]),
            float(row["signal_sd"]),
            float(row["noise_sd"]),
            str(row["lengthscale_mode"]),
            float(row["initial_lengthscale"]),
            float(row["initial_treatment_lengthscale"]),
        )
        groups.setdefault(key, []).append(row)

    summary_rows = []
    for key, group in sorted(groups.items()):
        (
            n_rff,
            signal_sd,
            noise_sd,
            lengthscale_mode,
            initial_lengthscale,
            initial_treatment_lengthscale,
        ) = key
        out: dict[str, object] = {
            "n_rff": n_rff,
            "signal_sd": signal_sd,
            "noise_sd": noise_sd,
            "lengthscale_mode": lengthscale_mode,
            "initial_lengthscale": initial_lengthscale,
            "initial_treatment_lengthscale": initial_treatment_lengthscale,
            "selected_lengthscale_mean": float(
                np.mean([float(row["lengthscale"]) for row in group])
            ),
            "selected_treatment_lengthscale_mean": float(
                np.mean(
                    [float(row["treatment_lengthscale"]) for row in group]
                )
            ),
            "n_reps": len(group),
        }
        for column in columns:
            values = np.array([float(row[column]) for row in group], dtype=float)
            out[f"{column}_mean"] = float(values.mean())
            out[f"{column}_median"] = float(np.median(values))
            out[f"{column}_min"] = float(values.min())
            out[f"{column}_max"] = float(values.max())
        summary_rows.append(out)
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    data_path = Path(args.data_path).expanduser().resolve()
    data = np.load(data_path)
    stop_realization = args.start_realization + args.reps

    run_name = args.run_name or datetime.now().strftime("ihdp_raw_rff_%Y%m%d_%H%M%S")
    output_root = Path(args.output_root).expanduser().resolve()
    output_dir = output_root / run_name
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} already exists. Use --overwrite or a new --run-name."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    config_record = {
        "run_name": run_name,
        "data_path": str(data_path),
        "start_realization": args.start_realization,
        "reps": args.reps,
        "draws": args.draws,
        "n_rff_grid": args.n_rff_grid,
        "noise_sd_grid": args.noise_sd_grid,
        "signal_sd_grid": args.signal_sd_grid,
        "lengthscale_grid": args.lengthscale_grid,
        "treatment_lengthscale_grid": args.treatment_lengthscale_grid,
        "lengthscale_mode": args.lengthscale_mode,
        "mml_lengthscale_min": args.mml_lengthscale_min,
        "mml_lengthscale_max": args.mml_lengthscale_max,
        "mml_restarts": args.mml_restarts,
        "mml_maxiter": args.mml_maxiter,
        "map_prior_lx_median": args.map_prior_lx_median,
        "map_prior_la_median": args.map_prior_la_median,
        "map_prior_log_sd_x": args.map_prior_log_sd_x,
        "map_prior_log_sd_a": args.map_prior_log_sd_a,
        "beta_prior_sd": args.beta_prior_sd,
        "jitter": args.jitter,
        "level": args.level,
        "seed": args.seed,
        "center_y": not args.no_center_y,
        "standardize_x": args.standardize_x,
        "standardize_y": args.standardize_y,
        "pi_model": args.pi_model,
        "pi_clip": args.pi_clip,
        "pi_logistic_c": args.pi_logistic_c,
        "source_hashes": {
            path.name: file_sha256(path)
            for path in (
                Path(__file__),
                CODE_DIR / "src" / "gp_exact.py",
                CODE_DIR / "src" / "metrics.py",
                CODE_DIR / "src" / "vi_rff.py",
            )
        },
    }
    (output_dir / "config.json").write_text(
        json.dumps(config_record, indent=2, sort_keys=True) + "\n"
    )

    diagnostic_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    for realization in range(args.start_realization, stop_realization):
        ihdp = load_ihdp_realization(
            data,
            realization,
            args.standardize_x,
            args.standardize_y,
        )
        x = np.asarray(ihdp["x"], dtype=float)
        a = np.asarray(ihdp["a"], dtype=float)
        y_model = np.asarray(ihdp["y_model"], dtype=float)
        y_scale = float(ihdp["y_sd"])
        if args.pi_model == "logistic":
            pi_hat = fit_logistic_propensity(
                x,
                a,
                clip=args.pi_clip,
                c=args.pi_logistic_c,
            )
        else:
            pi_hat = np.full_like(a, np.nan, dtype=float)
        pi_diagnostics = (
            propensity_diagnostics(pi_hat, args.pi_clip)
            if args.pi_model == "logistic"
            else {
                "pi_min": np.nan,
                "pi_p05": np.nan,
                "pi_mean": np.nan,
                "pi_p95": np.nan,
                "pi_max": np.nan,
                "pi_clip_fraction": np.nan,
            }
        )
        train_z = make_gp_inputs(x, a)
        control_z, treated_z = make_counterfactual_inputs(x)
        true_ate = float(ihdp["sample_ate_true"])
        mml_cache = {}

        config_grid = list(
            product(
                args.n_rff_grid,
                args.signal_sd_grid,
                args.lengthscale_grid,
                args.treatment_lengthscale_grid,
                args.noise_sd_grid,
            )
        )
        for config_index, (
            n_rff,
            signal_sd,
            lengthscale,
            treatment_lengthscale,
            noise_sd,
        ) in enumerate(config_grid):
            initial_gp_config = ExactGPConfig(
                signal_sd=signal_sd,
                lengthscale=lengthscale,
                treatment_lengthscale=treatment_lengthscale,
                noise_sd=noise_sd,
                jitter=args.jitter,
                center_y=not args.no_center_y,
            )
            mml_result = None
            mml_fit_seconds = 0.0
            if args.lengthscale_mode in {"mml", "map"}:
                cache_key = (
                    signal_sd,
                    noise_sd,
                    lengthscale,
                    treatment_lengthscale,
                )
                if cache_key not in mml_cache:
                    mml_start = perf_counter()
                    mml_result = fit_gp_lengthscales_empirical_bayes(
                        train_z,
                        y_model,
                        initial_gp_config,
                        selection_method=args.lengthscale_mode,
                        prior_medians=(
                            args.map_prior_lx_median
                            if args.map_prior_lx_median is not None
                            else lengthscale,
                            args.map_prior_la_median
                            if args.map_prior_la_median is not None
                            else treatment_lengthscale,
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
                        seed=realization,
                    )
                    mml_cache[cache_key] = (
                        mml_result,
                        perf_counter() - mml_start,
                    )
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
                        f"ihdp {realization}: "
                        f"l_x={mml_result.lengthscale:.6g} "
                        f"l_a={mml_result.treatment_lengthscale:.6g} "
                        f"{objective_label}={objective_value:.3f}",
                        flush=True,
                    )
                mml_result, mml_fit_seconds = mml_cache[cache_key]
                selected_gp_config = mml_result.config
            else:
                selected_gp_config = initial_gp_config
            feature_rng, full_rng, meanfield_rng, correction_rng = make_rngs(
                args.seed,
                realization,
                n_rff,
                config_index,
            )
            correction_weights = (
                bootstrap_weights(args.draws, x.shape[0], correction_rng)
                if args.pi_model == "logistic"
                else None
            )
            config = RFFConfig(
                n_rff=n_rff,
                signal_sd=signal_sd,
                lengthscale=selected_gp_config.lengthscale,
                treatment_lengthscale=(
                    selected_gp_config.treatment_lengthscale
                ),
                beta_prior_sd=args.beta_prior_sd,
                noise_sd=noise_sd,
                jitter=args.jitter,
                center_y=not args.no_center_y,
            )
            fit_start = perf_counter()
            posterior = fit_rff_bayesian_regression(
                train_z,
                y_model,
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
            ) * y_scale**2
            meanfield_ate_var = linear_functional_variance(
                posterior,
                sample_ate_coefficient,
                meanfield=True,
            ) * y_scale**2
            decomposition = linear_functional_variance_decomposition(
                posterior,
                sample_ate_coefficient,
            )
            decomposition = {
                key: value * y_scale**2 for key, value in decomposition.items()
            }
            coefficient_ratios = coefficient_variance_ratios(posterior)
            diagnostic_rows.append(
                {
                    "realization": realization,
                    "n": x.shape[0],
                    "n_features": x.shape[1],
                    "n_rff": n_rff,
                    "signal_sd": signal_sd,
                    "lengthscale_mode": args.lengthscale_mode,
                    "initial_lengthscale": lengthscale,
                    "initial_treatment_lengthscale": treatment_lengthscale,
                    "lengthscale": selected_gp_config.lengthscale,
                    "treatment_lengthscale": (
                        selected_gp_config.treatment_lengthscale
                    ),
                    "noise_sd": noise_sd,
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
                    "mml_success": (
                        int(mml_result.success)
                        if mml_result is not None
                        else np.nan
                    ),
                    "mml_n_iterations": (
                        mml_result.n_iterations
                        if mml_result is not None
                        else np.nan
                    ),
                    "mml_n_function_evaluations": (
                        mml_result.n_function_evaluations
                        if mml_result is not None
                        else np.nan
                    ),
                    "mml_at_boundary": (
                        int(mml_result.at_boundary)
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
                    "fit_seconds": fit_seconds,
                    "treated_share": float(a.mean()),
                    "pi_model": args.pi_model,
                    "pi_clip": args.pi_clip,
                    **pi_diagnostics,
                    "x_standardized": float(args.standardize_x),
                    "y_standardized": float(args.standardize_y),
                    "y_mean": ihdp["y_mean"],
                    "y_sd": y_scale,
                    "sample_ate_true": true_ate,
                    "ihdp_ate_field": ihdp["ihdp_ate_field"],
                    "meanfield_kl_to_full": meanfield_kl_to_full(posterior),
                    "coefficient_var_ratio_min": float(coefficient_ratios.min()),
                    "coefficient_var_ratio_median": float(
                        np.median(coefficient_ratios)
                    ),
                    "coefficient_var_ratio_max": float(coefficient_ratios.max()),
                    "sample_ate_full_sd_analytic": float(np.sqrt(full_ate_var)),
                    "sample_ate_full_var_total_analytic": decomposition[
                        "full_total"
                    ],
                    "sample_ate_full_var_diag_analytic": decomposition[
                        "full_diagonal"
                    ],
                    "sample_ate_full_var_cross_analytic": decomposition[
                        "full_cross"
                    ],
                    "sample_ate_full_cross_to_diag_ratio_analytic": (
                        decomposition["full_cross"]
                        / decomposition["full_diagonal"]
                    ),
                    "sample_ate_meanfield_sd_analytic": float(
                        np.sqrt(meanfield_ate_var)
                    ),
                    "sample_ate_meanfield_var_analytic": decomposition[
                        "meanfield_total"
                    ],
                    "sample_ate_meanfield_to_full_var_ratio_analytic": (
                        decomposition["meanfield_total"]
                        / decomposition["full_total"]
                    ),
                    "sample_ate_sd_ratio_analytic": float(
                        np.sqrt(meanfield_ate_var / full_ate_var)
                    ),
                }
            )

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
            for method, draws in beta_draws.items():
                mu0_draws, mu1_draws = counterfactual_draws_from_beta(
                    draws,
                    control_z,
                    treated_z,
                    posterior,
                )
                sample_ate_draws = (mu1_draws - mu0_draws).mean(axis=1)
                sample_ate_draws = sample_ate_draws * y_scale
                metric_rows.append(
                    {
                        "realization": realization,
                        "n_rff": n_rff,
                        "signal_sd": signal_sd,
                        "lengthscale_mode": args.lengthscale_mode,
                        "initial_lengthscale": lengthscale,
                        "initial_treatment_lengthscale": (
                            treatment_lengthscale
                        ),
                        "lengthscale": selected_gp_config.lengthscale,
                        "treatment_lengthscale": (
                            selected_gp_config.treatment_lengthscale
                        ),
                        "noise_sd": noise_sd,
                        "method": method,
                        "estimand": "sample_ate",
                        "true_value": true_ate,
                        **posterior_draw_metrics(
                            sample_ate_draws,
                            true_ate,
                            level=args.level,
                        ),
                    }
                )
                if args.pi_model == "logistic":
                    eif_draws = outcome_eif_draws(
                        mu0_draws,
                        mu1_draws,
                        a,
                        y_model,
                        pi_hat,
                        pi_clip=args.pi_clip,
                    )
                    corrected_draws = (
                        (mu1_draws - mu0_draws).mean(axis=1)
                        + np.sum(correction_weights * eif_draws, axis=1)
                    )
                    corrected_draws = corrected_draws * y_scale
                    metric_rows.append(
                        {
                            "realization": realization,
                            "n_rff": n_rff,
                            "signal_sd": signal_sd,
                            "lengthscale_mode": args.lengthscale_mode,
                            "initial_lengthscale": lengthscale,
                            "initial_treatment_lengthscale": (
                                treatment_lengthscale
                            ),
                            "lengthscale": selected_gp_config.lengthscale,
                            "treatment_lengthscale": (
                                selected_gp_config.treatment_lengthscale
                            ),
                            "noise_sd": noise_sd,
                            "method": f"{method}{CORRECTED_SUFFIX}",
                            "estimand": "sample_ate",
                            "true_value": true_ate,
                            **posterior_draw_metrics(
                                corrected_draws,
                                true_ate,
                                level=args.level,
                            ),
                        }
                    )

            print(
                f"finished realization {realization + 1}/{stop_realization} "
                f"n_rff={n_rff} signal_sd={signal_sd:g} "
                f"lengthscale={selected_gp_config.lengthscale:g} "
                f"treatment_lengthscale={selected_gp_config.treatment_lengthscale:g} "
                f"noise_sd={noise_sd:g}",
                flush=True,
            )

    aggregate_rows = aggregate_metric_rows(metric_rows, args.level)
    comparison_rows = paired_comparisons(metric_rows)
    diagnostic_summary_rows = summarize_diagnostics(diagnostic_rows)

    write_csv(output_dir / "diagnostics.csv", diagnostic_rows, list(DIAGNOSTIC_COLUMNS))
    write_csv(output_dir / "posterior_metrics.csv", metric_rows, list(METRIC_COLUMNS))
    write_csv(
        output_dir / "posterior_metrics_aggregated.csv",
        aggregate_rows,
        list(aggregate_rows[0]),
    )
    write_csv(
        output_dir / "paired_comparisons.csv",
        comparison_rows,
        list(comparison_rows[0]),
    )
    write_csv(
        output_dir / "covariance_decomposition_summary.csv",
        diagnostic_summary_rows,
        list(diagnostic_summary_rows[0]),
    )

    print(f"Wrote IHDP raw RFF outputs to {output_dir}")
    print()
    for row in diagnostic_summary_rows:
        print(
            f"n_rff={row['n_rff']} signal_sd={row['signal_sd']} "
            f"lengthscale={row['selected_lengthscale_mean']} "
            f"treatment_lengthscale={row['selected_treatment_lengthscale_mean']} "
            f"noise_sd={row['noise_sd']}: cross/diag mean="
            f"{row['sample_ate_full_cross_to_diag_ratio_analytic_mean']:.3f}, "
            f"MF/full SD mean="
            f"{row['sample_ate_sd_ratio_analytic_mean']:.3f}"
        )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        default=str(PROJECT_ROOT / "data" / "ihdp" / "ihdp_npci_1-1000.train.npz"),
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "outputs" / "module4_ihdp_raw_rff"),
    )
    parser.add_argument("--run-name")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--start-realization", type=int, default=0)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--n-rff-grid", type=parse_int_grid, default=[200])
    parser.add_argument("--noise-sd-grid", type=parse_float_grid, default=[1.0])
    parser.add_argument("--signal-sd-grid", type=parse_float_grid, default=[1.0])
    parser.add_argument("--lengthscale-grid", type=parse_float_grid, default=[0.8])
    parser.add_argument(
        "--treatment-lengthscale-grid",
        type=parse_float_grid,
        default=[0.5],
    )
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
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--no-center-y", action="store_true")
    parser.add_argument("--standardize-x", action="store_true")
    parser.add_argument("--standardize-y", action="store_true")
    parser.add_argument("--pi-model", choices=["none", "logistic"], default="none")
    parser.add_argument("--pi-clip", type=float, default=0.02)
    parser.add_argument("--pi-logistic-c", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
