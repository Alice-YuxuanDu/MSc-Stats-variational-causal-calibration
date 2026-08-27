"""Run paired exact-GP and SVGP experiments on IHDP semi-synthetic data.

This runner mirrors the synthetic Module 5 exact/SVGP comparison, but replaces
the synthetic DGP with IHDP realizations and uses a fixed estimated logistic
propensity score for one-step correction.
"""

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

from run_module4_experiment_grid import (  # noqa: E402
    bootstrap_weights,
    correction_suffix,
    parse_int_grid,
    wilson_interval,
)
from src.gp_exact import (  # noqa: E402
    ExactGPConfig,
    fit_exact_gp_predictive,
    fit_gp_lengthscales_empirical_bayes,
    make_counterfactual_inputs,
    make_gp_inputs,
    sample_counterfactual_functions,
)
from src.metrics import posterior_draw_metrics  # noqa: E402
from src.posterior_correction import outcome_eif_draws  # noqa: E402
from src.svgp import (  # noqa: E402
    SVGPConfig,
    exact_gp_log_marginal_likelihood,
    fit_svgp,
    fit_svgp_elbo_restart,
    linear_functional_variance,
    predict_svgp,
    sample_svgp_counterfactuals,
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_existing_key(data: np.lib.npyio.NpzFile, names: list[str]) -> str:
    for name in names:
        if name in data.files:
            return name
    raise KeyError(f"none of these keys were found in the IHDP file: {names}")


def select_x_realization(x_all: np.ndarray, realization: int) -> np.ndarray:
    """Select one IHDP realization from common NPZ layouts.

    Supported layouts include (n, p, R), (R, n, p), (n, R, p), and already
    selected (n, p).
    """

    x_all = np.asarray(x_all, dtype=float)
    if x_all.ndim == 2:
        return x_all
    if x_all.ndim != 3:
        raise ValueError(f"expected x to have 2 or 3 dimensions, got {x_all.shape}")

    # Common IHDP NPCI layout: (n, p, R), e.g. (672, 25, 1000).
    if x_all.shape[1] <= 100 and x_all.shape[2] > realization:
        return x_all[:, :, realization]
    # Alternative layout: (R, n, p), e.g. (1000, 672, 25).
    if x_all.shape[2] <= 100 and x_all.shape[0] > x_all.shape[1]:
        if x_all.shape[0] <= realization:
            raise IndexError(f"realization {realization} out of range for {x_all.shape}")
        return x_all[realization, :, :]
    # Alternative layout: (n, R, p), e.g. (672, 1000, 25).
    if x_all.shape[2] <= 100 and x_all.shape[1] > realization:
        return x_all[:, realization, :]

    raise ValueError(f"could not infer realization axis for x shape {x_all.shape}")


def select_vector_realization(
    values_all: np.ndarray,
    realization: int,
    n_units: int,
    name: str,
) -> np.ndarray:
    values_all = np.asarray(values_all, dtype=float)
    if values_all.ndim == 1:
        if values_all.shape[0] != n_units:
            raise ValueError(
                f"{name} has length {values_all.shape[0]}, expected {n_units}"
            )
        return values_all
    if values_all.ndim != 2:
        raise ValueError(f"expected {name} to have 1 or 2 dimensions, got {values_all.shape}")

    if values_all.shape[0] == n_units and values_all.shape[1] > realization:
        return values_all[:, realization]
    if values_all.shape[1] == n_units and values_all.shape[0] > realization:
        return values_all[realization, :]

    raise ValueError(
        f"could not infer realization axis for {name} shape {values_all.shape}"
    )


def load_ihdp_realization(
    data_path: Path,
    realization: int,
    standardize_x: bool,
    standardize_y: bool,
) -> dict[str, np.ndarray | float | int]:
    with np.load(data_path) as npz:
        x_key = first_existing_key(npz, ["x", "X", "covariates"])
        a_key = first_existing_key(npz, ["t", "a", "A", "z", "treatment"])
        y_key = first_existing_key(npz, ["yf", "y", "Y", "outcome"])

        x_raw = select_x_realization(npz[x_key], realization)
        n_units = x_raw.shape[0]
        a = select_vector_realization(npz[a_key], realization, n_units, a_key)
        y_raw = select_vector_realization(npz[y_key], realization, n_units, y_key)

        mu0 = mu1 = None
        if "mu0" in npz.files and "mu1" in npz.files:
            mu0 = select_vector_realization(npz["mu0"], realization, n_units, "mu0")
            mu1 = select_vector_realization(npz["mu1"], realization, n_units, "mu1")
        elif "ycf" in npz.files:
            ycf = select_vector_realization(npz["ycf"], realization, n_units, "ycf")
            mu0 = np.where(a > 0.5, ycf, y_raw)
            mu1 = np.where(a > 0.5, y_raw, ycf)
        else:
            raise KeyError(
                "IHDP file must contain either mu0/mu1 or ycf to compute true ATE."
            )

    a = (a > 0.5).astype(float)
    x = np.asarray(x_raw, dtype=float)
    y = np.asarray(y_raw, dtype=float)
    mu0 = np.asarray(mu0, dtype=float)
    mu1 = np.asarray(mu1, dtype=float)

    if standardize_x:
        x_mean = x.mean(axis=0)
        # Match the RFF/MFVI IHDP runner so one MML fit is reproducible across
        # all four posterior methods.
        x_sd = x.std(axis=0, ddof=1)
        x_sd = np.where(x_sd > 0.0, x_sd, 1.0)
        x = (x - x_mean) / x_sd

    y_mean = float(y.mean())
    y_sd = float(y.std(ddof=1))
    sample_ate_true = float(np.mean(mu1 - mu0))
    if standardize_y:
        y = (y - y_mean) / y_sd

    return {
        "x": x,
        "a": a,
        "y": y,
        "sample_ate_true": sample_ate_true,
        "n": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "treated_share": float(a.mean()),
        "y_mean": y_mean,
        "y_sd": y_sd,
        "outcome_scale": y_sd if standardize_y else 1.0,
    }


def fit_logistic_propensity(
    x: np.ndarray,
    a: np.ndarray,
    c: float,
    max_iter: int,
    clip: float,
) -> np.ndarray:
    try:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            C=c,
            solver="lbfgs",
            max_iter=max_iter,
        )
        model.fit(x, a.astype(int))
        pi = model.predict_proba(x)[:, 1]
    except Exception:
        # Fallback Newton solver with ridge approximately corresponding to 1 / C.
        ridge = 1.0 / c if c > 0.0 else 1.0
        design = np.column_stack([np.ones(x.shape[0]), x])
        beta = np.zeros(design.shape[1])
        penalty = ridge * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        for _ in range(max_iter):
            z = design @ beta
            pi = np.empty_like(z)
            positive = z >= 0
            pi[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
            exp_z = np.exp(z[~positive])
            pi[~positive] = exp_z / (1.0 + exp_z)
            weights = np.maximum(pi * (1.0 - pi), 1e-8)
            gradient = design.T @ (pi - a) + penalty @ beta
            hessian = (design.T * weights) @ design + penalty
            step = np.linalg.solve(hessian, gradient)
            beta -= step
            if np.max(np.abs(step)) < 1e-8:
                break
        z = design @ beta
        pi = 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))
    return np.clip(pi, clip, 1.0 - clip)


def make_rng(
    seed: int,
    realization: int,
    n_inducing: int,
    label: int,
) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([seed, realization, n_inducing, label])
    )


def sample_ate_draw_sets(
    effect_draws: np.ndarray,
    eif_draws: np.ndarray,
    corrected_weights: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "raw": effect_draws.mean(axis=1),
        "corrected": effect_draws.mean(axis=1)
        + np.sum(corrected_weights * eif_draws, axis=1),
    }


def add_metric_rows(
    rows: list[dict[str, float | int | str]],
    realization: int,
    n_inducing: int,
    method: str,
    draw_sets: dict[str, np.ndarray],
    true_value: float,
    level: float,
    corrected_suffix: str,
) -> None:
    entries = (
        (method, draw_sets["raw"]),
        (f"{method}_onestep_{corrected_suffix}", draw_sets["corrected"]),
    )
    for row_method, draws in entries:
        rows.append(
            {
                "realization": realization,
                "n_inducing": n_inducing,
                "estimand": "sample_ate",
                "method": row_method,
                "true_value": true_value,
                **posterior_draw_metrics(draws, true_value, level),
            }
        )


def aggregate_metrics(metric_df: pd.DataFrame, level: float) -> pd.DataFrame:
    keys = ["n_inducing", "estimand", "method"]
    rows = []
    for key, group in metric_df.groupby(keys, sort=True):
        n_reps = int(group.shape[0])
        coverage = float(group["covered"].mean())
        posterior_sd = float(group["posterior_sd"].mean())
        empirical_sd = float(group["posterior_mean"].std(ddof=1))
        lower, upper = wilson_interval(int(group["covered"].sum()), n_reps)
        rows.append(
            {
                **dict(zip(keys, key)),
                "n_reps": n_reps,
                "bias": float(group["bias"].mean()),
                "bias_mcse": float(group["bias"].std(ddof=1) / np.sqrt(n_reps)),
                "mean_abs_error": float(group["abs_error"].mean()),
                "median_abs_error": float(group["abs_error"].median()),
                "rmse": float(np.sqrt(group["squared_error"].mean())),
                "coverage": coverage,
                "coverage_mcse": float(np.sqrt(coverage * (1.0 - coverage) / n_reps)),
                "coverage_wilson_lower": lower,
                "coverage_wilson_upper": upper,
                "nominal_coverage": level,
                "mean_ci_width": float(group["ci_width"].mean()),
                "empirical_sd_posterior_mean": empirical_sd,
                "mean_posterior_sd": posterior_sd,
                "empirical_to_posterior_sd_ratio": empirical_sd / posterior_sd,
            }
        )
    return pd.DataFrame(rows)


def paired_comparisons(metric_df: pd.DataFrame) -> pd.DataFrame:
    index = ["realization", "n_inducing", "estimand"]
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
    comparisons = [("svgp_minus_exact_raw", "svgp", "exact_gp")]
    for suffix in corrected_suffixes:
        comparisons.extend(
            [
                (
                    f"svgp_minus_exact_onestep_{suffix}",
                    f"svgp_onestep_{suffix}",
                    f"exact_gp_onestep_{suffix}",
                ),
                (
                    f"onestep_{suffix}_minus_raw_exact",
                    f"exact_gp_onestep_{suffix}",
                    "exact_gp",
                ),
                (
                    f"onestep_{suffix}_minus_raw_svgp",
                    f"svgp_onestep_{suffix}",
                    "svgp",
                ),
            ]
        )

    rows = []
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
                    wide[("posterior_sd", left)] / wide[("posterior_sd", right)]
                ),
                "ci_width_ratio": (
                    wide[("ci_width", left)] / wide[("ci_width", right)]
                ),
            }
        ).reset_index()
        for key, group in frame.groupby(["n_inducing", "estimand"], sort=True):
            rows.append(
                {
                    "n_inducing": int(key[0]),
                    "estimand": key[1],
                    "comparison": comparison,
                    "n_reps": int(group.shape[0]),
                    "posterior_mean_difference": float(
                        group["posterior_mean_difference"].mean()
                    ),
                    "abs_error_difference": float(
                        group["abs_error_difference"].mean()
                    ),
                    "posterior_sd_ratio": float(group["posterior_sd_ratio"].mean()),
                    "ci_width_ratio": float(group["ci_width_ratio"].mean()),
                }
            )
    return pd.DataFrame(rows)


def save_outputs(
    output_dir: Path,
    diagnostic_rows: list[dict[str, float | int | str]],
    metric_rows: list[dict[str, float | int | str]],
    level: float,
) -> None:
    diagnostic_df = pd.DataFrame(diagnostic_rows).drop_duplicates(
        ["realization", "n_inducing"],
        keep="last",
    )
    metric_df = pd.DataFrame(metric_rows).drop_duplicates(
        ["realization", "n_inducing", "method", "estimand"],
        keep="last",
    )
    diagnostic_df.sort_values(["n_inducing", "realization"]).to_csv(
        output_dir / "diagnostics.csv",
        index=False,
    )
    metric_df.sort_values(["n_inducing", "estimand", "method", "realization"]).to_csv(
        output_dir / "posterior_metrics.csv",
        index=False,
    )
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
    output_dir = Path(args.output_root).expanduser().resolve() / run_name
    config_path = output_dir / "config.json"
    data_path = Path(args.data_path).expanduser().resolve()

    source_paths = (
        Path(__file__),
        CODE_DIR / "src" / "gp_exact.py",
        CODE_DIR / "src" / "svgp.py",
        CODE_DIR / "src" / "posterior_correction.py",
    )
    config_record = {
        "run_name": run_name,
        "data_path": str(data_path),
        "start_realization": args.start_realization,
        "reps": args.reps,
        "draws": args.draws,
        "n_inducing_grid": args.n_inducing_grid,
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
        "noise_sd": args.noise_sd,
        "jitter": args.jitter,
        "level": args.level,
        "seed": args.seed,
        "standardize_x": args.standardize_x,
        "standardize_y": args.standardize_y,
        "pi_model": args.pi_model,
        "pi_clip": args.pi_clip,
        "pi_logistic_c": args.pi_logistic_c,
        "pi_logistic_max_iter": args.pi_logistic_max_iter,
        "inducing_location_mode": args.inducing_location_mode,
        "inducing_location_restarts": args.inducing_location_restarts,
        "runtime_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "python": sys.version.split()[0],
        },
        "source_hashes": {
            path.name: file_sha256(path) for path in source_paths if path.exists()
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
    config_path.write_text(json.dumps(config_record, indent=2, sort_keys=True) + "\n")

    diagnostic_rows = load_rows(output_dir / "diagnostics.csv")
    metric_rows = load_rows(output_dir / "posterior_metrics.csv")
    completed = {
        (int(row["realization"]), int(row["n_inducing"]))
        for row in diagnostic_rows
    }

    completed_since_save = 0
    corrected_suffix = correction_suffix(args.pi_model)
    for rep in range(args.reps):
        realization = args.start_realization + rep
        data = load_ihdp_realization(
            data_path,
            realization,
            standardize_x=args.standardize_x,
            standardize_y=args.standardize_y,
        )
        x = data["x"]
        a = data["a"]
        y = data["y"]
        outcome_scale = float(data["outcome_scale"])
        n_units = int(data["n"])
        true_value = float(data["sample_ate_true"])

        pi = fit_logistic_propensity(
            x,
            a,
            c=args.pi_logistic_c,
            max_iter=args.pi_logistic_max_iter,
            clip=args.pi_clip,
        )

        train_z = make_gp_inputs(x, a)
        control_z, treated_z = make_counterfactual_inputs(x)
        test_z = np.vstack([control_z, treated_z])
        coefficient = np.concatenate(
            [-np.ones(n_units) / n_units, np.ones(n_units) / n_units]
        )

        exact_config = ExactGPConfig(
            signal_sd=args.signal_sd,
            lengthscale=args.lengthscale,
            treatment_lengthscale=args.treatment_lengthscale,
            noise_sd=args.noise_sd,
            jitter=args.jitter,
            center_y=True,
        )
        mml_fit_seconds = 0.0
        mml_result = None
        if args.lengthscale_mode in {"mml", "map"}:
            mml_start = perf_counter()
            mml_result = fit_gp_lengthscales_empirical_bayes(
                train_z,
                y,
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
                seed=realization,
            )
            mml_fit_seconds = perf_counter() - mml_start
            exact_config = mml_result.config
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
                f"{args.lengthscale_mode.upper()} ihdp {realization}: "
                f"l_x={exact_config.lengthscale:.6g} "
                f"l_a={exact_config.treatment_lengthscale:.6g} "
                f"{objective_label}={objective_value:.3f}",
                flush=True,
            )
        exact_start = perf_counter()
        exact = fit_exact_gp_predictive(train_z, y, test_z, exact_config)
        exact_fit_seconds = perf_counter() - exact_start
        exact_mu0, exact_mu1 = sample_counterfactual_functions(
            exact,
            n_units=n_units,
            n_draws=args.draws,
            rng=make_rng(args.seed, realization, 0, 11),
            jitter=args.jitter,
        )
        exact_effect = exact_mu1 - exact_mu0
        exact_eif = outcome_eif_draws(
            exact_mu0,
            exact_mu1,
            a,
            y,
            pi,
            pi_clip=args.pi_clip,
        )
        exact_ate_var = (
            float(max(coefficient @ exact.cov @ coefficient, 0.0))
            * outcome_scale**2
        )

        for n_inducing in args.n_inducing_grid:
            if (realization, n_inducing) in completed:
                continue

            corrected_weights = bootstrap_weights(
                args.draws,
                n_units,
                make_rng(args.seed, realization, n_inducing, 21),
            )
            svgp_config = SVGPConfig(
                n_inducing=n_inducing,
                signal_sd=args.signal_sd,
                lengthscale=exact_config.lengthscale,
                treatment_lengthscale=exact_config.treatment_lengthscale,
                noise_sd=args.noise_sd,
                jitter=args.jitter,
                center_y=True,
            )
            svgp_start = perf_counter()
            if args.inducing_location_mode == "balanced":
                svgp = fit_svgp(train_z, y, svgp_config)
            else:
                svgp = fit_svgp_elbo_restart(
                    train_z,
                    y,
                    svgp_config,
                    n_restarts=args.inducing_location_restarts,
                    seed=args.seed + 10_000 * realization + n_inducing,
                )
            svgp_predictive = predict_svgp(svgp, test_z)
            svgp_fit_seconds = perf_counter() - svgp_start
            svgp_mu0, svgp_mu1 = sample_svgp_counterfactuals(
                svgp_predictive,
                n_units=n_units,
                n_draws=args.draws,
                rng=make_rng(args.seed, realization, n_inducing, 31),
                jitter=args.jitter,
            )
            svgp_effect = svgp_mu1 - svgp_mu0
            svgp_eif = outcome_eif_draws(
                svgp_mu0,
                svgp_mu1,
                a,
                y,
                pi,
                pi_clip=args.pi_clip,
            )
            svgp_ate_var = (
                linear_functional_variance(svgp_predictive, coefficient)
                * outcome_scale**2
            )
            exact_lml = exact_gp_log_marginal_likelihood(train_z, y, svgp_config)

            exact_draw_sets = sample_ate_draw_sets(
                exact_effect,
                exact_eif,
                corrected_weights,
            )
            exact_draw_sets = {
                key: value * outcome_scale
                for key, value in exact_draw_sets.items()
            }
            svgp_draw_sets = sample_ate_draw_sets(
                svgp_effect,
                svgp_eif,
                corrected_weights,
            )
            svgp_draw_sets = {
                key: value * outcome_scale
                for key, value in svgp_draw_sets.items()
            }
            add_metric_rows(
                metric_rows,
                realization,
                n_inducing,
                "exact_gp",
                exact_draw_sets,
                true_value,
                args.level,
                corrected_suffix,
            )
            add_metric_rows(
                metric_rows,
                realization,
                n_inducing,
                "svgp",
                svgp_draw_sets,
                true_value,
                args.level,
                corrected_suffix,
            )

            diagnostic_rows.append(
                {
                    "realization": realization,
                    "n": n_units,
                    "n_features": int(data["n_features"]),
                    "n_inducing": n_inducing,
                    "signal_sd": args.signal_sd,
                    "lengthscale_mode": args.lengthscale_mode,
                    "initial_lengthscale": args.lengthscale,
                    "initial_treatment_lengthscale": (
                        args.treatment_lengthscale
                    ),
                    "lengthscale": exact_config.lengthscale,
                    "treatment_lengthscale": (
                        exact_config.treatment_lengthscale
                    ),
                    "noise_sd": args.noise_sd,
                    "mml_fit_seconds": mml_fit_seconds,
                    "mml_initial_log_marginal_likelihood": (
                        mml_result.initial_log_marginal_likelihood
                        if mml_result is not None
                        else np.nan
                    ),
                    "mml_log_marginal_likelihood": (
                        mml_result.log_marginal_likelihood
                        if mml_result is not None
                        else exact_lml
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
                    "standardize_x": int(args.standardize_x),
                    "standardize_y": int(args.standardize_y),
                    "treated_share": float(data["treated_share"]),
                    "sample_ate_true": true_value,
                    "y_mean": float(data["y_mean"]),
                    "y_sd": float(data["y_sd"]),
                    "pi_model": args.pi_model,
                    "pi_clip": args.pi_clip,
                    "pi_min": float(pi.min()),
                    "pi_p05": float(np.quantile(pi, 0.05)),
                    "pi_mean": float(pi.mean()),
                    "pi_p95": float(np.quantile(pi, 0.95)),
                    "pi_max": float(pi.max()),
                    "pi_clip_fraction": float(
                        np.mean((pi <= args.pi_clip) | (pi >= 1.0 - args.pi_clip))
                    ),
                    "inducing_location_mode": args.inducing_location_mode,
                    "inducing_location_restarts": (
                        args.inducing_location_restarts
                    ),
                    "exact_fit_seconds": exact_fit_seconds,
                    "svgp_fit_seconds": svgp_fit_seconds,
                    "speed_ratio_exact_over_svgp": (
                        exact_fit_seconds / svgp_fit_seconds
                    ),
                    "elbo": svgp.elbo,
                    "exact_log_marginal_likelihood": exact_lml,
                    "elbo_gap": exact_lml - svgp.elbo,
                    "trace_gap": svgp.trace_gap,
                    "relative_trace_gap": svgp.relative_trace_gap,
                    "predictive_mean_rmse_vs_exact": float(
                        np.sqrt(np.mean((svgp_predictive.mean - exact.mean) ** 2))
                    ),
                    "predictive_cov_relative_frobenius": float(
                        np.linalg.norm(svgp_predictive.cov - exact.cov, ord="fro")
                        / np.linalg.norm(exact.cov, ord="fro")
                    ),
                    "sample_ate_mean_difference": float(
                        outcome_scale
                        * coefficient
                        @ (svgp_predictive.mean - exact.mean)
                    ),
                    "sample_ate_exact_sd_analytic": float(np.sqrt(exact_ate_var)),
                    "sample_ate_svgp_sd_analytic": float(np.sqrt(svgp_ate_var)),
                    "sample_ate_svgp_to_exact_sd_ratio": float(
                        np.sqrt(svgp_ate_var / exact_ate_var)
                    ),
                }
            )

            completed.add((realization, n_inducing))
            completed_since_save += 1
            print(
                f"finished realization {rep + 1}/{args.reps} "
                f"(ihdp {realization}) m={n_inducing} "
                f"exact={exact_fit_seconds:.2f}s svgp={svgp_fit_seconds:.2f}s",
                flush=True,
            )
            if completed_since_save >= args.checkpoint_every:
                save_outputs(output_dir, diagnostic_rows, metric_rows, args.level)
                completed_since_save = 0

    save_outputs(output_dir, diagnostic_rows, metric_rows, args.level)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name")
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "outputs" / "module5_ihdp_svgp"),
    )
    parser.add_argument(
        "--data-path",
        default=str(PROJECT_ROOT / "data" / "ihdp" / "ihdp_npci_1-1000.train.npz"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--start-realization", type=int, default=0)
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument(
        "--n-inducing-grid",
        type=parse_int_grid,
        default=[160, 240, 320],
    )
    parser.add_argument("--signal-sd", type=float, default=1.0)
    parser.add_argument("--lengthscale", type=float, default=10.0)
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
    parser.add_argument("--noise-sd", type=float, default=0.5)
    parser.add_argument("--jitter", type=float, default=1e-8)
    parser.add_argument("--level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--standardize-x", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--standardize-y", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pi-model", choices=["logistic"], default="logistic")
    parser.add_argument("--pi-clip", type=float, default=0.02)
    parser.add_argument("--pi-logistic-c", type=float, default=1.0)
    parser.add_argument("--pi-logistic-max-iter", type=int, default=1000)
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
    return parser.parse_args()


if __name__ == "__main__":
    output = run(parse_args())
    print(f"Wrote experiment outputs to {output}")
