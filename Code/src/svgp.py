"""Full-covariance sparse variational Gaussian process regression.

This module implements the Gaussian-likelihood Titsias variational posterior.
The optimal distribution over inducing values is available in closed form, so
the implementation needs only NumPy and retains correlations between inducing
variables. It is the full-batch SGPR special case of an SVGP model.
"""

from dataclasses import dataclass

import numpy as np

from .gp_exact import (
    ExactGPConfig,
    exact_gp_log_marginal_likelihood as _exact_gp_log_marginal_likelihood,
    rbf_kernel,
    solve_cholesky,
    stable_cholesky,
)


@dataclass
class SVGPConfig:
    n_inducing: int = 40
    signal_sd: float = 1.0
    lengthscale: float = 0.8
    treatment_lengthscale: float = 0.5
    noise_sd: float = 1.0
    jitter: float = 1e-6
    center_y: bool = True

    @property
    def kernel_config(self) -> ExactGPConfig:
        return ExactGPConfig(
            signal_sd=self.signal_sd,
            lengthscale=self.lengthscale,
            treatment_lengthscale=self.treatment_lengthscale,
            noise_sd=self.noise_sd,
            jitter=self.jitter,
            center_y=self.center_y,
        )


@dataclass
class SVGPPosterior:
    inducing_z: np.ndarray
    variational_mean: np.ndarray
    variational_cov: np.ndarray
    y_mean: float
    elbo: float
    trace_gap: float
    relative_trace_gap: float
    config: SVGPConfig


@dataclass
class SVGPPredictive:
    mean: np.ndarray
    cov: np.ndarray


def select_balanced_inducing_inputs(
    train_z: np.ndarray,
    n_inducing: int,
    lengthscale: float
):
    """Select deterministic covariate locations and represent both treatments.

    A maximin design is constructed in the covariate space, then copied to the
    control and treated input planes. This avoids accidentally omitting one
    potential-outcome surface when treatment assignment is imbalanced.
    """

    train_z = np.asarray(train_z, dtype=float)
    covariates = train_z[:, :-1]
    n_control = (n_inducing + 1) // 2
    n_treated = n_inducing // 2
    indices = maximin_indices(
        covariates / lengthscale,
        max(n_control, n_treated),
    )
    control = np.column_stack(
        [covariates[indices[:n_control]], np.zeros(n_control)]
    )
    treated = np.column_stack(
        [covariates[indices[:n_treated]], np.ones(n_treated)]
    )
    return np.vstack([control, treated])


def select_random_balanced_inducing_inputs(
    train_z: np.ndarray,
    n_inducing: int,
    rng: np.random.Generator
):
    """Select a random balanced design over covariates and treatment planes."""

    train_z = np.asarray(train_z, dtype=float)
    covariates = train_z[:, :-1]
    n_control = (n_inducing + 1) // 2
    n_treated = n_inducing // 2
    n_select = max(n_control, n_treated)
    indices = rng.choice(covariates.shape[0], size=n_select, replace=False)
    control = np.column_stack(
        [covariates[indices[:n_control]], np.zeros(n_control)]
    )
    treated = np.column_stack(
        [covariates[indices[:n_treated]], np.ones(n_treated)]
    )
    return np.vstack([control, treated])


def maximin_indices(x: np.ndarray, n_select: int):
    """Greedy deterministic maximin subset selection."""

    x = np.asarray(x, dtype=float)
    center = x.mean(axis=0)
    first = int(np.argmin(np.sum((x - center) ** 2, axis=1)))
    selected = np.empty(n_select, dtype=int)
    selected[0] = first
    min_distance = np.sum((x - x[first]) ** 2, axis=1)
    min_distance[first] = -np.inf

    for position in range(1, n_select):
        index = int(np.argmax(min_distance))
        selected[position] = index
        distance = np.sum((x - x[index]) ** 2, axis=1)
        min_distance = np.minimum(min_distance, distance)
        min_distance[selected[: position + 1]] = -np.inf
    return selected


def fit_svgp(
    train_z: np.ndarray,
    train_y: np.ndarray,
    config: SVGPConfig,
    inducing_z: np.ndarray | None = None
):
    """Fit the optimal Titsias variational distribution q(u)."""

    train_z = np.asarray(train_z, dtype=float)
    train_y = np.asarray(train_y, dtype=float)

    if inducing_z is None:
        inducing_z = select_balanced_inducing_inputs(
            train_z,
            config.n_inducing,
            config.lengthscale,
        )
    else:
        inducing_z = np.asarray(inducing_z, dtype=float)

    y_mean = float(train_y.mean()) if config.center_y else 0.0
    y_centered = train_y - y_mean
    noise_var = config.noise_sd**2
    kernel_config = config.kernel_config

    k_mm = rbf_kernel(inducing_z, inducing_z, kernel_config)
    k_mm = k_mm + config.jitter * np.eye(inducing_z.shape[0])
    chol_mm = stable_cholesky(k_mm, config.jitter)
    k_mn = rbf_kernel(inducing_z, train_z, kernel_config)
    gram = k_mn @ k_mn.T

    sigma = k_mm + gram / noise_var
    chol_sigma = stable_cholesky(sigma, config.jitter)
    sigma_inv_kmn_y = solve_cholesky(chol_sigma, k_mn @ y_centered)
    variational_mean = k_mm @ sigma_inv_kmn_y / noise_var
    variational_cov = k_mm @ solve_cholesky(chol_sigma, k_mm)
    variational_cov = 0.5 * (variational_cov + variational_cov.T)

    logdet_c = (
        train_z.shape[0] * np.log(noise_var)
        + logdet_from_cholesky(chol_sigma)
        - logdet_from_cholesky(chol_mm)
    )
    quadratic = (
        y_centered @ y_centered / noise_var
        - (k_mn @ y_centered) @ sigma_inv_kmn_y / noise_var**2
    )
    trace_q = float(np.trace(solve_cholesky(chol_mm, gram)))
    trace_k = train_z.shape[0] * config.signal_sd**2
    trace_gap = max(float(trace_k - trace_q), 0.0)
    log_normal = -0.5 * (
        train_z.shape[0] * np.log(2.0 * np.pi)
        + logdet_c
        + quadratic
    )
    elbo = float(log_normal - trace_gap / (2.0 * noise_var))

    return SVGPPosterior(
        inducing_z=inducing_z,
        variational_mean=variational_mean,
        variational_cov=variational_cov,
        y_mean=y_mean,
        elbo=elbo,
        trace_gap=trace_gap,
        relative_trace_gap=trace_gap / trace_k,
        config=config,
    )


def fit_svgp_elbo_restart(
    train_z: np.ndarray,
    train_y: np.ndarray,
    config: SVGPConfig,
    n_restarts: int = 8,
    seed: int = 0
):
    """Fit SVGP after selecting inducing locations by ELBO over restarts.

    The first candidate is the deterministic balanced maximin design used in
    the main experiments. Remaining candidates are random balanced designs over
    the observed covariates copied to both treatment planes. This is a
    reproducible finite-candidate ELBO optimisation, not continuous
    optimisation over all inducing coordinates.
    """

    train_z = np.asarray(train_z, dtype=float)
    train_y = np.asarray(train_y, dtype=float)
    rng = np.random.default_rng(seed)

    candidates = [
        select_balanced_inducing_inputs(
            train_z,
            config.n_inducing,
            config.lengthscale,
        )
    ]
    for _ in range(n_restarts - 1):
        candidates.append(
            select_random_balanced_inducing_inputs(
                train_z,
                config.n_inducing,
                rng,
            )
        )

    best: SVGPPosterior | None = None
    for inducing_z in candidates:
        posterior = fit_svgp(train_z, train_y, config, inducing_z=inducing_z)
        if best is None or posterior.elbo > best.elbo:
            best = posterior
    if best is None:  # pragma: no cover
        raise RuntimeError("ELBO restart selection produced no candidate.")
    return best


def predict_svgp(
    posterior: SVGPPosterior,
    test_z: np.ndarray
):
    """Return the joint latent-function distribution q(f(test_z))."""

    test_z = np.asarray(test_z, dtype=float)
    kernel_config = posterior.config.kernel_config
    k_mm = rbf_kernel(
        posterior.inducing_z,
        posterior.inducing_z,
        kernel_config,
    )
    k_mm = k_mm + posterior.config.jitter * np.eye(
        posterior.inducing_z.shape[0]
    )
    chol_mm = stable_cholesky(k_mm, posterior.config.jitter)
    k_mt = rbf_kernel(posterior.inducing_z, test_z, kernel_config)
    k_tm = k_mt.T
    k_tt = rbf_kernel(test_z, test_z, kernel_config)

    kmm_inv_kmt = solve_cholesky(chol_mm, k_mt)
    mean = (
        posterior.y_mean
        + k_tm @ solve_cholesky(chol_mm, posterior.variational_mean)
    )
    cov = (
        k_tt
        - k_tm @ kmm_inv_kmt
        + kmm_inv_kmt.T
        @ posterior.variational_cov
        @ kmm_inv_kmt
    )
    cov = 0.5 * (cov + cov.T)
    return SVGPPredictive(mean=mean, cov=cov)


def sample_svgp_functions(
    predictive: SVGPPredictive,
    n_draws: int,
    rng: np.random.Generator,
    jitter: float = 1e-6
):
    chol = stable_cholesky(predictive.cov, jitter)
    standard = rng.normal(size=(predictive.mean.size, n_draws))
    return (predictive.mean[:, None] + chol @ standard).T


def sample_svgp_counterfactuals(
    predictive: SVGPPredictive,
    n_units: int,
    n_draws: int,
    rng: np.random.Generator,
    jitter: float = 1e-6
):
    draws = sample_svgp_functions(predictive, n_draws, rng, jitter)
    return draws[:, :n_units], draws[:, n_units:]


def linear_functional_variance(
    predictive: SVGPPredictive,
    coefficient: np.ndarray
):
    coefficient = np.asarray(coefficient, dtype=float)
    variance = coefficient @ predictive.cov @ coefficient
    return float(max(variance, 0.0))


def exact_gp_log_marginal_likelihood(
    train_z: np.ndarray,
    train_y: np.ndarray,
    config: SVGPConfig
):
    """Compute the exact GP log marginal likelihood for diagnostics."""

    return _exact_gp_log_marginal_likelihood(
        train_z,
        train_y,
        config.kernel_config,
    )


def logdet_from_cholesky(chol: np.ndarray):
    return float(2.0 * np.log(np.diag(chol)).sum())
