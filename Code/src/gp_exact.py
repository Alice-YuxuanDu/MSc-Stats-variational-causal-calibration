"""Exact Gaussian process regression utilities for Module 2.

The implementation is intentionally small and dependency-light. In addition
to exact prediction, it supports empirical-Bayes selection of the shared
covariate and treatment lengthscales by maximum marginal likelihood (MML).
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import cho_solve
from scipy.optimize import minimize


@dataclass
class ExactGPConfig:
    signal_sd: float = 1.0
    lengthscale: float = 0.8
    treatment_lengthscale: float = 0.5
    noise_sd: float = 1.0
    jitter: float = 1e-6
    center_y: bool = True

@dataclass
class ExactGPPosterior:
    mean: np.ndarray
    cov: np.ndarray
    train_y_mean: float


@dataclass
class GPLengthscaleMMLResult:
    """Result of exact-GP MML or regularized-MAP lengthscale fitting."""

    config: ExactGPConfig
    selection_method: str
    initial_log_marginal_likelihood: float
    log_marginal_likelihood: float
    initial_log_prior: float
    log_prior: float
    initial_log_posterior: float
    log_posterior: float
    success: bool
    n_iterations: int
    n_function_evaluations: int
    message: str
    at_boundary: bool

    @property
    def lengthscale(self):
        return self.config.lengthscale

    @property
    def treatment_lengthscale(self):
        return self.config.treatment_lengthscale


def make_gp_inputs(x: np.ndarray, a: np.ndarray):
    x = np.asarray(x, dtype=float)
    a = np.asarray(a, dtype=float).reshape(-1, 1)
    return np.column_stack([x, a])


def make_counterfactual_inputs(x: np.ndarray):
    n = x.shape[0]
    control = make_gp_inputs(x, np.zeros(n))
    treated = make_gp_inputs(x, np.ones(n))
    return control, treated


def rbf_kernel(z1: np.ndarray, z2: np.ndarray, config: ExactGPConfig):
    z1 = np.asarray(z1, dtype=float)
    z2 = np.asarray(z2, dtype=float)
    x1, a1 = z1[:, :-1], z1[:, -1:]
    x2, a2 = z2[:, :-1], z2[:, -1:]
    x_sqdist = pairwise_sqdist(x1 / config.lengthscale, x2 / config.lengthscale)
    a_sqdist = pairwise_sqdist(a1 / config.treatment_lengthscale, a2 / config.treatment_lengthscale)
    return config.signal_sd**2 * np.exp(-0.5 * (x_sqdist + a_sqdist))


def exact_gp_log_marginal_likelihood(
    train_z: np.ndarray,
    train_y: np.ndarray,
    config: ExactGPConfig
):
    """Return the exact Gaussian log marginal likelihood.

    ``lengthscale`` is shared by all covariate columns and
    ``treatment_lengthscale`` applies to the final treatment column.
    """

    train_z, train_y = validate_training_data(train_z, train_y)
    y_mean = float(train_y.mean()) if config.center_y else 0.0
    y_centered = train_y - y_mean
    covariance = rbf_kernel(train_z, train_z, config)
    covariance = covariance + (
        config.noise_sd**2 + config.jitter
    ) * np.eye(train_z.shape[0])
    chol = stable_cholesky(covariance, config.jitter)
    quadratic = y_centered @ solve_cholesky(chol, y_centered)
    logdet = 2.0 * np.log(np.diag(chol)).sum()
    return float(
        -0.5
        * (
            train_z.shape[0] * np.log(2.0 * np.pi)
            + logdet
            + quadratic
        )
    )


def fit_gp_lengthscales_mml(
    train_z: np.ndarray,
    train_y: np.ndarray,
    initial_config: ExactGPConfig,
    bounds: tuple[float, float] = (0.05, 50.0),
    n_restarts: int = 3,
    maxiter: int = 100,
    seed: int = 0
):
    """Select ``l_x`` and ``l_a`` by exact maximum marginal likelihood.

    Signal and noise scales are held at the values in ``initial_config``.
    Optimization is over log lengthscales, which enforces positivity. The
    first start uses the supplied lengthscales and later starts are drawn
    reproducibly within the log-bounds.
    """

    return _fit_gp_lengthscales(
        train_z=train_z,
        train_y=train_y,
        initial_config=initial_config,
        bounds=bounds,
        n_restarts=n_restarts,
        maxiter=maxiter,
        seed=seed,
        selection_method="mml",
        prior_medians=None,
        prior_log_sds=None,
    )


def fit_gp_lengthscales_map(
    train_z: np.ndarray,
    train_y: np.ndarray,
    initial_config: ExactGPConfig,
    prior_medians: tuple[float, float] | None = None,
    prior_log_sds: tuple[float, float] = (0.5, 0.5),
    bounds: tuple[float, float] = (0.05, 50.0),
    n_restarts: int = 3,
    maxiter: int = 100,
    seed: int = 0
):
    """Select ``l_x`` and ``l_a`` by regularized MAP.

    Independent Gaussian priors are placed on ``log(l_x)`` and ``log(l_a)``.
    ``prior_medians`` gives the corresponding medians on the original
    lengthscale scale. When it is omitted, the initial configuration values
    are used as the prior medians.
    """

    if prior_medians is None:
        prior_medians = (
            initial_config.lengthscale,
            initial_config.treatment_lengthscale,
        )
    return _fit_gp_lengthscales(
        train_z=train_z,
        train_y=train_y,
        initial_config=initial_config,
        bounds=bounds,
        n_restarts=n_restarts,
        maxiter=maxiter,
        seed=seed,
        selection_method="map",
        prior_medians=prior_medians,
        prior_log_sds=prior_log_sds,
    )


def fit_gp_lengthscales_empirical_bayes(
    train_z: np.ndarray,
    train_y: np.ndarray,
    initial_config: ExactGPConfig,
    selection_method: str = "mml",
    prior_medians: tuple[float, float] | None = None,
    prior_log_sds: tuple[float, float] = (0.5, 0.5),
    bounds: tuple[float, float] = (0.05, 50.0),
    n_restarts: int = 3,
    maxiter: int = 100,
    seed: int = 0
):
    """Dispatch to MML or regularized-MAP lengthscale selection."""

    if selection_method == "mml":
        return fit_gp_lengthscales_mml(
            train_z,
            train_y,
            initial_config,
            bounds=bounds,
            n_restarts=n_restarts,
            maxiter=maxiter,
            seed=seed,
        )
    if selection_method == "map":
        return fit_gp_lengthscales_map(
            train_z,
            train_y,
            initial_config,
            prior_medians=prior_medians,
            prior_log_sds=prior_log_sds,
            bounds=bounds,
            n_restarts=n_restarts,
            maxiter=maxiter,
            seed=seed,
        )
    raise ValueError("selection_method must be 'mml' or 'map'.")


def _fit_gp_lengthscales(
    train_z: np.ndarray,
    train_y: np.ndarray,
    initial_config: ExactGPConfig,
    bounds: tuple[float, float],
    n_restarts: int,
    maxiter: int,
    seed: int,
    selection_method: str,
    prior_medians: tuple[float, float] | None,
    prior_log_sds: tuple[float, float] | None
):
    """Shared optimizer for MML and regularized MAP selection."""

    train_z, train_y = validate_training_data(train_z, train_y)
    lower, upper = (float(value) for value in bounds)

    if selection_method == "map":
        prior_medians_array = np.asarray(prior_medians, dtype=float)
        prior_log_sds_array = np.asarray(prior_log_sds, dtype=float)
        prior_log_means = np.log(prior_medians_array)
    else:
        prior_log_means = np.zeros(2)
        prior_log_sds_array = np.ones(2)

    y_mean = float(train_y.mean()) if initial_config.center_y else 0.0
    y_centered = train_y - y_mean
    x = train_z[:, :-1]
    a = train_z[:, -1:]
    x_sqdist = pairwise_sqdist(x, x)
    a_sqdist = pairwise_sqdist(a, a)
    identity = np.eye(train_z.shape[0])
    noise_diagonal = initial_config.noise_sd**2 + initial_config.jitter
    log_bounds = (np.log(lower), np.log(upper))

    def log_prior_and_gradient(log_lengthscales: np.ndarray):
        if selection_method == "mml":
            return 0.0, np.zeros(2)
        standardized = (
            log_lengthscales - prior_log_means
        ) / prior_log_sds_array
        log_prior = -0.5 * np.sum(standardized**2)
        gradient = -(
            log_lengthscales - prior_log_means
        ) / prior_log_sds_array**2
        return float(log_prior), gradient

    def objective(log_lengthscales: np.ndarray):
        lengthscale, treatment_lengthscale = np.exp(log_lengthscales)
        scaled_x = x_sqdist / lengthscale**2
        scaled_a = a_sqdist / treatment_lengthscale**2
        signal_covariance = initial_config.signal_sd**2 * np.exp(
            -0.5 * (scaled_x + scaled_a)
        )
        covariance = signal_covariance + noise_diagonal * identity
        chol = stable_cholesky(covariance, initial_config.jitter)
        alpha = cho_solve(
            (chol, True),
            y_centered,
            check_finite=False,
        )
        logdet = 2.0 * np.log(np.diag(chol)).sum()
        log_marginal_likelihood = -0.5 * (
            train_z.shape[0] * np.log(2.0 * np.pi)
            + logdet
            + y_centered @ alpha
        )

        covariance_inverse = cho_solve(
            (chol, True),
            identity,
            check_finite=False,
        )
        common = np.outer(alpha, alpha) - covariance_inverse
        derivatives = (
            signal_covariance * scaled_x,
            signal_covariance * scaled_a,
        )
        gradient = np.array(
            [0.5 * np.sum(common * derivative) for derivative in derivatives]
        )
        log_prior, prior_gradient = log_prior_and_gradient(log_lengthscales)
        return (
            -float(log_marginal_likelihood + log_prior),
            -(gradient + prior_gradient),
        )

    initial = np.log(
        np.clip(
            [
                initial_config.lengthscale,
                initial_config.treatment_lengthscale,
            ],
            lower,
            upper,
        )
    )
    starts = [initial]
    rng = np.random.default_rng(seed)
    for _ in range(n_restarts - 1):
        starts.append(rng.uniform(log_bounds[0], log_bounds[1], size=2))

    best = None
    for start in starts:
        candidate = minimize(
            objective,
            start,
            method="L-BFGS-B",
            jac=True,
            bounds=[log_bounds, log_bounds],
            options={"maxiter": maxiter},
        )
        if best is None or candidate.fun < best.fun:
            best = candidate

    if best is None:
        raise RuntimeError("MML optimization produced no result.")
    selected = np.exp(best.x)
    selected_config = replace(
        initial_config,
        lengthscale=float(selected[0]),
        treatment_lengthscale=float(selected[1]),
    )
    boundary_tolerance = 1e-4
    at_boundary = bool(
        np.any(np.isclose(best.x, log_bounds[0], atol=boundary_tolerance))
        or np.any(np.isclose(best.x, log_bounds[1], atol=boundary_tolerance))
    )
    initial_log_marginal_likelihood = exact_gp_log_marginal_likelihood(
        train_z,
        train_y,
        initial_config,
    )
    selected_log_marginal_likelihood = exact_gp_log_marginal_likelihood(
        train_z,
        train_y,
        selected_config,
    )
    initial_log_prior, _ = log_prior_and_gradient(initial)
    selected_log_prior, _ = log_prior_and_gradient(best.x)
    return GPLengthscaleMMLResult(
        config=selected_config,
        selection_method=selection_method,
        initial_log_marginal_likelihood=initial_log_marginal_likelihood,
        log_marginal_likelihood=selected_log_marginal_likelihood,
        initial_log_prior=initial_log_prior,
        log_prior=selected_log_prior,
        initial_log_posterior=(
            initial_log_marginal_likelihood + initial_log_prior
        ),
        log_posterior=(
            selected_log_marginal_likelihood + selected_log_prior
        ),
        success=bool(best.success),
        n_iterations=int(best.nit),
        n_function_evaluations=int(best.nfev),
        message=str(best.message),
        at_boundary=at_boundary,
    )


def pairwise_sqdist(x1: np.ndarray, x2: np.ndarray):
    x1_norm = np.sum(x1**2, axis=1)[:, None]
    x2_norm = np.sum(x2**2, axis=1)[None, :]
    sqdist = x1_norm + x2_norm - 2.0 * x1 @ x2.T
    return np.maximum(sqdist, 0.0)


def fit_exact_gp_predictive(
    train_z: np.ndarray,
    train_y: np.ndarray,
    test_z: np.ndarray,
    config: ExactGPConfig
):
    train_z, train_y = validate_training_data(train_z, train_y)

    y_mean = float(train_y.mean()) if config.center_y else 0.0
    y_centered = train_y - y_mean

    k_train = rbf_kernel(train_z, train_z, config)
    k_train = k_train + (config.noise_sd**2 + config.jitter) * np.eye(train_z.shape[0])
    chol = stable_cholesky(k_train, config.jitter)

    k_test_train = rbf_kernel(test_z, train_z, config)
    alpha = solve_cholesky(chol, y_centered)
    mean = y_mean + k_test_train @ alpha

    v = np.linalg.solve(chol, k_test_train.T)
    k_test = rbf_kernel(test_z, test_z, config)
    cov = k_test - v.T @ v
    cov = 0.5 * (cov + cov.T)
    cov = cov + config.jitter * np.eye(test_z.shape[0])
    return ExactGPPosterior(mean=mean, cov=cov, train_y_mean=y_mean)


def validate_training_data(
    train_z: np.ndarray,
    train_y: np.ndarray,
):
    train_z = np.asarray(train_z, dtype=float)
    train_y = np.asarray(train_y, dtype=float)
    return train_z, train_y


def solve_cholesky(chol: np.ndarray, rhs: np.ndarray):
    return np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))


def stable_cholesky(matrix: np.ndarray, jitter: float, max_tries: int = 6):
    eye = np.eye(matrix.shape[0])
    current_jitter = jitter
    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(matrix + current_jitter * eye)
        except np.linalg.LinAlgError:
            current_jitter *= 10.0
    return np.linalg.cholesky(matrix + current_jitter * eye)


def sample_counterfactual_functions(
    posterior: ExactGPPosterior,
    n_units: int,
    n_draws: int,
    rng: np.random.Generator,
    jitter: float = 1e-6
):
    """Sample counterfactual outcome functions at observed covariates.

    The posterior mean/covariance must be ordered as all controls followed by
    all treated counterfactual inputs.
    """

    chol = stable_cholesky(posterior.cov, jitter)
    standard = rng.normal(size=(posterior.mean.size, n_draws))
    function_draws = posterior.mean[:, None] + chol @ standard
    control = function_draws[:n_units, :].T
    treated = function_draws[n_units:, :].T
    return control, treated


def sample_counterfactual_effects(
    posterior: ExactGPPosterior,
    n_units: int,
    n_draws: int,
    rng: np.random.Generator,
    jitter: float = 1e-6
):
    """Sample treatment-effect vectors at observed covariates."""

    control, treated = sample_counterfactual_functions(
        posterior,
        n_units=n_units,
        n_draws=n_draws,
        rng=rng,
        jitter=jitter,
    )
    return treated - control


def sample_cate_draws(effect_draws: np.ndarray):
    return np.asarray(effect_draws).mean(axis=1)


def population_ate_bb_draws(
    effect_draws: np.ndarray,
    rng: np.random.Generator
):
    n_draws, n_units = effect_draws.shape
    weights = rng.dirichlet(np.ones(n_units), size=n_draws)
    return np.sum(weights * effect_draws, axis=1)
