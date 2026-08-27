"""Random Fourier feature Bayesian regression and analytic mean-field VI.

Conditional on a sampled RFF map, the model is conjugate Bayesian linear
regression. The mean-field approximation implemented here is the exact
minimizer of KL(q || p) over Gaussian distributions with diagonal covariance.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class RFFConfig:
    n_rff: int = 200
    signal_sd: float = 1.0
    lengthscale: float = 0.8
    treatment_lengthscale: float = 0.5
    beta_prior_sd: float = 1.0
    noise_sd: float = 1.0
    jitter: float = 1e-8
    center_y: bool = True

@dataclass
class RFFMap:
    frequencies: np.ndarray
    phases: np.ndarray
    scale: float

    def transform(self, z: np.ndarray):
        z = np.asarray(z, dtype=float)
        return self.scale * np.cos(z @ self.frequencies + self.phases)


@dataclass
class RFFPosterior:
    mean: np.ndarray
    precision: np.ndarray
    covariance: np.ndarray
    meanfield_var: np.ndarray
    y_mean: float
    feature_map: RFFMap


def make_rff_map(input_dim: int, config: RFFConfig, rng: np.random.Generator):
    lengthscales = np.full(input_dim, config.lengthscale, dtype=float)
    lengthscales[-1] = config.treatment_lengthscale
    frequencies = rng.normal(
        loc=0.0,
        scale=1.0 / lengthscales[:, None],
        size=(input_dim, config.n_rff),
    )
    phases = rng.uniform(0.0, 2.0 * np.pi, size=config.n_rff)
    scale = config.signal_sd * np.sqrt(2.0 / config.n_rff)
    return RFFMap(frequencies=frequencies, phases=phases, scale=scale)


def fit_rff_bayesian_regression(
    train_z: np.ndarray,
    train_y: np.ndarray,
    config: RFFConfig,
    rng: np.random.Generator
):
    train_z = np.asarray(train_z, dtype=float)
    train_y = np.asarray(train_y, dtype=float)
    feature_map = make_rff_map(train_z.shape[1], config, rng)
    phi = feature_map.transform(train_z)

    y_mean = float(train_y.mean()) if config.center_y else 0.0
    y_centered = train_y - y_mean

    noise_var = config.noise_sd**2
    prior_var = config.beta_prior_sd**2
    precision = phi.T @ phi / noise_var + np.eye(config.n_rff) / prior_var
    chol_precision = stable_cholesky(precision, config.jitter)

    rhs = phi.T @ y_centered / noise_var
    mean = solve_cholesky(chol_precision, rhs)
    covariance = solve_cholesky(chol_precision, np.eye(config.n_rff))
    covariance = 0.5 * (covariance + covariance.T)
    meanfield_var = 1.0 / np.diag(precision)

    return RFFPosterior(
        mean=mean,
        precision=precision,
        covariance=covariance,
        meanfield_var=meanfield_var,
        y_mean=y_mean,
        feature_map=feature_map,
    )


def sample_full_beta(
    posterior: RFFPosterior,
    n_draws: int,
    rng: np.random.Generator,
    jitter: float = 1e-8,
):
    chol_cov = stable_cholesky(posterior.covariance, jitter)
    standard = rng.normal(size=(posterior.mean.size, n_draws))
    return (posterior.mean[:, None] + chol_cov @ standard).T


def sample_meanfield_beta(
    posterior: RFFPosterior,
    n_draws: int,
    rng: np.random.Generator,
):
    sd = np.sqrt(posterior.meanfield_var)
    standard = rng.normal(size=(n_draws, posterior.mean.size))
    return posterior.mean[None, :] + standard * sd[None, :]


def predict_from_beta_draws(
    beta_draws: np.ndarray,
    z: np.ndarray,
    posterior: RFFPosterior,
):
    phi = posterior.feature_map.transform(z)
    return beta_draws @ phi.T + posterior.y_mean


def counterfactual_draws_from_beta(
    beta_draws: np.ndarray,
    control_z: np.ndarray,
    treated_z: np.ndarray,
    posterior: RFFPosterior,
):
    mu0 = predict_from_beta_draws(beta_draws, control_z, posterior)
    mu1 = predict_from_beta_draws(beta_draws, treated_z, posterior)
    return mu0, mu1


def linear_functional_variance(
    posterior: RFFPosterior,
    coefficient: np.ndarray,
    meanfield: bool = False,
):
    """Return the posterior variance of coefficient.T @ beta."""

    coefficient = np.asarray(coefficient, dtype=float)
    if meanfield:
        variance = np.sum(coefficient**2 * posterior.meanfield_var)
    else:
        variance = coefficient @ posterior.covariance @ coefficient
    return float(max(variance, 0.0))


def linear_functional_variance_decomposition(
    posterior: RFFPosterior,
    coefficient: np.ndarray,
):
    """Decompose Var(coefficient.T @ beta) into diagonal and cross terms."""

    coefficient = np.asarray(coefficient, dtype=float)
    full_total = float(coefficient @ posterior.covariance @ coefficient)
    full_diagonal = float(
        np.sum(coefficient**2 * np.diag(posterior.covariance))
    )
    full_cross = full_total - full_diagonal
    meanfield_total = float(np.sum(coefficient**2 * posterior.meanfield_var))
    return {
        "full_total": full_total,
        "full_diagonal": full_diagonal,
        "full_cross": full_cross,
        "meanfield_total": meanfield_total,
    }


def meanfield_kl_to_full(posterior: RFFPosterior):
    """Compute KL(q_meanfield || p_full) for the conjugate Gaussian model."""

    _, logdet_precision = np.linalg.slogdet(posterior.precision)
    logdet_diagonal = float(np.log(np.diag(posterior.precision)).sum())
    return 0.5 * (logdet_diagonal - float(logdet_precision))


def coefficient_variance_ratios(posterior: RFFPosterior):
    """Return mean-field variance divided by full marginal variance."""

    full_marginal_var = np.diag(posterior.covariance)
    return posterior.meanfield_var / full_marginal_var


def solve_cholesky(chol: np.ndarray, rhs: np.ndarray):
    return np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))


def stable_cholesky(matrix: np.ndarray, jitter: float, max_tries: int = 6):
    eye = np.eye(matrix.shape[0])
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        pass

    current_jitter = jitter
    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(matrix + current_jitter * eye)
        except np.linalg.LinAlgError:
            current_jitter *= 10.0
    return np.linalg.cholesky(matrix + current_jitter * eye)
