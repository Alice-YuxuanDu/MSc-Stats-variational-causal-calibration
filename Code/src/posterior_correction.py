"""One-step posterior correction utilities for causal functionals."""


import numpy as np


def outcome_eif_draws(
    mu0_draws: np.ndarray,
    mu1_draws: np.ndarray,
    a: np.ndarray,
    y: np.ndarray,
    pi: np.ndarray,
    pi_clip: float = 0.02
):
    """Compute the outcome part of the ATE efficient influence function.

    For each posterior draw of the nuisance outcome functions, this returns

        A / pi(X) * {Y - mu_1(X)}
        - (1 - A) / {1 - pi(X)} * {Y - mu_0(X)}.

    The output has shape (n_draws, n_units).
    """

    mu0_draws = np.asarray(mu0_draws, dtype=float)
    mu1_draws = np.asarray(mu1_draws, dtype=float)
    a = np.asarray(a, dtype=float)
    y = np.asarray(y, dtype=float)
    pi = np.clip(np.asarray(pi, dtype=float), pi_clip, 1.0 - pi_clip)

    treated_residual = (a / pi) * (y[None, :] - mu1_draws)
    control_residual = ((1.0 - a) / (1.0 - pi)) * (y[None, :] - mu0_draws)
    return treated_residual - control_residual


def one_step_sample_cate_draws(
    effect_draws: np.ndarray,
    eif_y_draws: np.ndarray,
    rng: np.random.Generator
):
    """One-step corrected posterior draws for the sample CATE estimand."""

    effect_draws = np.asarray(effect_draws, dtype=float)
    eif_y_draws = np.asarray(eif_y_draws, dtype=float)

    n_draws, n_units = effect_draws.shape
    weights = rng.dirichlet(np.ones(n_units), size=n_draws)
    raw = np.mean(effect_draws, axis=1)
    correction = np.sum(weights * eif_y_draws, axis=1)
    return raw + correction


def one_step_population_ate_draws(
    effect_draws: np.ndarray,
    eif_y_draws: np.ndarray,
    rng: np.random.Generator
):
    """One-step corrected posterior draws for the population ATE.

    This is the Bayesian-bootstrap analogue of the AIPW signal:

        tau(X) + A/pi(X){Y-mu_1(X)} - (1-A)/(1-pi(X)){Y-mu_0(X)}.
    """

    effect_draws = np.asarray(effect_draws, dtype=float)
    eif_y_draws = np.asarray(eif_y_draws, dtype=float)

    n_draws, n_units = effect_draws.shape
    weights = rng.dirichlet(np.ones(n_units), size=n_draws)
    aipw_signal = effect_draws + eif_y_draws
    return np.sum(weights * aipw_signal, axis=1)
