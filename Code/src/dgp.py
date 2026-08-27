"""Synthetic causal data-generating processes for ATE experiments.

The first DGP is deliberately simple enough to know the true ATE exactly,
while still containing confounding, nonlinear response surfaces, and tunable
overlap. Later modules can reuse the same interface for GP/VI methods.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np


Overlap = Literal["strong", "weak"]
# Literal type for overlap regimes: "strong" or "weak"
Effect = Literal["homogeneous", "heterogeneous"]
# Literal type for effect regimes: "homogeneous" or "heterogeneous"


@dataclass
class SyntheticCausalDGP:
    """Nonlinear potential-outcome DGP with binary treatment.

    Covariates are sampled independently from Uniform(-1, 1). Treatment is
    assigned by a logistic propensity score. Outcomes are generated from

        Y = mu_0(X) + A * tau(X) + epsilon.

    The heterogeneous treatment effect is centered so that E[tau(X)] = 1 under
    the covariate distribution, giving an exact population ATE of 1.0 in both
    effect regimes.
    """

    n_features: int = 5
    overlap: Overlap = "strong"
    effect: Effect = "heterogeneous"
    noise_sd: float = 1.0

    def sample_covariates(self, n: int, rng: np.random.Generator):
        return rng.uniform(-1.0, 1.0, size=(n, self.n_features))

    def propensity(self, x: np.ndarray):
        x = np.asarray(x)
        scale = 0.9 if self.overlap == "strong" else 2.2
        linear = scale * (
            0.9 * x[:, 0]
            - 0.7 * x[:, 1]
            + 0.35 * x[:, 0] * x[:, 1]
            + 0.25 * np.sin(np.pi * x[:, 2])
        )
        return sigmoid(linear)

    def mu0(self, x: np.ndarray):
        x = np.asarray(x)
        return (
            0.7 * np.sin(np.pi * x[:, 0])
            + 0.5 * x[:, 1] ** 2
            - 0.4 * x[:, 2]
            + 0.25 * x[:, 3] * x[:, 4]
        )

    def tau(self, x: np.ndarray):
        x = np.asarray(x)
        if self.effect == "homogeneous":
            return np.ones(x.shape[0])
        return 1.0 + 0.5 * x[:, 0] + 0.25 * np.sin(np.pi * x[:, 1])

    def mu1(self, x: np.ndarray):
        return self.mu0(x) + self.tau(x)

    @property
    def true_ate(self):
        return 1.0

    def sample(self, n: int, seed: int | None = None):
        rng = np.random.default_rng(seed)
        x = self.sample_covariates(n, rng)
        pi = self.propensity(x)
        a = rng.binomial(1, pi)
        mu0 = self.mu0(x)
        tau = self.tau(x)
        mu1 = mu0 + tau
        y0 = mu0 + rng.normal(0.0, self.noise_sd, size=n)
        y1 = mu1 + rng.normal(0.0, self.noise_sd, size=n)
        y = np.where(a == 1, y1, y0)

        return {
            "x": x,
            "a": a,
            "y": y,
            "pi": pi,
            "mu0": mu0,
            "mu1": mu1,
            "tau": tau,
            "y0": y0,
            "y1": y1,
            "true_ate": np.array(self.true_ate),
        }


def sigmoid(z: np.ndarray):
    z = np.asarray(z)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def available_scenarios():
    return [
        ("strong", "homogeneous"),
        ("strong", "heterogeneous"),
        ("weak", "homogeneous"),
        ("weak", "heterogeneous"),
    ]


def summarize_dataset(data: dict[str, np.ndarray]):
    a = data["a"]
    pi = data["pi"]
    tau = data["tau"]
    return {
        "n": float(a.size),
        "treated_share": float(a.mean()),
        "propensity_min": float(pi.min()),
        "propensity_p05": float(np.quantile(pi, 0.05)),
        "propensity_mean": float(pi.mean()),
        "propensity_p95": float(np.quantile(pi, 0.95)),
        "propensity_max": float(pi.max()),
        "sample_tau_mean": float(tau.mean()),
    }
