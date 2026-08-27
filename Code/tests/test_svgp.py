"""Tests for the full-covariance sparse variational GP."""

import unittest

import numpy as np

from src.gp_exact import (
    ExactGPConfig,
    fit_exact_gp_predictive,
    make_counterfactual_inputs,
    make_gp_inputs,
)
from src.svgp import (
    SVGPConfig,
    exact_gp_log_marginal_likelihood,
    fit_svgp,
    predict_svgp,
    sample_svgp_counterfactuals,
    select_balanced_inducing_inputs,
)


class SVGPTests(unittest.TestCase):
    def test_balanced_inducing_inputs_cover_both_treatments(self):
        rng = np.random.default_rng(4)
        train_z = make_gp_inputs(
            rng.normal(size=(20, 3)),
            np.zeros(20),
        )

        inducing = select_balanced_inducing_inputs(
            train_z,
            n_inducing=9,
            lengthscale=0.8,
        )

        self.assertEqual(inducing.shape, (9, 4))
        self.assertEqual(np.sum(inducing[:, -1] == 0.0), 5)
        self.assertEqual(np.sum(inducing[:, -1] == 1.0), 4)

    def test_elbo_is_below_exact_log_marginal_likelihood(self):
        rng = np.random.default_rng(8)
        train_z = make_gp_inputs(
            rng.normal(size=(24, 2)),
            rng.binomial(1, 0.5, size=24),
        )
        y = rng.normal(size=24)
        config = SVGPConfig(
            n_inducing=8,
            noise_sd=0.7,
            jitter=1e-8,
        )

        posterior = fit_svgp(train_z, y, config)
        exact = exact_gp_log_marginal_likelihood(train_z, y, config)

        self.assertLessEqual(posterior.elbo, exact + 1e-7)
        self.assertGreaterEqual(posterior.trace_gap, 0.0)
        self.assertGreaterEqual(posterior.relative_trace_gap, 0.0)

    def test_all_training_inputs_recover_exact_gp(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(12, 2))
        a = rng.binomial(1, 0.5, size=12)
        train_z = make_gp_inputs(x, a)
        y = rng.normal(size=12)
        control_z, treated_z = make_counterfactual_inputs(x)
        test_z = np.vstack([control_z, treated_z])
        svgp_config = SVGPConfig(
            n_inducing=train_z.shape[0],
            noise_sd=0.9,
            jitter=1e-10,
        )
        exact_config = ExactGPConfig(
            noise_sd=0.9,
            jitter=1e-10,
        )

        sparse = fit_svgp(
            train_z,
            y,
            svgp_config,
            inducing_z=train_z,
        )
        sparse_predictive = predict_svgp(sparse, test_z)
        exact_predictive = fit_exact_gp_predictive(
            train_z,
            y,
            test_z,
            exact_config,
        )

        np.testing.assert_allclose(
            sparse_predictive.mean,
            exact_predictive.mean,
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            sparse_predictive.cov,
            exact_predictive.cov,
            rtol=2e-5,
            atol=2e-6,
        )

    def test_counterfactual_sampling_shapes(self):
        rng = np.random.default_rng(21)
        x = rng.normal(size=(15, 2))
        train_z = make_gp_inputs(x, rng.binomial(1, 0.5, size=15))
        control_z, treated_z = make_counterfactual_inputs(x)
        posterior = fit_svgp(
            train_z,
            rng.normal(size=15),
            SVGPConfig(n_inducing=6),
        )
        predictive = predict_svgp(
            posterior,
            np.vstack([control_z, treated_z]),
        )

        mu0, mu1 = sample_svgp_counterfactuals(
            predictive,
            n_units=15,
            n_draws=30,
            rng=rng,
        )

        self.assertEqual(mu0.shape, (30, 15))
        self.assertEqual(mu1.shape, (30, 15))


if __name__ == "__main__":
    unittest.main()
