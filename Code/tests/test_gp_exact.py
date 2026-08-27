"""Tests for exact-GP prediction and MML lengthscale selection."""

import unittest

import numpy as np

from src.gp_exact import (
    ExactGPConfig,
    exact_gp_log_marginal_likelihood,
    fit_gp_lengthscales_map,
    fit_gp_lengthscales_mml,
    make_gp_inputs,
    rbf_kernel,
)


class ExactGPMMLTests(unittest.TestCase):
    def test_mml_improves_log_marginal_likelihood(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(45, 2))
        a = rng.binomial(1, 0.5, size=45)
        z = make_gp_inputs(x, a)
        true_config = ExactGPConfig(
            lengthscale=1.6,
            treatment_lengthscale=0.35,
            signal_sd=1.0,
            noise_sd=0.15,
            jitter=1e-8,
            center_y=False,
        )
        covariance = rbf_kernel(z, z, true_config)
        covariance += true_config.noise_sd**2 * np.eye(z.shape[0])
        y = rng.multivariate_normal(np.zeros(z.shape[0]), covariance)
        initial_config = ExactGPConfig(
            lengthscale=0.25,
            treatment_lengthscale=2.0,
            signal_sd=true_config.signal_sd,
            noise_sd=true_config.noise_sd,
            jitter=1e-8,
            center_y=False,
        )

        result = fit_gp_lengthscales_mml(
            z,
            y,
            initial_config,
            bounds=(0.05, 10.0),
            n_restarts=2,
            maxiter=80,
            seed=7,
        )

        self.assertGreaterEqual(
            result.log_marginal_likelihood,
            result.initial_log_marginal_likelihood - 1e-8,
        )
        self.assertAlmostEqual(
            result.log_marginal_likelihood,
            exact_gp_log_marginal_likelihood(z, y, result.config),
            places=6,
        )
        self.assertEqual(result.config.signal_sd, initial_config.signal_sd)
        self.assertEqual(result.config.noise_sd, initial_config.noise_sd)
        self.assertTrue(0.05 <= result.lengthscale <= 10.0)
        self.assertTrue(0.05 <= result.treatment_lengthscale <= 10.0)

    def test_map_regularizes_weakly_identified_lengthscales(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(30, 2))
        a = rng.binomial(1, 0.5, size=30)
        z = make_gp_inputs(x, a)
        y = rng.normal(scale=0.02, size=30)
        config = ExactGPConfig(
            signal_sd=0.05,
            noise_sd=3.0,
            lengthscale=0.8,
            treatment_lengthscale=0.5,
            jitter=1e-8,
        )

        result = fit_gp_lengthscales_map(
            z,
            y,
            config,
            prior_medians=(0.8, 0.5),
            prior_log_sds=(0.1, 0.1),
            bounds=(0.05, 50.0),
            n_restarts=2,
            maxiter=80,
            seed=3,
        )

        self.assertEqual(result.selection_method, "map")
        self.assertGreaterEqual(
            result.log_posterior,
            result.initial_log_posterior - 1e-8,
        )
        self.assertAlmostEqual(
            result.log_posterior,
            result.log_marginal_likelihood + result.log_prior,
            places=8,
        )
        self.assertLess(abs(np.log(result.lengthscale / 0.8)), 0.02)
        self.assertLess(
            abs(np.log(result.treatment_lengthscale / 0.5)),
            0.02,
        )

if __name__ == "__main__":
    unittest.main()
