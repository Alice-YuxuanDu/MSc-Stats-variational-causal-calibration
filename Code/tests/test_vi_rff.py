"""Tests for the conjugate RFF posterior and analytic mean-field VI."""

import unittest

import numpy as np

from src.vi_rff import (
    RFFConfig,
    RFFMap,
    RFFPosterior,
    coefficient_variance_ratios,
    fit_rff_bayesian_regression,
    linear_functional_variance,
    linear_functional_variance_decomposition,
    meanfield_kl_to_full,
)


class RFFMeanFieldTests(unittest.TestCase):
    def test_fit_matches_closed_form_gaussian_posterior(self) -> None:
        rng = np.random.default_rng(7)
        z = rng.normal(size=(12, 3))
        y = rng.normal(size=12)
        config = RFFConfig(
            n_rff=8,
            noise_sd=0.7,
            beta_prior_sd=1.3,
            center_y=False,
        )

        posterior = fit_rff_bayesian_regression(z, y, config, rng)
        phi = posterior.feature_map.transform(z)
        expected_precision = (
            phi.T @ phi / config.noise_sd**2
            + np.eye(config.n_rff) / config.beta_prior_sd**2
        )
        expected_covariance = np.linalg.inv(expected_precision)
        expected_mean = (
            expected_covariance @ phi.T @ y / config.noise_sd**2
        )

        np.testing.assert_allclose(posterior.precision, expected_precision)
        np.testing.assert_allclose(
            posterior.covariance,
            expected_covariance,
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(posterior.mean, expected_mean)
        np.testing.assert_allclose(
            posterior.meanfield_var,
            1.0 / np.diag(expected_precision),
        )

    def test_meanfield_shrinks_each_coefficient_marginal(self) -> None:
        rng = np.random.default_rng(11)
        z = rng.normal(size=(30, 4))
        y = rng.normal(size=30)
        posterior = fit_rff_bayesian_regression(
            z,
            y,
            RFFConfig(n_rff=15),
            rng,
        )

        ratios = coefficient_variance_ratios(posterior)
        self.assertTrue(np.all(ratios > 0.0))
        self.assertTrue(np.all(ratios <= 1.0 + 1e-12))
        self.assertGreaterEqual(meanfield_kl_to_full(posterior), 0.0)

    def test_linear_functional_can_be_wider_under_meanfield(self) -> None:
        precision = np.array([[2.0, 1.8], [1.8, 2.0]])
        covariance = np.linalg.inv(precision)
        posterior = RFFPosterior(
            mean=np.zeros(2),
            precision=precision,
            covariance=covariance,
            meanfield_var=1.0 / np.diag(precision),
            y_mean=0.0,
            feature_map=RFFMap(
                frequencies=np.eye(2),
                phases=np.zeros(2),
                scale=1.0,
            ),
        )
        coefficient = np.ones(2)

        full_variance = linear_functional_variance(posterior, coefficient)
        meanfield_variance = linear_functional_variance(
            posterior,
            coefficient,
            meanfield=True,
        )

        self.assertGreater(meanfield_variance, full_variance)
        self.assertTrue(np.all(coefficient_variance_ratios(posterior) < 1.0))

    def test_linear_functional_variance_decomposition(self) -> None:
        covariance = np.array(
            [
                [2.0, -0.5],
                [-0.5, 1.0],
            ]
        )
        posterior = RFFPosterior(
            mean=np.zeros(2),
            precision=np.linalg.inv(covariance),
            covariance=covariance,
            meanfield_var=np.array([0.75, 0.25]),
            y_mean=0.0,
            feature_map=RFFMap(
                frequencies=np.eye(2),
                phases=np.zeros(2),
                scale=1.0,
            ),
        )
        coefficient = np.array([2.0, 3.0])

        decomposition = linear_functional_variance_decomposition(
            posterior,
            coefficient,
        )

        self.assertAlmostEqual(decomposition["full_diagonal"], 17.0)
        self.assertAlmostEqual(decomposition["full_cross"], -6.0)
        self.assertAlmostEqual(decomposition["full_total"], 11.0)
        self.assertAlmostEqual(decomposition["meanfield_total"], 5.25)

if __name__ == "__main__":
    unittest.main()
