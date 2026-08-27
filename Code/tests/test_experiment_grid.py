"""Tests for the paired Module 4 experiment utilities."""

import unittest

import numpy as np

from run_module4_experiment_grid import (
    correction_suffix,
    fit_logistic_propensity,
    overlap_diagnostics,
    parse_int_grid,
    parse_pi_clip_grid,
    posterior_draw_sets,
    wilson_interval,
)
from run_module4_ihdp_rff import aggregate_metric_rows


class ExperimentGridTests(unittest.TestCase):
    def test_grid_parsers(self):
        self.assertEqual(parse_int_grid("50,100,200"), [50, 100, 200])
        self.assertEqual(parse_pi_clip_grid("0.01,0.05"), [0.01, 0.05])

    def test_shared_weights_determine_all_bootstrap_draws(self):
        effects = np.array([[1.0, 2.0], [3.0, 4.0]])
        eif = np.array([[0.5, -0.5], [1.0, -1.0]])
        weights = np.array([[0.25, 0.75], [0.6, 0.4]])

        draws = posterior_draw_sets(
            effects,
            eif,
            weights,
            weights,
            weights,
        )

        np.testing.assert_allclose(draws["sample_ate_raw"], [1.5, 3.5])
        np.testing.assert_allclose(
            draws["population_ate_bb_raw"],
            [1.75, 3.4],
        )
        np.testing.assert_allclose(
            draws["sample_ate_corrected"],
            [1.25, 3.7],
        )
        np.testing.assert_allclose(
            draws["population_ate_bb_corrected"],
            [1.5, 3.6],
        )

    def test_overlap_diagnostics_detect_extreme_weights(self):
        diagnostics = overlap_diagnostics(
            np.array([1.0, 0.0]),
            np.array([0.001, 0.999]),
            pi_clip=0.02,
        )

        self.assertEqual(diagnostics["propensity_clip_fraction"], 1.0)
        self.assertAlmostEqual(
            diagnostics["observed_inverse_weight_max"],
            50.0,
        )
        self.assertLessEqual(
            diagnostics["observed_inverse_weight_ess_fraction"],
            1.0,
        )

    def test_logistic_propensity_estimator_tracks_linear_assignment(self):
        x = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]])
        a = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

        pi_hat = fit_logistic_propensity(
            x,
            a,
            ridge=1.0,
            max_iter=100,
            tolerance=1e-10,
            clip=0.05,
        )

        self.assertEqual(correction_suffix("logistic"), "logistic_pi")
        self.assertTrue(np.all(pi_hat >= 0.05))
        self.assertTrue(np.all(pi_hat <= 0.95))
        self.assertGreater(pi_hat[-1], pi_hat[0])

    def test_wilson_interval_is_informative_at_full_coverage(self):
        lower, upper = wilson_interval(10, 10)
        self.assertGreater(lower, 0.7)
        self.assertAlmostEqual(upper, 1.0)

    def test_ihdp_mml_aggregation_keeps_realizations_together(self):
        base = {
            "n_rff": 100,
            "signal_sd": 1.0,
            "noise_sd": 0.5,
            "lengthscale_mode": "mml",
            "initial_lengthscale": 10.0,
            "initial_treatment_lengthscale": 0.5,
            "estimand": "sample_ate",
            "method": "rff_full_gaussian",
            "covered": 1.0,
            "posterior_sd": 0.1,
            "bias": 0.0,
            "abs_error": 0.0,
            "squared_error": 0.0,
            "ci_width": 0.4,
        }
        rows = [
            {
                **base,
                "realization": 0,
                "lengthscale": 9.0,
                "treatment_lengthscale": 0.7,
                "posterior_mean": 1.0,
            },
            {
                **base,
                "realization": 1,
                "lengthscale": 11.0,
                "treatment_lengthscale": 0.9,
                "posterior_mean": 1.1,
            },
        ]

        aggregated = aggregate_metric_rows(rows, level=0.95)

        self.assertEqual(len(aggregated), 1)
        self.assertEqual(aggregated[0]["n_reps"], 2)
        self.assertAlmostEqual(aggregated[0]["selected_lengthscale_mean"], 10.0)
        self.assertAlmostEqual(
            aggregated[0]["selected_treatment_lengthscale_mean"],
            0.8,
        )


if __name__ == "__main__":
    unittest.main()
