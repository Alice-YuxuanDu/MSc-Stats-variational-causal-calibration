# Analysis Summary: .

Generated: 2026-08-05T15:33:43

## CSV Inventory

| file | rows | columns |
| --- | --- | --- |
| dataset_summaries.csv | 800 | 11 |
| diagnostics.csv | 800 | 49 |
| paired_comparisons.csv | 32 | 10 |
| posterior_metrics.csv | 6400 | 16 |
| posterior_metrics_aggregated.csv | 32 | 20 |

## Results

### Dataset Diagnostics

| scenario | n_reps | n | treated_share | propensity_min | propensity_p05 | propensity_mean | propensity_p95 | propensity_max | sample_tau_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strong_heterogeneous | 200 | 250 | 0.5035 | 0.1558 | 0.2444 | 0.5015 | 0.7077 | 0.7704 | 0.9996 |
| strong_homogeneous | 200 | 250 | 0.4979 | 0.1546 | 0.2441 | 0.5022 | 0.7095 | 0.7694 | 1 |
| weak_heterogeneous | 200 | 250 | 0.5146 | 0.01637 | 0.06023 | 0.5137 | 0.8981 | 0.9495 | 0.9989 |
| weak_homogeneous | 200 | 250 | 0.5134 | 0.01632 | 0.06054 | 0.5145 | 0.8985 | 0.9496 | 1 |

### Posterior Calibration

| scenario | pi_clip | estimand | method | n_reps | bias | mean_abs_error | rmse | coverage | mean_ci_width | mean_posterior_sd | empirical_to_posterior_sd_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strong_heterogeneous | 0.02 | population_ate_bb | rff_full_gaussian | 200 | -0.04318 | 0.1147 | 0.1433 | 0.945 | 0.5355 | 0.137 | 1 |
| strong_heterogeneous | 0.02 | population_ate_bb | rff_full_gaussian_onestep_logistic_pi | 200 | -0.01334 | 0.1102 | 0.1385 | 0.97 | 0.5534 | 0.1413 | 0.9778 |
| strong_heterogeneous | 0.02 | population_ate_bb | rff_meanfield_vi | 200 | -0.04266 | 0.1148 | 0.1434 | 1 | 1.984 | 0.5072 | 0.2706 |
| strong_heterogeneous | 0.02 | population_ate_bb | rff_meanfield_vi_onestep_logistic_pi | 200 | -0.0136 | 0.1102 | 0.1382 | 0.98 | 0.6133 | 0.1562 | 0.883 |
| strong_homogeneous | 0.02 | population_ate_bb | rff_full_gaussian | 200 | -0.02612 | 0.116 | 0.1425 | 0.955 | 0.5319 | 0.1361 | 1.032 |
| strong_homogeneous | 0.02 | population_ate_bb | rff_full_gaussian_onestep_logistic_pi | 200 | 0.0005041 | 0.1169 | 0.1443 | 0.95 | 0.55 | 0.1405 | 1.03 |
| strong_homogeneous | 0.02 | population_ate_bb | rff_meanfield_vi | 200 | -0.02505 | 0.1153 | 0.142 | 1 | 2.121 | 0.5423 | 0.2585 |
| strong_homogeneous | 0.02 | population_ate_bb | rff_meanfield_vi_onestep_logistic_pi | 200 | 0.0003048 | 0.1166 | 0.1442 | 0.975 | 0.6092 | 0.1552 | 0.931 |
| weak_heterogeneous | 0.02 | population_ate_bb | rff_full_gaussian | 200 | 0.01634 | 0.1346 | 0.1681 | 0.935 | 0.6048 | 0.155 | 1.082 |
| weak_heterogeneous | 0.02 | population_ate_bb | rff_full_gaussian_onestep_logistic_pi | 200 | 0.03075 | 0.1453 | 0.1793 | 0.955 | 0.6705 | 0.1704 | 1.039 |
| weak_heterogeneous | 0.02 | population_ate_bb | rff_meanfield_vi | 200 | 0.01823 | 0.1351 | 0.1689 | 1 | 2.04 | 0.5209 | 0.3232 |
| weak_heterogeneous | 0.02 | population_ate_bb | rff_meanfield_vi_onestep_logistic_pi | 200 | 0.03056 | 0.1458 | 0.1798 | 0.965 | 0.7502 | 0.1903 | 0.9333 |
| weak_homogeneous | 0.02 | population_ate_bb | rff_full_gaussian | 200 | -0.0001716 | 0.1286 | 0.1591 | 0.95 | 0.6007 | 0.1537 | 1.038 |
| weak_homogeneous | 0.02 | population_ate_bb | rff_full_gaussian_onestep_logistic_pi | 200 | 0.005954 | 0.1365 | 0.1734 | 0.94 | 0.6546 | 0.1663 | 1.045 |
| weak_homogeneous | 0.02 | population_ate_bb | rff_meanfield_vi | 200 | -6.667e-05 | 0.1287 | 0.1592 | 1 | 2.139 | 0.5467 | 0.292 |
| weak_homogeneous | 0.02 | population_ate_bb | rff_meanfield_vi_onestep_logistic_pi | 200 | 0.005909 | 0.1367 | 0.1735 | 0.96 | 0.7348 | 0.1868 | 0.9308 |

### Paired Comparisons

| scenario | pi_clip | estimand | comparison | n_reps | posterior_mean_difference | abs_error_difference | posterior_sd_ratio | ci_width_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strong_heterogeneous | 0.02 | population_ate_bb | meanfield_minus_full_onestep_logistic_pi | 200 | -0.0002591 | -5.079e-05 | 1.106 | 1.109 |
| strong_heterogeneous | 0.02 | population_ate_bb | meanfield_minus_full_raw | 200 | 0.0005192 | 0.0001278 | 3.707 | 3.712 |
| strong_heterogeneous | 0.02 | population_ate_bb | onestep_logistic_pi_minus_raw_full | 200 | 0.02983 | -0.004481 | 1.032 | 1.034 |
| strong_heterogeneous | 0.02 | population_ate_bb | onestep_logistic_pi_minus_raw_meanfield | 200 | 0.02906 | -0.004659 | 0.3153 | 0.3165 |
| strong_homogeneous | 0.02 | population_ate_bb | meanfield_minus_full_onestep_logistic_pi | 200 | -0.0001993 | -0.0002681 | 1.106 | 1.109 |
| strong_homogeneous | 0.02 | population_ate_bb | meanfield_minus_full_raw | 200 | 0.00107 | -0.0006866 | 3.989 | 3.994 |
| strong_homogeneous | 0.02 | population_ate_bb | onestep_logistic_pi_minus_raw_full | 200 | 0.02663 | 0.0009451 | 1.033 | 1.034 |
| strong_homogeneous | 0.02 | population_ate_bb | onestep_logistic_pi_minus_raw_meanfield | 200 | 0.02536 | 0.001364 | 0.296 | 0.297 |
| weak_heterogeneous | 0.02 | population_ate_bb | meanfield_minus_full_onestep_logistic_pi | 200 | -0.0001875 | 0.000434 | 1.12 | 1.122 |
| weak_heterogeneous | 0.02 | population_ate_bb | meanfield_minus_full_raw | 200 | 0.001889 | 0.000495 | 3.366 | 3.379 |
| weak_heterogeneous | 0.02 | population_ate_bb | onestep_logistic_pi_minus_raw_full | 200 | 0.01441 | 0.01076 | 1.099 | 1.108 |
| weak_heterogeneous | 0.02 | population_ate_bb | onestep_logistic_pi_minus_raw_meanfield | 200 | 0.01234 | 0.0107 | 0.3739 | 0.3764 |
| weak_homogeneous | 0.02 | population_ate_bb | meanfield_minus_full_onestep_logistic_pi | 200 | -4.538e-05 | 0.0002159 | 1.126 | 1.126 |
| weak_homogeneous | 0.02 | population_ate_bb | meanfield_minus_full_raw | 200 | 0.000105 | 8.165e-05 | 3.565 | 3.569 |
| weak_homogeneous | 0.02 | population_ate_bb | onestep_logistic_pi_minus_raw_full | 200 | 0.006126 | 0.007869 | 1.082 | 1.089 |
| weak_homogeneous | 0.02 | population_ate_bb | onestep_logistic_pi_minus_raw_meanfield | 200 | 0.005975 | 0.008003 | 0.3536 | 0.3556 |

### Computational Diagnostics

| scenario | meanfield_kl_to_full |
| --- | --- |
| strong_heterogeneous | 55.41 |
| strong_homogeneous | 56.54 |
| weak_heterogeneous | 55.7 |
| weak_homogeneous | 56.81 |
