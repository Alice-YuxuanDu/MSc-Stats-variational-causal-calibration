"""Evaluation utilities for causal effect experiments."""


import numpy as np


def posterior_draw_metrics(
    draws: np.ndarray,
    true_value: float,
    level: float = 0.95
):
    draws = np.asarray(draws, dtype=float)
    alpha = 1.0 - level
    lower, upper = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    mean = float(draws.mean())
    return {
        "posterior_mean": mean,
        "bias": mean - true_value,
        "abs_error": abs(mean - true_value),
        "squared_error": (mean - true_value) ** 2,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "ci_width": float(upper - lower),
        "covered": float(lower <= true_value <= upper),
        "posterior_sd": float(draws.std(ddof=1)) if draws.size > 1 else 0.0,
    }


def point_estimate_metrics(estimate: float, true_value: float):
    return {
        "estimate": float(estimate),
        "bias": float(estimate - true_value),
        "abs_error": float(abs(estimate - true_value)),
        "squared_error": float((estimate - true_value) ** 2),
    }


def aggregate_metric_rows(rows: list[dict[str, float | str | int]]):
    label_keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, str)
    ]
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float, np.integer, np.floating))
    ]
    out: dict[str, float | str] = {key: str(rows[0][key])
                                   for key in label_keys}
    for key in numeric_keys:
        values = np.array([float(row[key]) for row in rows], dtype=float)
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_se"] = (
            float(values.std(ddof=1) / np.sqrt(values.size))
            if values.size > 1
            else 0.0
        )
    return out
