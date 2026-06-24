from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence, cast
import math
import warnings

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import kstest


from singleGP_model import (
    _assert_z_score_invariance,
    _raw_observation_arrays,
    inverse_transform_predictions,
    object_level_empirical_flux_scale,
    predict_observation_distribution,
)

PREDICTIVE_STD_EPSILON = 1e-12


def _pit_bin_edges(bins):
    """Compute the edges of the PIT bins."""
    if isinstance(bins, (int, np.integer)):
        return np.linspace(0.0, 1.0, int(bins) + 1)
    return np.asarray(bins, dtype=float)


def _object_array_from_result(result, primary_key, n_values, fallback_key=None):
    value = result.get(primary_key, None)
    if value is None and fallback_key is not None:
        value = result.get(fallback_key, None)
    if value is None:
        return np.full(n_values, None, dtype=object)

    values = np.asarray(value, dtype=object)
    if values.ndim == 0:
        return np.full(n_values, values.item(), dtype=object)
    return values


def _valid_pit_values(pit_values):
    pit_values = np.asarray(pit_values, dtype=float).reshape(-1)
    return pit_values[np.isfinite(pit_values)]


def compute_pit_values(y_true, mu_pred, sigma_pred, epsilon=PREDICTIVE_STD_EPSILON):
    """
    Compute standardized residuals and probability integral transform values.
    """
    y_true = np.asarray(y_true, dtype=float)
    mu_pred = np.asarray(mu_pred, dtype=float)
    sigma_pred = np.maximum(np.asarray(sigma_pred, dtype=float), epsilon)
    if not np.all(sigma_pred > 0):
        raise AssertionError("sigma_pred must be positive after clipping.")

    z_values = (y_true - mu_pred) / sigma_pred
    pit_values = np.clip(ndtr(z_values), 0.0, 1.0)
    return z_values, pit_values


def compute_ks_pit(pit_values):
    """
    Compute the Kolmogorov-Smirnov distance between PIT values and Uniform(0, 1).
    """
    valid_pit = _valid_pit_values(pit_values)
    if len(valid_pit) == 0:
        return np.nan

    return float(kstest(valid_pit, "uniform").statistic)


def compute_pit_histogram(pit_values, bins=20, density=True):
    """
    Histogram PIT values on the fixed [0, 1] range.
    """
    valid_pit = _valid_pit_values(pit_values)
    bin_edges = _pit_bin_edges(bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    if len(valid_pit) == 0:
        return bin_edges, bin_centers, np.full(len(bin_centers), np.nan)

    hist_values, bin_edges = np.histogram(valid_pit, bins=bin_edges, range=(0.0, 1.0), density=density)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_edges, bin_centers, hist_values.astype(float)


def compute_pit_reliability_curve(pit_values, q_grid=None):
    """
    Empirical PIT CDF evaluated over a nominal probability grid.
    """
    if q_grid is None:
        q_grid = np.linspace(0.0, 1.0, 101)
    q_grid = np.asarray(q_grid, dtype=float)
    valid_pit = _valid_pit_values(pit_values)
    if len(valid_pit) == 0:
        return q_grid, np.full_like(q_grid, np.nan, dtype=float)

    empirical_cdf_values = np.asarray([np.mean(valid_pit <= q) for q in q_grid], dtype=float)
    return q_grid, empirical_cdf_values


def compute_object_weighted_pit_histogram(pit_by_object, bins=20):
    """
    Average density-normalized PIT histograms with each object weighted equally.
    """
    bin_edges = _pit_bin_edges(bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    densities = []
    for pit_values in pit_by_object.values():
        valid_pit = _valid_pit_values(pit_values)
        if len(valid_pit) == 0:
            continue
        hist_values, _ = np.histogram(valid_pit, bins=bin_edges, range=(0.0, 1.0), density=True)
        densities.append(hist_values)
    if len(densities) == 0:
        return bin_edges, bin_centers, np.full(len(bin_centers), np.nan)
    return bin_edges, bin_centers, np.mean(np.vstack(densities), axis=0)


def compute_object_weighted_pit_reliability_curve(pit_by_object, q_grid=None):
    """
    Average object-level PIT empirical CDFs with each object weighted equally.
    """
    if q_grid is None:
        q_grid = np.linspace(0.0, 1.0, 101)
    q_grid = np.asarray(q_grid, dtype=float)
    curves = []
    for pit_values in pit_by_object.values():
        valid_pit = _valid_pit_values(pit_values)
        if len(valid_pit) == 0:
            continue
        _, empirical_cdf = compute_pit_reliability_curve(valid_pit, q_grid=q_grid)
        curves.append(empirical_cdf)
    if len(curves) == 0:
        return q_grid, np.full_like(q_grid, np.nan, dtype=float)
    return q_grid, np.mean(np.vstack(curves), axis=0)


def compute_object_weighted_ks_pit(pit_by_object, q_grid=None):
    """
    KS-PIT from the average object-level PIT empirical CDF.
    """
    q_grid, mean_empirical_cdf = compute_object_weighted_pit_reliability_curve(
        pit_by_object,
        q_grid=q_grid,
    )
    valid = np.isfinite(mean_empirical_cdf) & np.isfinite(q_grid)
    if not np.any(valid):
        return np.nan
    return float(np.max(np.abs(mean_empirical_cdf[valid] - q_grid[valid])))


def plot_pit_histogram(pit_values, title=None, bins=20, save_path=None, density=True):
    """
    Plot a PIT histogram with the Uniform(0, 1) reference density.
    """
    import matplotlib.pyplot as plt

    bin_edges, _, hist_values = compute_pit_histogram(pit_values, bins=bins, density=density)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        bin_edges[:-1],
        hist_values,
        width=np.diff(bin_edges),
        align="edge",
        edgecolor="black",
        alpha=0.8,
    )
    if density:
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="ideal")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density" if density else "Count")
    if title is not None:
        ax.set_title(title)
    if density:
        ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def plot_pit_reliability_curve(pit_values, title=None, q_grid=None, save_path=None):
    """
    Plot the empirical PIT CDF against the ideal diagonal.
    """
    import matplotlib.pyplot as plt

    q_grid, empirical_cdf_values = compute_pit_reliability_curve(pit_values, q_grid=q_grid)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(q_grid, empirical_cdf_values, label="empirical")
    ax.plot([0.0, 1.0], [0.0, 1.0], color="black", linestyle="--", linewidth=1, label="ideal")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Nominal probability q")
    ax.set_ylabel("Empirical fraction PIT <= q")
    if title is not None:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def _pit_by_object_from_results(object_results):
    pit_by_object = {}
    for result_idx, result in enumerate(object_results):
        object_id = _hashable_object_id(_scalar_from_result_value(result.get("object_id", None)))
        if object_id is None:
            object_id = result_idx
        pit_values = result.get("pit_values", None)
        if pit_values is None:
            _, pit_values = compute_pit_values(result["y_true"], result["y_pred"], result["y_std"])
        pit_by_object[object_id] = np.asarray(pit_values, dtype=float).reshape(-1)
    return pit_by_object


def negative_log_predictive_density(y_true, mean, variance):
    """
    Compute pointwise negative log predictive density (NLPD) under a Gaussian.
    """
    y_true = np.asarray(y_true)
    mean = np.asarray(mean)
    variance = np.maximum(np.asarray(variance), PREDICTIVE_STD_EPSILON)

    return 0.5 * (np.log(2 * np.pi * variance) + ((y_true - mean) ** 2) / variance)   # Gaussian NLPD formula


def gaussian_crps(y_true, mean, sigma_pred, epsilon=PREDICTIVE_STD_EPSILON):
    """
    Compute the closed-form Gaussian CRPS loss for each prediction.

    Lower CRPS is better. The predictive standard deviation is clipped with the
    same small numerical floor used elsewhere for standardized residuals.
    """
    y_true = np.asarray(y_true, dtype=float)
    mean = np.asarray(mean, dtype=float)
    sigma_pred = np.maximum(np.asarray(sigma_pred, dtype=float), epsilon)
    if not np.all(sigma_pred > 0):
        raise AssertionError("sigma_pred must be positive after clipping.")

    z = (y_true - mean) / sigma_pred
    phi = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    # Closed-form Gaussian CRPS loss.
    return sigma_pred * (z * (2 * ndtr(z) - 1) + 2 * phi - 1 / np.sqrt(np.pi))


def RMSE(y_true, y_pred):
    """Compute root mean squared error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def cover_factor(gp, data, sigma_multiplier=1.0):
    """
    Compute the fraction of held-out observations that fall within the GP predictive mean ± sigma_multiplier * predictive std.
    """
    mean, std, _ = predict_observation_distribution(gp, data)
    lower_bound = mean - sigma_multiplier * std
    upper_bound = mean + sigma_multiplier * std

    covered = (data["y"] >= lower_bound) & (data["y"] <= upper_bound)
    cover_fraction = np.mean(covered)

    return cover_fraction


def evaluate_heldout_metrics(
        gp,
        heldout_data,
        train_data=None,
        object_data=None,
        object_flux_scale=None,
        nrmse_quantile=0.95,
        nrmse_epsilon=1e-8,
        coverage_sigmas=(1.0, 2.0, 3.0),
        include_yerr=True,
        yerr_scale=1.0,
        noise_floor=0.0,
        extra_noise=None,
        peak_window=0.25,
        evaluate_raw_metrics=True,
        assert_z_invariance=True,
):
    """
    Evaluate coverage in normalized GP space and NLPD/RMSE in raw flux space by default.
    If evaluate_raw_metrics=False, NLPD/RMSE are computed in normalized space instead.
    """
    mean_norm, std_norm, variance_norm = predict_observation_distribution(
        gp,
        heldout_data,
        include_yerr=include_yerr,
        yerr_scale=yerr_scale,
        noise_floor=noise_floor,
        extra_noise=extra_noise,
    )
    std_norm = np.maximum(std_norm, PREDICTIVE_STD_EPSILON)
    if not np.all(std_norm > 0):
        raise AssertionError("sigma_pred must be positive after clipping.")

    y_true_norm = np.asarray(heldout_data["y"])
    n_heldout = len(y_true_norm)
    if n_heldout == 0:
        raise ValueError("Each object must have at least one held-out prediction.")
    errors_norm = y_true_norm - mean_norm

    coverage = {}
    coverage_counts = {}
    for sigma in coverage_sigmas:
        covered = np.abs(errors_norm) <= sigma * std_norm
        key = f"coverage_{sigma:g}sigma"
        coverage[key] = float(np.mean(covered))
        coverage_counts[key] = int(np.sum(covered))

    y_raw, yerr_raw, scale, background = _raw_observation_arrays(heldout_data)
    mean_raw, variance_raw = inverse_transform_predictions(
        mean_norm,
        variance_norm,
        scale,
        background,
    )
    raw_std_floor = PREDICTIVE_STD_EPSILON * scale
    std_raw = np.maximum(np.sqrt(variance_raw), raw_std_floor)
    variance_raw = np.maximum(variance_raw, raw_std_floor ** 2)

    if assert_z_invariance:
        _assert_z_score_invariance(
            errors_norm,
            std_norm,
            y_raw,
            mean_raw,
            std_raw,
            scale,
        )

    if evaluate_raw_metrics:
        y_metric = y_raw
        mean_metric = mean_raw
        std_metric = std_raw
        variance_metric = variance_raw
        yerr_metric = yerr_raw
        metric_space = "raw"
    else:
        y_metric = y_true_norm
        mean_metric = mean_norm
        std_metric = std_norm
        variance_metric = variance_norm
        yerr_metric = np.asarray(heldout_data["yerr"])
        metric_space = "normalized"

    errors_metric = y_metric - mean_metric
    squared_errors = errors_metric ** 2
    z_values, pit_values = compute_pit_values(y_metric, mean_metric, std_metric)
    ks_pit_object = compute_ks_pit(pit_values)
    n_pit_object = int(np.sum(np.isfinite(pit_values)))
    rmse = float(np.sqrt(np.mean(squared_errors)))
    if object_flux_scale is None:
        if object_data is not None:
            object_flux_scale = object_level_empirical_flux_scale(
                example=object_data,
                q=nrmse_quantile,
                epsilon=nrmse_epsilon,
            )
        else:
            object_flux_scale = object_level_empirical_flux_scale(
                    train_data,
                    heldout_data,
                    q=nrmse_quantile,
                    epsilon=nrmse_epsilon,
            )
    object_flux_scale = float(max(float(object_flux_scale), nrmse_epsilon))
    per_point_nlpd = negative_log_predictive_density(y_metric, mean_metric, variance_metric)
    per_point_crps = gaussian_crps(y_metric, mean_metric, std_metric)

    t_test = np.asarray(heldout_data["t"])
    band = heldout_data.get("band", None)
    obj_id = heldout_data.get("obj_id", None)

    if train_data is not None:
        train_t = np.asarray(train_data["t"])
        train_time_min = float(np.min(train_t))
        train_time_max = float(np.max(train_t))
        # Below computation can answer: Does GP calibration degrade for extrapolative held-out points?
        # marks held-out points that are outside the training time range
        outside_train_range = (t_test < train_time_min) | (t_test > train_time_max)  
        # This measures how far outside the training range each held-out point is.
        distance_to_train_range = np.maximum.reduce([
            train_time_min - t_test,
            t_test - train_time_max,
            np.zeros_like(t_test),
        ])
        # This counts the number of training points. Useful for spotting sparse-object failures.
        n_train = len(train_t)
    else:
        train_time_min = np.nan
        train_time_max = np.nan
        outside_train_range = np.full(n_heldout, False)
        distance_to_train_range = np.full(n_heldout, np.nan)
        n_train = None

    # This marks whether each held-out point is close to the object’s global peak time
    if train_data is not None:
        all_t = np.concatenate([train_data["t"], heldout_data["t"]])
        all_y = np.concatenate([train_data["y"], heldout_data["y"]])

        peak_time = all_t[np.argmax(np.abs(all_y))]
    else:
        peak_time = t_test[np.argmax(np.abs(y_true_norm))]
    near_peak = np.abs(t_test - peak_time) <= peak_window

    return {
        "n_heldout": n_heldout,
        "n_train": n_train,
        "train_time_min": train_time_min,
        "train_time_max": train_time_max,
        "metric_space": metric_space,
        "mean_nlpd": float(np.mean(per_point_nlpd)),
        "total_nlpd": float(np.sum(per_point_nlpd)),
        "mean_crps": float(np.mean(per_point_crps)),
        "total_crps": float(np.sum(per_point_crps)),
        "rmse": rmse,
        "nrmse": float(rmse / object_flux_scale),
        "ncrps": float(np.mean(per_point_crps) / object_flux_scale),
        "sse": float(np.sum(squared_errors)),
        "object_flux_scale": object_flux_scale,
        "nrmse_quantile": float(nrmse_quantile),
        "coverage": coverage,
        "coverage_counts": coverage_counts,
        "per_point_nlpd": per_point_nlpd,
        "per_point_crps": per_point_crps,
        "squared_errors": squared_errors,
        "z_values": z_values,
        "pit_values": pit_values,
        "ks_pit_object": ks_pit_object,
        "n_pit_object": n_pit_object,
        "y_true": y_metric,
        "y_pred": mean_metric,
        "y_std": std_metric,
        "yerr": yerr_metric,
        "y_true_norm": y_true_norm,
        "y_pred_norm": mean_norm,
        "y_std_norm": std_norm,
        "predictive_variance_norm": variance_norm,
        "y_true_raw": y_raw,
        "y_pred_raw": mean_raw,
        "y_std_raw": std_raw,
        "yerr_raw": yerr_raw,
        "time": t_test,
        "X_test": np.asarray(heldout_data["X"]).reshape(n_heldout, -1),
        "object_id": np.repeat(obj_id, n_heldout),
        "band": np.repeat(band, n_heldout),
        "outside_train_range": outside_train_range,
        "distance_to_train_range": distance_to_train_range,
        "near_peak": near_peak,
        "predictive_variance": variance_metric,
        "flux_scale": scale,
        "background_flux": background,
    }


def summarize_object_metric_results(object_results):
    """
    Aggregate per-object held-out metrics two ways.

    Observation-weighted metrics pool all held-out observations together.
    Object-weighted metrics average the per-object metric values equally.
    """
    if len(object_results) == 0:
        raise ValueError("object_results must contain at least one result.")

    n_objects = len(object_results)
    n_total = int(sum(result["n_heldout"] for result in object_results))
    if n_total == 0:
        raise ValueError("At least one held-out observation is required.")
    train_total = int(sum(result["n_train"] for result in object_results if result["n_train"] is not None))

    coverage_keys = sorted(object_results[0]["coverage"].keys())
    pit_by_object = _pit_by_object_from_results(object_results)
    valid_pit_arrays = [
        valid_pit
        for valid_pit in (_valid_pit_values(pit_values) for pit_values in pit_by_object.values())
        if len(valid_pit) > 0
    ]
    pooled_pit = np.concatenate(valid_pit_arrays) if valid_pit_arrays else np.array([], dtype=float)
    ks_pit_obs_weighted = compute_ks_pit(pooled_pit)
    ks_pit_object_weighted = compute_object_weighted_ks_pit(pit_by_object)

    observation_weighted = {
        "nlpd": float(sum(result["total_nlpd"] for result in object_results) / n_total),
        "crps": float(sum(result["total_crps"] for result in object_results) / n_total),
        "rmse": float(np.sqrt(sum(result["sse"] for result in object_results) / n_total)),
        "nrmse": float(np.sqrt(np.average(
            [result["nrmse"] ** 2 for result in object_results],
            weights=[result["n_heldout"] for result in object_results],
        ))),
        "ks_pit": ks_pit_obs_weighted,
    }
    object_weighted = {
        "nlpd": float(np.mean([result["mean_nlpd"] for result in object_results])),
        "crps": float(np.mean([result["mean_crps"] for result in object_results])),
        "rmse": float(np.mean([result["rmse"] for result in object_results])),
        "nrmse": float(np.mean([result["nrmse"] for result in object_results])),
        "ks_pit": ks_pit_object_weighted,
    }

    for key in coverage_keys:
        observation_weighted[key] = float(
            sum(result["coverage_counts"][key] for result in object_results) / n_total
        )
        object_weighted[key] = float(
            np.mean([result["coverage"][key] for result in object_results])
        )

    return {
        "n_objects": n_objects,
        "n_heldout_total": n_total,
        "n_train_total": train_total,
        "ks_pit_obs_weighted": ks_pit_obs_weighted,
        "ks_pit_object_weighted": ks_pit_object_weighted,
        "observation_weighted": observation_weighted,
        "object_weighted": object_weighted,
    }


def _scalar_from_result_value(value):
    """
    Extract a scalar value from a result field that may be a scalar, a 0-dim array, or a 1-element array. Return None for empty arrays.
    """
    arr = np.asarray(value, dtype=object)
    if arr.ndim == 0:
        return arr.item()
    if len(arr) == 0:
        return None
    return arr.reshape(-1)[0]


def _hashable_object_id(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return tuple(value.reshape(-1).tolist())
    if isinstance(value, list):
        return tuple(value)
    return value


def _require_valid_class_label(result):
    class_label = result.get("obj_type", result.get("class", None))
    class_label = _scalar_from_result_value(class_label)
    if class_label is None or pd.isna(class_label):
        object_id = _scalar_from_result_value(result.get("object_id", None))
        raise ValueError(f"Object {object_id!r} is missing a valid class label.")
    class_label = str(class_label)
    if class_label == "":
        object_id = _scalar_from_result_value(result.get("object_id", None))
        raise ValueError(f"Object {object_id!r} is missing a valid class label.")
    return class_label


def _object_metric_row(result: Mapping[str, Any]) -> dict[str, Any]:
    n_test_object = int(result.get("n_heldout", len(result.get("y_true", []))))
    if n_test_object < 1:
        object_id = _scalar_from_result_value(result.get("object_id", None))
        raise ValueError(f"Object {object_id!r} must have at least one held-out prediction.")

    y_true = np.asarray(result["y_true"], dtype=float)
    y_pred = np.asarray(result["y_pred"], dtype=float)
    y_std = np.maximum(np.asarray(result["y_std"], dtype=float), PREDICTIVE_STD_EPSILON)
    if not np.all(y_std > 0):
        raise AssertionError("sigma_pred must be positive after clipping.")

    z, pit = compute_pit_values(y_true, y_pred, y_std)
    abs_z = np.abs(z)
    squared_errors = (y_true - y_pred) ** 2
    rmse_object = float(result.get("rmse", np.sqrt(np.mean(squared_errors))))
    nrmse_object = result.get("nrmse", np.nan)
    nlpd_object = float(result.get(
        "mean_nlpd",
        np.mean(negative_log_predictive_density(
            y_true,
            y_pred,
            np.maximum(y_std ** 2, PREDICTIVE_STD_EPSILON),
        )),
    ))
    crps_object = float(result.get(
        "mean_crps",
        np.mean(gaussian_crps(y_true, y_pred, y_std)),
    ))

    coverage = result.get("coverage", {})
    n_target_train_object = result.get("n_target_train_object", result.get("n_train", np.nan))
    object_flux_scale = result.get("object_flux_scale", np.nan)
    object_flux_scale = float(object_flux_scale) if object_flux_scale is not None else np.nan
    ncrps_object = (
        float(crps_object / object_flux_scale)
        if np.isfinite(object_flux_scale) and object_flux_scale > 0
        else np.nan
    )

    return {
        "object_id": _scalar_from_result_value(result.get("object_id", None)),
        "class": _require_valid_class_label(result),
        "rmse_object": rmse_object,
        "nrmse_object": float(nrmse_object) if nrmse_object is not None else np.nan,
        "nlpd_object": nlpd_object,
        "crps_object": crps_object,
        "ncrps_object": ncrps_object,
        "coverage_1sigma_object": float(coverage.get("coverage_1sigma", np.mean(abs_z <= 1))),
        "coverage_2sigma_object": float(coverage.get("coverage_2sigma", np.mean(abs_z <= 2))),
        "coverage_3sigma_object": float(coverage.get("coverage_3sigma", np.mean(abs_z <= 3))),
        "z_mean_object": float(np.mean(z)),
        "z_std_object": float(np.std(z)),
        "ks_pit_object": float(result.get("ks_pit_object", compute_ks_pit(pit))),
        "n_pit_object": int(result.get("n_pit_object", np.sum(np.isfinite(pit)))),
        "n_test_object": n_test_object,
        "object_flux_scale": object_flux_scale,
        # n_target_train statistics describe target-band data availability for the single-band GP.
        "n_target_train_object": int(n_target_train_object) if n_target_train_object is not None else np.nan,
    }


def single_band_gp_object_metric_table(
        object_results: Sequence[Mapping[str, Any]],
        min_object_flux_scale: float | None = 1e-6,
) -> pd.DataFrame:
    """
    Build per-object single-band GP held-out metrics before class aggregation.
    """
    object_table, _ = _single_band_gp_object_and_skipped_tables(
        object_results,
        min_object_flux_scale=min_object_flux_scale,
    )
    return object_table


def _single_band_gp_object_and_skipped_tables(
        object_results: Sequence[Mapping[str, Any]],
        min_object_flux_scale: float | None = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build kept and skipped per-object tables for class aggregation.
    """
    if len(object_results) == 0:
        raise ValueError("object_results must contain at least one result.")

    rows = []
    skipped_rows = []
    for result in object_results:
        row = _object_metric_row(result)
        scale = row["object_flux_scale"]
        if (
                min_object_flux_scale is not None
                and (not np.isfinite(scale) or scale < min_object_flux_scale)
        ):
            skipped_rows.append({
                "object_id": row["object_id"],
                "class": row["class"],
                "object_flux_scale": scale,
                "nrmse_object": row["nrmse_object"],
                "skip_reason": f"object_flux_scale_lt_{min_object_flux_scale:g}",
            })
            continue
        rows.append(row)

    object_table = pd.DataFrame(rows)
    skipped_table = pd.DataFrame(skipped_rows)
    return object_table, skipped_table


def _mean_std(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    return float(np.mean(values)), float(np.std(values))


def summarize_single_band_gp_class_metrics(
        object_results: Sequence[Mapping[str, Any]],
        min_test_for_zstd: int = 5,
        sparse_threshold: int = 10,
        min_object_flux_scale: float | None = 1e-6,
        output_path: str | Path | None = "single_band_gp_class_summary.csv",
        print_table: bool = True,
        plot_pit: bool = False,
        pit_output_dir: str | Path | None = None,
        pit_bins: int = 20,
) -> pd.DataFrame:
    """
    Summarize single-band GP evaluation metrics by object class.

    Class-level metrics are object-weighted: metrics are computed per object
    first, then summarized across objects within each class.
    """
    object_table, skipped_table = _single_band_gp_object_and_skipped_tables(
        object_results,
        min_object_flux_scale=min_object_flux_scale,
    )
    if object_table.empty:
        raise ValueError(
            "No objects remain after filtering on object_flux_scale. "
            f"min_object_flux_scale={min_object_flux_scale:g}"
        )
    if plot_pit and pit_output_dir is not None:
        pit_output_dir = Path(pit_output_dir)
        pit_output_dir.mkdir(parents=True, exist_ok=True)
    skipped_counts: dict[Any, int] = {}
    if not skipped_table.empty:
        skipped_classes = cast(pd.Series, skipped_table["class"])
        for skipped_class in skipped_classes:
            skipped_counts[skipped_class] = skipped_counts.get(skipped_class, 0) + 1
    rows = []

    performance_metrics = [
        ("rmse", "rmse_object"),
        ("nrmse", "nrmse_object"),
        ("nlpd", "nlpd_object"),
        ("crps", "crps_object"),
        ("ncrps", "ncrps_object"),
        ("coverage_1sigma", "coverage_1sigma_object"),
        ("coverage_2sigma", "coverage_2sigma_object"),
        ("coverage_3sigma", "coverage_3sigma_object"),
        ("z_mean", "z_mean_object"),
        ("z_std", "z_std_object"),
        ("ks_pit", "ks_pit_object"),
    ]

    class_series = cast(pd.Series, object_table["class"])
    for class_label in sorted(pd.unique(class_series), key=str):
        group = object_table.loc[class_series.eq(class_label), :].copy()
        n_objects = int(group.shape[0])
        if n_objects < 2:
            #raise AssertionError("Classes with very few objects should be included, not dropped.")
            continue

        n_target_train = np.asarray(group["n_target_train_object"], dtype=float)
        object_flux_scale = np.asarray(group["object_flux_scale"], dtype=float)
        sparse_mask = n_target_train < sparse_threshold
        row = {
            "class": class_label,
            "n_objects": n_objects,
            "n_objects_skipped_small_flux": int(skipped_counts.get(class_label, 0)),
            "min_object_flux_scale": np.nan if min_object_flux_scale is None else float(min_object_flux_scale),
            "n_target_train_mean": float(np.mean(n_target_train)),
            "n_target_train_std": float(np.std(n_target_train)),
            "n_target_train_median": float(np.median(n_target_train)),
            "n_target_train_min": int(np.min(n_target_train)),
            "n_target_train_max": int(np.max(n_target_train)),
            "n_target_train_p10": float(np.percentile(n_target_train, 10)),
            "n_target_train_p25": float(np.percentile(n_target_train, 25)),
            "n_target_train_p75": float(np.percentile(n_target_train, 75)),
            "n_target_train_p90": float(np.percentile(n_target_train, 90)),
            "n_objects_target_train_lt_5": int(np.sum(sparse_mask)),
            "frac_objects_target_train_lt_5": float(np.mean(sparse_mask)),
            "object_flux_scale_min": float(np.nanmin(object_flux_scale)),
            "object_flux_scale_p10": float(np.nanpercentile(object_flux_scale, 10)),
            "object_flux_scale_median": float(np.nanmedian(object_flux_scale)),
            "object_flux_scale_mean": float(np.nanmean(object_flux_scale)),
        }

        # The main performance metrics are summarized across all objects in the class
        for metric_name, column in performance_metrics:
            metric_mean, metric_std = _mean_std(group[column])
            row[f"{metric_name}_mean"] = metric_mean
            row[f"{metric_name}_std"] = metric_std

        if plot_pit:
            class_object_ids = set(group["object_id"].tolist())
            class_pit_arrays = []
            for result in object_results:
                object_id = _scalar_from_result_value(result.get("object_id", None))
                if object_id not in class_object_ids:
                    continue
                pit_values = result.get("pit_values", None)
                if pit_values is None:
                    _, pit_values = compute_pit_values(result["y_true"], result["y_pred"], result["y_std"])
                valid_pit = _valid_pit_values(pit_values)
                if len(valid_pit) > 0:
                    class_pit_arrays.append(valid_pit)
            class_pit = np.concatenate(class_pit_arrays) if class_pit_arrays else np.array([], dtype=float)
            safe_class = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(class_label))
            histogram_path = None
            reliability_path = None
            if pit_output_dir is not None:
                pit_dir = Path(pit_output_dir)
                histogram_path = pit_dir / f"pit_histogram_{safe_class}.png"
                reliability_path = pit_dir / f"pit_reliability_{safe_class}.png"
            plot_pit_histogram(
                class_pit,
                title=f"PIT histogram: {class_label}",
                bins=pit_bins,
                save_path=histogram_path,
            )
            plot_pit_reliability_curve(
                class_pit,
                title=f"PIT reliability: {class_label}",
                save_path=reliability_path,
            )

        # Within-object z_std is unstable for very small n_test_object, so the filtered columns are included for interpretation.
        n_test_series = cast(pd.Series, group["n_test_object"])
        zstd_group = group.loc[n_test_series.ge(min_test_for_zstd), :]
        row["n_objects_zstd_n_test_ge_5"] = int(zstd_group.shape[0])
        if zstd_group.empty:
            row["z_std_mean_n_test_ge_5"] = np.nan
            row["z_std_std_n_test_ge_5"] = np.nan
        else:
            zstd_mean, zstd_std = _mean_std(zstd_group["z_std_object"])
            row["z_std_mean_n_test_ge_5"] = zstd_mean
            row["z_std_std_n_test_ge_5"] = zstd_std

        rows.append(row)

    summary = pd.DataFrame(rows)
    if output_path is not None:
        summary.to_csv(output_path, index=False)
    if print_table:
        print(summary)
    return summary


def _get_mogp_prediction_function():
    from MOGP_model import predict_mogp_observation_distribution

    return predict_mogp_observation_distribution


def _band_equal_mask(values, band):
    from MOGP_model import _band_equal_mask as mogp_band_equal_mask

    return mogp_band_equal_mask(values, band)


def _assert_mogp_z_score_invariance(y_norm, mean_norm, std_norm, y_raw, mean_raw, std_raw, scale):
    z_norm = (np.asarray(y_norm) - np.asarray(mean_norm)) / np.maximum(std_norm, PREDICTIVE_STD_EPSILON)
    z_raw = (np.asarray(y_raw) - np.asarray(mean_raw)) / np.maximum(std_raw, scale * PREDICTIVE_STD_EPSILON)
    if not np.allclose(z_norm, z_raw, rtol=1e-5, atol=1e-5):
        max_diff = float(np.max(np.abs(z_norm - z_raw)))
        raise AssertionError(f"MOGP raw and normalized z-scores differ: max_abs_diff={max_diff:g}")


def evaluate_mogp_heldout_metrics(
        gp,
        heldout_data,
        train_data=None,
        object_data=None,
        object_flux_scale=None,
        nrmse_quantile=0.95,
        nrmse_epsilon=1e-8,
        coverage_sigmas=(1.0, 2.0, 3.0),
        include_yerr=True,
        yerr_scale=1.0,
        noise_floor=0.0,
        evaluate_raw_metrics=True,
        assert_z_invariance=True,
):
    predict_mogp_observation_distribution = _get_mogp_prediction_function()
    mean_norm, std_norm, variance_norm = predict_mogp_observation_distribution(
        gp,
        heldout_data,
        include_yerr=include_yerr,
        yerr_scale=yerr_scale,
        noise_floor=noise_floor,
        return_raw_flux=False,
    )
    y_norm = np.asarray(heldout_data["y"])
    errors_norm = y_norm - mean_norm

    coverage = {}
    coverage_counts = {}
    for sigma in coverage_sigmas:
        covered = np.abs(errors_norm) <= sigma * std_norm
        key = f"coverage_{sigma:g}sigma"
        coverage[key] = float(np.mean(covered))
        coverage_counts[key] = int(np.sum(covered))

    scale = float(heldout_data["flux_scale"])
    background = np.asarray(heldout_data.get("background_flux", 0.0), dtype=float)
    mean_raw, variance_raw = inverse_transform_predictions(mean_norm, variance_norm, scale, 0.0)
    mean_raw = mean_raw + background
    std_raw = np.sqrt(np.maximum(variance_raw, 0.0))
    y_raw = np.asarray(heldout_data["y_raw"])
    yerr_raw = np.asarray(heldout_data["yerr_raw"])

    if assert_z_invariance:
        _assert_mogp_z_score_invariance(
            y_norm,
            mean_norm,
            std_norm,
            y_raw,
            mean_raw,
            std_raw,
            scale,
        )

    if evaluate_raw_metrics:
        y_metric = y_raw
        mean_metric = mean_raw
        std_metric = std_raw
        variance_metric = variance_raw
        yerr_metric = yerr_raw
        metric_space = "raw"
    else:
        y_metric = y_norm
        mean_metric = mean_norm
        std_metric = std_norm
        variance_metric = variance_norm
        yerr_metric = np.asarray(heldout_data["yerr"])
        metric_space = "normalized"

    squared_errors = (y_metric - mean_metric) ** 2
    z_values, pit_values = compute_pit_values(y_metric, mean_metric, std_metric)
    ks_pit_object = compute_ks_pit(pit_values)
    n_pit_object = int(np.sum(np.isfinite(pit_values)))
    rmse = float(np.sqrt(np.mean(squared_errors)))
    if object_flux_scale is None:
        object_flux_scale = object_level_empirical_flux_scale(
            train_data,
            heldout_data,
            example=object_data,
            q=nrmse_quantile,
            epsilon=nrmse_epsilon,
        )
    object_flux_scale = float(max(float(object_flux_scale), nrmse_epsilon))
    per_point_nlpd = negative_log_predictive_density(y_metric, mean_metric, variance_metric)
    per_point_crps = gaussian_crps(y_metric, mean_metric, std_metric)

    n_heldout = len(y_metric)
    if train_data is not None:
        train_t = np.asarray(train_data["t"])
        train_time_min = float(np.min(train_t))
        train_time_max = float(np.max(train_t))
        outside_train_range = (heldout_data["t"] < train_time_min) | (heldout_data["t"] > train_time_max)
        distance_to_train_range = np.maximum.reduce([
            train_time_min - heldout_data["t"],
            heldout_data["t"] - train_time_max,
            np.zeros_like(heldout_data["t"]),
        ])
        n_train = len(train_t)
        all_t = np.concatenate([train_data["t"], heldout_data["t"]])
        all_y = np.concatenate([train_data["y"], heldout_data["y"]])
        peak_time = all_t[np.argmax(np.abs(all_y))]
    else:
        train_time_min = np.nan
        train_time_max = np.nan
        outside_train_range = np.full(n_heldout, False)
        distance_to_train_range = np.full(n_heldout, np.nan)
        n_train = None
        peak_time = heldout_data["t"][np.argmax(np.abs(y_norm))]
    near_peak = np.abs(heldout_data["t"] - peak_time) <= 0.25

    return {
        "n_heldout": n_heldout,
        "n_train": n_train,
        "train_time_min": train_time_min,
        "train_time_max": train_time_max,
        "metric_space": metric_space,
        "mean_nlpd": float(np.mean(per_point_nlpd)),
        "total_nlpd": float(np.sum(per_point_nlpd)),
        "mean_crps": float(np.mean(per_point_crps)),
        "total_crps": float(np.sum(per_point_crps)),
        "rmse": rmse,
        "nrmse": float(rmse / object_flux_scale),
        "ncrps": float(np.mean(per_point_crps) / object_flux_scale),
        "sse": float(np.sum(squared_errors)),
        "object_flux_scale": object_flux_scale,
        "nrmse_quantile": float(nrmse_quantile),
        "coverage": coverage,
        "coverage_counts": coverage_counts,
        "per_point_nlpd": per_point_nlpd,
        "per_point_crps": per_point_crps,
        "squared_errors": squared_errors,
        "z_values": z_values,
        "pit_values": pit_values,
        "ks_pit_object": ks_pit_object,
        "n_pit_object": n_pit_object,
        "y_true": y_metric,
        "y_pred": mean_metric,
        "y_std": std_metric,
        "yerr": yerr_metric,
        "y_true_norm": y_norm,
        "y_pred_norm": mean_norm,
        "y_std_norm": std_norm,
        "predictive_variance_norm": variance_norm,
        "y_true_raw": y_raw,
        "y_pred_raw": mean_raw,
        "y_std_raw": std_raw,
        "yerr_raw": yerr_raw,
        "time": np.asarray(heldout_data["t"]),
        "X_test": np.asarray(heldout_data["X"]).reshape(n_heldout, -1),
        "object_id": np.repeat(heldout_data.get("obj_id", None), n_heldout),
        "band": np.asarray(heldout_data["band"], dtype=object),
        "outside_train_range": outside_train_range,
        "distance_to_train_range": distance_to_train_range,
        "near_peak": near_peak,
        "predictive_variance": variance_metric,
        "flux_scale": scale,
        "background_flux": background,
    }


"""--------------------ablation study functions--------------------"""

def _metrics_row_from_result(model_name, metrics, train_data, target_band, heldout_indices, notes=None):
    """
    It is used by run_target_band_ablation_study() in MOGP_model.py 
    to build the rows comparing single-band GP, target-only MOGP, real-wavelength MOGP, shuffled controls, etc.
    """
    y_true = np.asarray(metrics["y_true"], dtype=float)
    y_pred = np.asarray(metrics["y_pred"], dtype=float)
    y_std = np.maximum(np.asarray(metrics["y_std"], dtype=float), PREDICTIVE_STD_EPSILON)
    z, pit = compute_pit_values(y_true, y_pred, y_std)
    train_band = np.asarray(train_data["band"], dtype=object)
    if train_band.ndim == 0:
        train_band = np.repeat(train_band.item(), len(train_data["y"]))
    target_mask = _band_equal_mask(train_band, target_band)
    coverage = metrics["coverage"]
    object_flux_scale = float(metrics["object_flux_scale"])
    mean_crps = float(metrics["mean_crps"])
    ncrps = float(metrics.get("ncrps", mean_crps / max(object_flux_scale, PREDICTIVE_STD_EPSILON)))
    return {
        "model": model_name,
        "object_id": metrics["object_id"][0] if len(metrics["object_id"]) else None,
        "target_band": target_band,
        "n_train_target_band": int(np.sum(target_mask)),
        "n_train_other_bands": int(len(train_band) - np.sum(target_mask)),
        "n_heldout_target_band": int(metrics["n_heldout"]),
        "heldout_indices": np.asarray(heldout_indices, dtype=int),
        "rmse": float(metrics["rmse"]),
        "nrmse": float(metrics["nrmse"]),
        "crps": mean_crps,
        "ncrps": ncrps,
        "object_flux_scale": object_flux_scale,
        "nlpd": float(metrics["mean_nlpd"]),
        "coverage_1sigma": float(coverage["coverage_1sigma"]),
        "coverage_2sigma": float(coverage["coverage_2sigma"]),
        "coverage_3sigma": float(coverage["coverage_3sigma"]),
        "z_score_mean": float(np.mean(z)),
        "z_score_std": float(np.std(z)),
        "ks_pit": float(metrics.get("ks_pit_object", compute_ks_pit(pit))),
        "z_scores": z,
        "pit": pit,
        "y_true": y_true,
        "y_pred": y_pred,
        "notes": notes,
    }


def summarize_metrics_by_band(object_results):
    """
    Aggregate per-observation metrics into per-band summaries across objects.
    """
    rows = []
    for result in object_results:
        for band in np.unique(result["band"]):
            mask = np.asarray(result["band"]) == band
            y = np.asarray(result["y_true"])[mask]
            pred = np.asarray(result["y_pred"])[mask]
            std = np.maximum(np.asarray(result["y_std"])[mask], 1e-12)
            per_point_nlpd = negative_log_predictive_density(y, pred, std ** 2)
            per_point_crps = gaussian_crps(y, pred, std)
            z, pit = compute_pit_values(y, pred, std)
            rows.append({
                "band": band,
                "n_heldout": int(np.sum(mask)),
                "total_nlpd": float(np.sum(per_point_nlpd)),
                "total_crps": float(np.sum(per_point_crps)),
                "sse": float(np.sum((y - pred) ** 2)),
                "object_flux_scale": float(result.get("object_flux_scale", np.nan)),
                "coverage_1sigma_count": int(np.sum(np.abs(z) <= 1)),
                "coverage_2sigma_count": int(np.sum(np.abs(z) <= 2)),
                "coverage_3sigma_count": int(np.sum(np.abs(z) <= 3)),
                "pit_values": pit,
            })

    summary = {}
    for band in sorted({row["band"] for row in rows}, key=str):
        band_rows = [row for row in rows if row["band"] == band]
        n = sum(row["n_heldout"] for row in band_rows)
        band_nrmse_values = [
            np.sqrt(row["sse"] / row["n_heldout"]) / max(row["object_flux_scale"], 1e-8)
            for row in band_rows
            if np.isfinite(row["object_flux_scale"])
        ]
        band_ncrps_values = [
            (row["total_crps"] / row["n_heldout"]) / max(row["object_flux_scale"], 1e-8)
            for row in band_rows
            if np.isfinite(row["object_flux_scale"])
        ]
        pit_by_object = {
            row_idx: row["pit_values"]
            for row_idx, row in enumerate(band_rows)
        }
        pooled_pit_arrays = [
            _valid_pit_values(row["pit_values"])
            for row in band_rows
            if len(_valid_pit_values(row["pit_values"])) > 0
        ]
        pooled_pit = np.concatenate(pooled_pit_arrays) if pooled_pit_arrays else np.array([], dtype=float)
        summary[band] = {
            "n_heldout": int(n),
            "nlpd": float(sum(row["total_nlpd"] for row in band_rows) / n),
            "crps": float(sum(row["total_crps"] for row in band_rows) / n),
            "rmse": float(np.sqrt(sum(row["sse"] for row in band_rows) / n)),
            "nrmse": float(np.mean(band_nrmse_values)) if band_nrmse_values else np.nan,
            "ncrps": float(np.mean(band_ncrps_values)) if band_ncrps_values else np.nan,
            "coverage_1sigma": float(sum(row["coverage_1sigma_count"] for row in band_rows) / n),
            "coverage_2sigma": float(sum(row["coverage_2sigma_count"] for row in band_rows) / n),
            "coverage_3sigma": float(sum(row["coverage_3sigma_count"] for row in band_rows) / n),
            "ks_pit_obs_weighted": compute_ks_pit(pooled_pit),
            "ks_pit_object_weighted": compute_object_weighted_ks_pit(pit_by_object),
        }
    return summary


def _canonical_ablation_model_name(model):
    model = str(model)
    if model.startswith("mogp_shuffled_wavelength_control_seed"):
        return "mogp_shuffled_wavelength_control"
    return model


def collapse_shuffled_wavelength_controls(
        results_df,
        keep_seed_level=False,
):
    """
    Collapse shuffled wavelength seed rows to one object-band row per model.

    C vs shuffled controls tests whether real wavelength structure matters
    beyond generic cross-band information sharing.  Collapsing prevents one
    object with many shuffle seeds from being overrepresented in aggregates.
    """
    df = results_df.copy()
    if "model" not in df.columns:
        raise ValueError("results_df must contain a 'model' column.")

    df["model_original"] = df["model"]
    df["model"] = df["model"].map(_canonical_ablation_model_name)
    if keep_seed_level:
        return df

    group_cols = ["model", "object_id", "target_band"]
    missing = [col for col in group_cols if col not in df.columns]
    if missing:
        raise ValueError(f"results_df missing required columns for shuffled collapse: {missing}")

    numeric_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in {"shuffle_repeat"}
    ]
    first_cols = [
        col for col in df.columns
        if col not in set(group_cols + numeric_cols)
    ]
    agg_spec = {col: "mean" for col in numeric_cols}
    agg_spec.update({col: "first" for col in first_cols})

    collapsed = df.groupby(group_cols, as_index=False, dropna=False).agg(agg_spec)
    return collapsed


def _prepare_ablation_results_df(
        results_df,
        collapse_shuffled=True,
        keep_seed_level=False,
):
    df = results_df.copy()
    required = _raw_ablation_required_columns()
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "results_df is missing required raw ablation columns: "
            f"{missing}. Pass the raw rows dataframe, for example "
            "`df = pd.DataFrame(all_ablation_rows); save_gp_ablation_reports(df)`. "
            "If you already called aggregate_gp_ablation_results(df), that aggregate "
            "can only be saved as a precomputed aggregate."
        )

    df = df[pd.to_numeric(df["n_heldout_target_band"], errors="coerce") > 0].copy()
    if len(df) == 0:
        raise ValueError("No rows remain after excluding n_heldout_target_band <= 0.")

    if collapse_shuffled:
        df = collapse_shuffled_wavelength_controls(df, keep_seed_level=keep_seed_level)

    return df


def _raw_ablation_required_columns():
    return {
        "model",
        "object_id",
        "target_band",
        "n_heldout_target_band",
        "rmse",
        "nrmse",
        "nlpd",
        "crps",
        "ncrps",
        "coverage_1sigma",
        "coverage_2sigma",
        "coverage_3sigma",
        "z_score_mean",
        "z_score_std",
        "ks_pit",
    }


def _looks_like_precomputed_ablation_aggregate(df):
    aggregate_markers = {
        "model",
        "n_object_band_cases",
        "n_total_heldout",
        "rmse_obs_weighted",
        "nrmse_obs_weighted",
        "nlpd_obs_weighted",
    }
    return aggregate_markers.issubset(df.columns)


def _standard_error(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= 1:
        return np.nan
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.nan
    return float(np.sum(weights[valid] * values[valid]) / np.sum(weights[valid]))


def _aggregate_residual_level_if_available(group):
    """
    Prefer residual-level arrays for z/RMSE when the dataframe carries them.
    Expected optional columns are y_true/y_pred and/or z_scores.
    """
    y_true_values = []
    y_pred_values = []
    z_values = []
    if {"y_true", "y_pred"}.issubset(group.columns):
        for _, row in group.iterrows():
            y_true = row.get("y_true")
            y_pred = row.get("y_pred")
            if y_true is not None and y_pred is not None:
                y_true_values.append(np.asarray(y_true, dtype=float).reshape(-1))
                y_pred_values.append(np.asarray(y_pred, dtype=float).reshape(-1))
    if "z_scores" in group.columns:
        for _, row in group.iterrows():
            z = row.get("z_scores")
            if z is not None:
                z_values.append(np.asarray(z, dtype=float).reshape(-1))
    pit_values = []
    if "pit" in group.columns:
        for _, row in group.iterrows():
            pit = row.get("pit")
            if pit is not None:
                pit_values.append(np.asarray(pit, dtype=float).reshape(-1))

    out = {}
    if len(y_true_values) > 0:
        y_true_all = np.concatenate(y_true_values)
        y_pred_all = np.concatenate(y_pred_values)
        out["rmse_obs_weighted"] = float(np.sqrt(np.mean((y_true_all - y_pred_all) ** 2)))
    if len(z_values) > 0:
        z_all = np.concatenate(z_values)
        out["z_score_mean_obs_weighted"] = float(np.mean(z_all))
        out["z_score_std_obs_weighted"] = float(np.std(z_all))
    if len(pit_values) > 0:
        out["ks_pit_obs_weighted"] = compute_ks_pit(np.concatenate(pit_values))
    return out


def aggregate_gp_ablation_results(
        results_df,
        group_cols=None,
        residual_level_available=False,
        collapse_shuffled=True,
        keep_seed_level=False,
        min_cases_warn=5,
):
    """
    Aggregate GP ablation rows using observation- and object-weighted summaries.

    Observation-weighted metrics describe performance over all held-out
    observations.  Object-weighted metrics describe performance for a typical
    object-band case and prevent high-cadence objects from dominating.
    """
    if group_cols is None:
        group_cols = ["model"]
    df = _prepare_ablation_results_df(
        results_df,
        collapse_shuffled=collapse_shuffled,
        keep_seed_level=keep_seed_level,
    )

    rows = []
    for group_key, group in df.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = {col: value for col, value in zip(group_cols, group_key)}

        n = group["n_heldout_target_band"].astype(float).to_numpy()
        row["n_object_band_cases"] = int(len(group))
        row["n_unique_objects"] = int(group["object_id"].nunique())
        row["n_total_heldout"] = int(np.sum(n))

        if row["n_object_band_cases"] < min_cases_warn:
            warnings.warn(
                f"Group {row} has only {row['n_object_band_cases']} object-band cases; "
                "do not treat this as global evidence.",
                RuntimeWarning,
            )

        row["rmse_obs_weighted"] = float(
            np.sqrt(np.sum(n * group["rmse"].astype(float).to_numpy() ** 2) / np.sum(n))
        )
        row["nrmse_obs_weighted"] = float(
            np.sqrt(np.sum(n * group["nrmse"].astype(float).to_numpy() ** 2) / np.sum(n))
        )
        row["nlpd_obs_weighted"] = _weighted_mean(group["nlpd"], n)
        row["crps_obs_weighted"] = _weighted_mean(group["crps"], n)
        row["ncrps_obs_weighted"] = _weighted_mean(group["ncrps"], n)
        if "pit" in group.columns:
            pit_arrays = [
                _valid_pit_values(row.get("pit"))
                for _, row in group.iterrows()
                if row.get("pit") is not None and len(_valid_pit_values(row.get("pit"))) > 0
            ]
            pooled_pit = np.concatenate(pit_arrays) if pit_arrays else np.array([], dtype=float)
            row["ks_pit_obs_weighted"] = compute_ks_pit(pooled_pit)
            row["ks_pit_object_weighted"] = compute_object_weighted_ks_pit(
                {idx: pit_values for idx, pit_values in enumerate(pit_arrays)}
            )
        else:
            row["ks_pit_obs_weighted"] = _weighted_mean(group["ks_pit"], n)
            row["ks_pit_object_weighted"] = float(np.mean(group["ks_pit"].astype(float).to_numpy()))
        for cov in ("coverage_1sigma", "coverage_2sigma", "coverage_3sigma"):
            row[f"{cov}_obs_weighted"] = _weighted_mean(group[cov], n)

        z_mean_obs = _weighted_mean(group["z_score_mean"], n)
        z_second = _weighted_mean(
            group["z_score_std"].astype(float) ** 2 + group["z_score_mean"].astype(float) ** 2,
            n,
        )
        row["z_score_mean_obs_weighted"] = z_mean_obs
        row["z_score_std_obs_weighted"] = float(
            np.sqrt(max(z_second - z_mean_obs ** 2, 0.0))
        )

        if residual_level_available:
            row.update(_aggregate_residual_level_if_available(group))

        object_metrics = {
            "rmse": "rmse_object_weighted",
            "nrmse": "nrmse_object_weighted",
            "nlpd": "nlpd_object_weighted",
            "crps": "crps_object_weighted",
            "ncrps": "ncrps_object_weighted",
            "coverage_1sigma": "coverage_1sigma_object_weighted",
            "coverage_2sigma": "coverage_2sigma_object_weighted",
            "coverage_3sigma": "coverage_3sigma_object_weighted",
            "z_score_mean": "z_score_mean_object_weighted",
            "z_score_std": "z_score_std_object_weighted",
        }
        for src, dest in object_metrics.items():
            values = group[src].astype(float).to_numpy()
            row[dest] = float(np.mean(values))
            # compute object-weighted standard deviation and standard error for the metric
            row[f"{src}_object_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            row[f"{src}_object_se"] = _standard_error(values)
        ks_values = group["ks_pit"].astype(float).to_numpy()
        row["ks_pit_object_std"] = float(np.std(ks_values, ddof=1)) if len(ks_values) > 1 else np.nan
        row["ks_pit_object_se"] = _standard_error(ks_values)

        rows.append(row)

    return pd.DataFrame(rows)


def compare_models_aggregated(
        results_df,
        model_pairs=None,
        collapse_shuffled=True,
        keep_seed_level=False,
):
    """
    Paired model comparisons matched exactly by object_id and target_band.

    C vs D, mogp_real_wavelength vs mogp_independent_band_control, is the key
    comparison for whether cross-band covariance helps beyond merely adding
    more data points.  A/B matching is a preprocessing sanity check.
    """
    if model_pairs is None:
        model_pairs = [
            ("mogp_real_wavelength", "single_band_gp"),
            ("mogp_real_wavelength", "mogp_independent_band_control"),
            ("mogp_real_wavelength", "mogp_shuffled_wavelength_control"),
            ("same_total_train_budget_existing", "single_band_gp"),
            ("mogp_target_only", "single_band_gp"),
        ]
    df = _prepare_ablation_results_df(
        results_df,
        collapse_shuffled=collapse_shuffled,
        keep_seed_level=keep_seed_level,
    )

    out_rows = []
    for model_a, model_b in model_pairs:
        a = df[df["model"] == model_a].copy()
        b = df[df["model"] == model_b].copy()
        paired = a.merge(
            b,
            on=["object_id", "target_band"],
            suffixes=("_a", "_b"),
            how="inner",
        )
        if len(paired) == 0:
            warnings.warn(
                f"No matched object-band pairs for {model_a} vs {model_b}.",
                RuntimeWarning,
            )
            continue

        delta_rmse = paired["rmse_a"].astype(float) - paired["rmse_b"].astype(float)
        delta_nrmse = paired["nrmse_a"].astype(float) - paired["nrmse_b"].astype(float)
        delta_nlpd = paired["nlpd_a"].astype(float) - paired["nlpd_b"].astype(float)
        delta_crps = paired["crps_a"].astype(float) - paired["crps_b"].astype(float)
        delta_ncrps = paired["ncrps_a"].astype(float) - paired["ncrps_b"].astype(float)
        delta_cov1 = paired["coverage_1sigma_a"].astype(float) - paired["coverage_1sigma_b"].astype(float)
        delta_z_std = paired["z_score_std_a"].astype(float) - paired["z_score_std_b"].astype(float)
        delta_ks_pit = paired["ks_pit_a"].astype(float) - paired["ks_pit_b"].astype(float)
        better_z = (
            np.abs(paired["z_score_std_a"].astype(float) - 1.0)
            < np.abs(paired["z_score_std_b"].astype(float) - 1.0)
        )

        row = {
            "model_a": model_a,
            "model_b": model_b,
            "n_matched_object_band_pairs": int(len(paired)),
            "delta_rmse_mean": float(np.mean(delta_rmse)),
            "delta_rmse_median": float(np.median(delta_rmse)),
            "delta_rmse_se": _standard_error(delta_rmse),
            "delta_nrmse_mean": float(np.mean(delta_nrmse)),
            "delta_nrmse_median": float(np.median(delta_nrmse)),
            "delta_nrmse_se": _standard_error(delta_nrmse),
            "delta_nlpd_mean": float(np.mean(delta_nlpd)),
            "delta_nlpd_median": float(np.median(delta_nlpd)),
            "delta_nlpd_se": _standard_error(delta_nlpd),
            "delta_crps_mean": float(np.mean(delta_crps)),
            "delta_crps_median": float(np.median(delta_crps)),
            "delta_crps_se": _standard_error(delta_crps),
            "delta_ncrps_mean": float(np.mean(delta_ncrps)),
            "delta_ncrps_median": float(np.median(delta_ncrps)),
            "delta_ncrps_se": _standard_error(delta_ncrps),
            "delta_coverage_1sigma_mean": float(np.mean(delta_cov1)),
            "delta_coverage_1sigma_median": float(np.median(delta_cov1)),
            "delta_coverage_1sigma_se": _standard_error(delta_cov1),
            "delta_z_score_std_mean": float(np.mean(delta_z_std)),
            "delta_z_score_std_median": float(np.median(delta_z_std)),
            "delta_z_score_std_se": _standard_error(delta_z_std),
            "delta_ks_pit_mean": float(np.mean(delta_ks_pit)),
            "delta_ks_pit_median": float(np.median(delta_ks_pit)),
            "delta_ks_pit_se": _standard_error(delta_ks_pit),
            "fraction_improved_rmse": float(np.mean(delta_rmse < 0)),
            "fraction_improved_nrmse": float(np.mean(delta_nrmse < 0)),
            "fraction_improved_nlpd": float(np.mean(delta_nlpd < 0)),
            "fraction_improved_crps": float(np.mean(delta_crps < 0)),
            "fraction_improved_ncrps": float(np.mean(delta_ncrps < 0)),
            "fraction_improved_ks_pit": float(np.mean(delta_ks_pit < 0)),
            "fraction_better_calibrated_z_std": float(np.mean(better_z)),
        }
        out_rows.append(row)

    return pd.DataFrame(out_rows)


def save_gp_ablation_reports(
        results_df,
        output_dir=".",
        collapse_shuffled=True,
        keep_seed_level=False,
):
    """
    Save standard ablation aggregate CSV reports.

    Pass the raw row-level dataframe from run_target_band_ablation_study for the
    full report set. If an already aggregated dataframe is passed, only that
    precomputed aggregate can be saved because object-level pairing information
    has already been collapsed away.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df = results_df.copy()

    raw_missing = sorted(_raw_ablation_required_columns() - set(results_df.columns))
    if raw_missing and _looks_like_precomputed_ablation_aggregate(results_df):
        paths = {
            "by_model": output_dir / "gp_ablation_aggregated_by_model.csv",
        }
        results_df.to_csv(paths["by_model"], index=False)
        return {
            "aggregated_by_model": results_df,
            "aggregated_by_model_and_band": None,
            "paired_model_comparisons": None,
            "paths": paths,
            "notes": (
                "Input was already aggregated, so only the provided aggregate "
                "was saved. Pass raw ablation rows to save gp_ablation_rows, "
                "by-model-and-band, and paired-comparison reports."
            ),
        }

    by_model = aggregate_gp_ablation_results(
        results_df,
        group_cols=["model"],
        collapse_shuffled=collapse_shuffled,
        keep_seed_level=keep_seed_level,
    )
    by_model_band = aggregate_gp_ablation_results(
        results_df,
        group_cols=["model", "target_band"],
        collapse_shuffled=collapse_shuffled,
        keep_seed_level=keep_seed_level,
    )
    comparisons = compare_models_aggregated(
        results_df,
        collapse_shuffled=collapse_shuffled,
        keep_seed_level=keep_seed_level,
    )

    paths = {
        "rows": output_dir / "gp_ablation_rows.csv",
        "by_model": output_dir / "gp_ablation_aggregated_by_model.csv",
        "by_model_and_band": output_dir / "gp_ablation_aggregated_by_model_and_band.csv",
        "paired_model_comparisons": output_dir / "gp_ablation_paired_model_comparisons.csv",
    }
    results_df.copy().to_csv(paths["rows"], index=False)
    by_model.to_csv(paths["by_model"], index=False)
    by_model_band.to_csv(paths["by_model_and_band"], index=False)
    comparisons.to_csv(paths["paired_model_comparisons"], index=False)

    return {
        "aggregated_by_model": by_model,
        "aggregated_by_model_and_band": by_model_band,
        "paired_model_comparisons": comparisons,
        "paths": paths,
    }


def plot_gp_ablation_summary(
        results_df,
        output_dir=None,
        collapse_shuffled=True,
):
    """
    Create optional summary plots for ablation aggregates.
    """
    import matplotlib.pyplot as plt

    output_dir = None if output_dir is None else Path(output_dir)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    agg = aggregate_gp_ablation_results(
        results_df,
        group_cols=["model"],
        collapse_shuffled=collapse_shuffled,
    )
    prepared = _prepare_ablation_results_df(results_df, collapse_shuffled=collapse_shuffled)

    figures = {}
    for metric, title, filename in [
        ("nlpd_obs_weighted", "Observation-weighted NLPD by model", "nlpd_observation_weighted_by_model.png"),
        ("nlpd_object_weighted", "Object-weighted NLPD by model", "nlpd_object_weighted_by_model.png"),
        ("crps_obs_weighted", "Observation-weighted CRPS by model", "crps_observation_weighted_by_model.png"),
        ("crps_object_weighted", "Object-weighted CRPS by model", "crps_object_weighted_by_model.png"),
        ("ncrps_obs_weighted", "Observation-weighted NCRPS by model", "ncrps_observation_weighted_by_model.png"),
        ("ncrps_object_weighted", "Object-weighted NCRPS by model", "ncrps_object_weighted_by_model.png"),
        ("nrmse_obs_weighted", "Observation-weighted NRMSE by model", "nrmse_observation_weighted_by_model.png"),
        ("nrmse_object_weighted", "Object-weighted NRMSE by model", "nrmse_object_weighted_by_model.png"),
        ("ks_pit_obs_weighted", "Observation-weighted KS-PIT by model", "ks_pit_observation_weighted_by_model.png"),
        ("ks_pit_object_weighted", "Object-weighted KS-PIT by model", "ks_pit_object_weighted_by_model.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        ordered = agg.sort_values(metric)
        ax.bar(ordered["model"], ordered[metric])
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        if output_dir is not None:
            fig.savefig(output_dir / filename, bbox_inches="tight")
        figures[metric] = fig

    paired = prepared.pivot_table(
        index=["object_id", "target_band"],
        columns="model",
        values="nlpd",
        aggfunc="mean",
    )
    if {"mogp_real_wavelength", "mogp_independent_band_control"}.issubset(paired.columns):
        delta = paired["mogp_real_wavelength"] - paired["mogp_independent_band_control"]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(delta.dropna(), bins=20, alpha=0.8)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title("Paired delta NLPD: real wavelength - independent-band control")
        ax.set_xlabel("Delta NLPD")
        ax.set_ylabel("Object-band count")
        fig.tight_layout()
        if output_dir is not None:
            fig.savefig(output_dir / "delta_nlpd_real_vs_independent.png", bbox_inches="tight")
        figures["delta_nlpd_real_vs_independent"] = fig

    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = agg.sort_values("z_score_std_object_weighted")
    ax.bar(ordered["model"], ordered["z_score_std_object_weighted"])
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="ideal")
    ax.set_ylabel("Object-weighted z-score std")
    ax.set_title("z-score std by model")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    if output_dir is not None:
        fig.savefig(output_dir / "z_score_std_by_model.png", bbox_inches="tight")
    figures["z_score_std_by_model"] = fig

    return figures
