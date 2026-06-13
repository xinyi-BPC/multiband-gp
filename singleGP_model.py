from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern
import numpy as np
import pandas as pd
from scipy.special import ndtr

PREDICTIVE_STD_EPSILON = 1e-12

def fit_basic_gp(
        data,
        kernel_type="matern",
        length_scale=0.3,
        length_scale_bounds=(0.05, 5.0),
        constant_value=1.0,
        constant_value_bounds=(1e-2, 1e2),
        yerr_scale=1.0,
        noise_floor=0.0,
        jitter=1e-8,
        n_restarts_optimizer=5,
        random_state=0,
        print_kernel=False,
):
    X = data['X']
    y = data['y']

    if kernel_type == "rbf":
        kernel = ConstantKernel(
            constant_value,
            constant_value_bounds,
        ) * RBF(
            length_scale=length_scale,
            length_scale_bounds=length_scale_bounds,
        )
    elif kernel_type == "matern":
        kernel = ConstantKernel(
            constant_value,
            constant_value_bounds,
        ) * Matern(
            length_scale=length_scale,
            length_scale_bounds=length_scale_bounds,
            nu=1.5,
        )
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")
    
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=(yerr_scale * data["yerr"])**2 + noise_floor**2 + jitter,
        normalize_y=False,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=random_state,
    )

    gp.fit(X, y)
    if print_kernel:
        print(f"MOGP learned kernel: {gp.kernel_}")
    
    return gp


def predict_observation_distribution(
        gp,
        data,
        include_yerr=True,
        yerr_scale=1.0,
        noise_floor=0.0,
        extra_noise=None,
        return_raw_flux=False,
):
    """
    Predict the distribution for observed flux values at data['X'].

    sklearn's GP predictive std is for the latent function. For held-out
    observations, add the test measurement error and any modeled observation noise.
    """
    mean, latent_std = gp.predict(data["X"], return_std=True)
    variance = latent_std ** 2
    if extra_noise is not None:
        noise_floor = extra_noise

    if include_yerr:
        variance = variance + (yerr_scale * np.asarray(data["yerr"])) ** 2
    if noise_floor is not None and noise_floor > 0:
        variance = variance + noise_floor ** 2

    if return_raw_flux:
        flux_scale = data["flux_scale"]
        background_flux = data.get("background_flux", 0.0)
        mean = mean * flux_scale + background_flux   # recover raw flux mean
        variance = variance * flux_scale ** 2      # recover raw flux variance

    return mean, np.sqrt(variance), variance


def inverse_transform_predictions(mu_norm, var_norm, scale, background):
    """
    Transform normalized predictive mean and variance back to raw flux units.
    """
    scale = float(scale)
    background = float(background)
    mu_raw = np.asarray(mu_norm) * scale + background
    var_raw = np.asarray(var_norm) * scale ** 2

    return mu_raw, var_raw


def _assert_z_score_invariance(
        errors_norm,
        std_norm,
        y_raw,
        mean_raw,
        std_raw,
        scale,
        floor_norm=1e-12,
):
    """
    Check that normalized and raw z-scores match under a linear flux transform.
    """
    scale = float(scale)
    raw_floor = floor_norm * scale
    z_norm = np.asarray(errors_norm) / np.maximum(np.asarray(std_norm), floor_norm)
    z_raw = (np.asarray(y_raw) - np.asarray(mean_raw)) / np.maximum(
        np.asarray(std_raw),
        raw_floor,
    )
    if not np.allclose(z_norm, z_raw, rtol=1e-5, atol=1e-5):
        max_diff = float(np.max(np.abs(z_norm - z_raw)))
        raise AssertionError(
            f"Normalized and raw z-scores do not match. max_abs_diff={max_diff:g}"
        )


def _raw_observation_arrays(data):
    """
    Extract raw observation arrays from the processed data.
    """
    scale = float(data["flux_scale"])
    background = float(data.get("background_flux", 0.0))
    y_norm = np.asarray(data["y"])
    yerr_norm = np.asarray(data["yerr"])

    y_raw = np.asarray(data.get("y_raw", y_norm * scale + background))
    yerr_raw = np.asarray(data.get("yerr_raw", yerr_norm * scale))

    return y_raw, yerr_raw, scale, background


def object_level_empirical_flux_scale(
        *datasets,
        example=None,
        bands=None,
        q=0.95,
        epsilon=1e-8,
):
    """
    Compute an object-level empirical raw-flux scale for evaluation only.

    NRMSE uses this all-observed-data object scale post hoc; it must not be
    used for training, splitting, fitting, hyperparameter optimization, or
    prediction.
    """
    q = float(q)
    if q > 1.0:
        q = q / 100.0
    q = min(max(q, 0.0), 1.0)
    epsilon = float(epsilon)
    flux_parts = []
    band_filter = None if bands is None else {str(band) for band in np.atleast_1d(bands)}

    # If an example dictionary is provided, it may contain a "lightcurve" key with raw flux arrays to include in the scale computation.
    if example is not None and "lightcurve" in example:
        lc = example["lightcurve"]
        flux = np.asarray(lc["flux"], dtype=float)
        valid = np.isfinite(flux)
        if "time" in lc:
            time = np.asarray(lc["time"], dtype=float)
            valid &= np.isfinite(time)
        if band_filter is not None and "band" in lc:
            band = np.asarray(lc["band"], dtype=object)
            valid &= np.asarray([str(value) in band_filter for value in band])
        flux_parts.append(flux[valid])

    # if no example is provided, or if the example does not contain a "lightcurve" key, then we fall back to looking for raw flux arrays in the provided datasets.
    for data in datasets:
        if data is None:
            continue
        if "y_raw" in data:
            flux = np.asarray(data["y_raw"], dtype=float)
        else:
            flux, _, _, _ = _raw_observation_arrays(data)
        valid = np.isfinite(flux)
        if band_filter is not None and "band" in data:
            band = np.asarray(data["band"], dtype=object)
            if band.ndim == 0:
                band = np.repeat(band.item(), len(flux))
            valid &= np.asarray([str(value) in band_filter for value in band])
        flux_parts.append(flux[valid])

    flux_parts = [part.reshape(-1) for part in flux_parts if len(part) > 0]
    if len(flux_parts) == 0:
        return 1.0  # default scale if no valid flux values are found

    abs_flux = np.abs(np.concatenate(flux_parts))
    abs_flux = abs_flux[np.isfinite(abs_flux)]
    if len(abs_flux) == 0:
        return float(epsilon)
    return float(max(np.quantile(abs_flux, q), epsilon))


def negative_log_predictive_density(y_true, mean, variance):
    """
    Compute pointwise negative log predictive density under a Gaussian.
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


def percentile_abs_difference(a, b, percentiles=(50, 90, 99)):
    """
    Summarize absolute differences between two same-shaped arrays.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"Array shapes must match: {a.shape} != {b.shape}")

    return np.percentile(np.abs(a - b), percentiles)


def compare_preprocessed_flux_arrays(
        data_no_filter,
        data_filter,
        percentiles=(50, 90, 99),
        print_summary=True,
):
    """
    Compare normalized and raw flux arrays from two preprocessing modes.

    The two data dictionaries should refer to the same object, band, and split.
    """
    y_no_filter_norm = np.asarray(data_no_filter["y"])
    y_filter_norm = np.asarray(data_filter["y"])
    y_no_filter_raw, yerr_no_filter_raw, _, _ = _raw_observation_arrays(data_no_filter)
    y_filter_raw, yerr_filter_raw, _, _ = _raw_observation_arrays(data_filter)

    summary = {
        "y_norm_abs_diff_percentiles": percentile_abs_difference(
            y_no_filter_norm,
            y_filter_norm,
            percentiles,
        ),
        "y_raw_abs_diff_percentiles": percentile_abs_difference(
            y_no_filter_raw,
            y_filter_raw,
            percentiles,
        ),
        "yerr_raw_abs_diff_percentiles": percentile_abs_difference(
            yerr_no_filter_raw,
            yerr_filter_raw,
            percentiles,
        ),
        "percentiles": tuple(percentiles),
    }
    if print_summary:
        print("abs(y_no_filter_norm - y_filter_norm):", summary["y_norm_abs_diff_percentiles"])
        print("abs(y_no_filter_raw - y_filter_raw):", summary["y_raw_abs_diff_percentiles"])
        print("abs(yerr_no_filter_raw - yerr_filter_raw):", summary["yerr_raw_abs_diff_percentiles"])

    return summary


def compare_raw_prediction_distributions(
        gp_no_filter,
        data_no_filter,
        gp_filter,
        data_filter,
        percentiles=(50, 90, 99),
        include_yerr=True,
        yerr_scale=1.0,
        noise_floor=0.0,
        extra_noise=None,
        print_summary=True,
):
    """
    Compare raw-space GP predictive means and standard deviations.
    """
    mu_no_filter_raw, std_no_filter_raw, var_no_filter_raw = predict_observation_distribution(
        gp_no_filter,
        data_no_filter,
        include_yerr=include_yerr,
        yerr_scale=yerr_scale,
        noise_floor=noise_floor,
        extra_noise=extra_noise,
        return_raw_flux=True,
    )
    mu_filter_raw, std_filter_raw, var_filter_raw = predict_observation_distribution(
        gp_filter,
        data_filter,
        include_yerr=include_yerr,
        yerr_scale=yerr_scale,
        noise_floor=noise_floor,
        extra_noise=extra_noise,
        return_raw_flux=True,
    )

    summary = {
        "mu_raw_abs_diff_percentiles": percentile_abs_difference(
            mu_no_filter_raw,
            mu_filter_raw,
            percentiles,
        ),
        "std_raw_abs_diff_percentiles": percentile_abs_difference(
            std_no_filter_raw,
            std_filter_raw,
            percentiles,
        ),
        "var_raw_abs_diff_percentiles": percentile_abs_difference(
            var_no_filter_raw,
            var_filter_raw,
            percentiles,
        ),
        "percentiles": tuple(percentiles),
    }
    if print_summary:
        print("abs(mu_no_filter_raw - mu_filter_raw):", summary["mu_raw_abs_diff_percentiles"])
        print("abs(std_no_filter_raw - std_filter_raw):", summary["std_raw_abs_diff_percentiles"])
        print("abs(var_no_filter_raw - var_filter_raw):", summary["var_raw_abs_diff_percentiles"])

    return summary


def RMSE(y_true, y_pred):
    """Compute root mean squared error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def evaluate_heldout_rmse(
        gp,
        heldout_data,
        evaluate_raw_metrics=True,
        train_data=None,
        object_data=None,
        object_flux_scale=None,
        nrmse_quantile=0.95,
        nrmse_epsilon=1e-8,
):
    """
    Compare held-out observations with the GP predictive mean using RMSE.
    """
    mean, _, _ = predict_observation_distribution(
        gp,
        heldout_data,
        include_yerr=False,
        return_raw_flux=evaluate_raw_metrics,
    )
    if evaluate_raw_metrics:
        y_true, _, _, _ = _raw_observation_arrays(heldout_data)
    else:
        y_true = heldout_data["y"]
    rmse = RMSE(y_true, mean)
    if object_flux_scale is None:
        object_flux_scale = object_level_empirical_flux_scale(
            train_data,
            heldout_data,
            example=object_data,
            q=nrmse_quantile,
            epsilon=nrmse_epsilon,
        )
    object_flux_scale = float(max(float(object_flux_scale), nrmse_epsilon))

    return {
        "rmse": float(rmse),
        "nrmse": float(rmse / object_flux_scale),
        "object_flux_scale": object_flux_scale,
        "nrmse_quantile": float(nrmse_quantile),
        "y_pred": mean,
        "metric_space": "raw" if evaluate_raw_metrics else "normalized",
    }

def evaluate_heldout_nlpd(
        gp,
        heldout_data,
        include_yerr=True,
        yerr_scale=1.0,
        noise_floor=0.0,
        extra_noise=None,
        evaluate_raw_metrics=True,
):
    """
    Compare held-out observations with the GP predictive distribution using NLPD.
    """
    mean, std, variance = predict_observation_distribution(
        gp,
        heldout_data,
        include_yerr=include_yerr,
        yerr_scale=yerr_scale,
        noise_floor=noise_floor,
        extra_noise=extra_noise,
        return_raw_flux=evaluate_raw_metrics,
    )
    if evaluate_raw_metrics:
        y_true, _, _, _ = _raw_observation_arrays(heldout_data)
    else:
        y_true = heldout_data["y"]
    per_point_nlpd = negative_log_predictive_density(
        y_true,
        mean,
        variance,
    )

    return {
        "mean_nlpd": float(np.mean(per_point_nlpd)),
        "total_nlpd": float(np.sum(per_point_nlpd)),
        "per_point_nlpd": per_point_nlpd,
        "y_pred": mean,
        "y_std": std,
        "predictive_variance": variance,
        "metric_space": "raw" if evaluate_raw_metrics else "normalized",
    }

def extract_basic_gp_features(gp, data, mode="observed", n_grid=200):
    if mode == "observed":
        t = data['t']
        t_min = t.min()
        t_max = t.max()
    elif mode == "fixed":
        t_min = -200
        t_max = 400
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    t_grid = np.linspace(t_min, t_max, n_grid)
    X_grid = t_grid.reshape(-1, 1)

    mean, std = gp.predict(X_grid, return_std=True)

    peak_idx = np.argmax(mean)

    peak_time = t_grid[peak_idx]
    peak_flux = mean[peak_idx]

    # approximate decay slope after peak
    after_peak = t_grid > peak_time

    if np.sum(after_peak) >= 5:
        x_decay = t_grid[after_peak][:30]
        y_decay = mean[after_peak][:30]
        decay_slope = np.polyfit(x_decay, y_decay, deg=1)[0]
    else:
        decay_slope = np.nan

    mean_uncertainty = np.mean(std)

    return {
        "peak_time": peak_time,
        "peak_flux": peak_flux,
        "decay_slope": decay_slope,
        "mean_uncertainty": mean_uncertainty,
        "duration": data["t"].max() - data["t"].min(),
        "first_time": data["t"].min(),
        "last_time": data["t"].max(),
    }

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
        "sse": float(np.sum(squared_errors)),
        "object_flux_scale": object_flux_scale,
        "nrmse_quantile": float(nrmse_quantile),
        "coverage": coverage,
        "coverage_counts": coverage_counts,
        "per_point_nlpd": per_point_nlpd,
        "per_point_crps": per_point_crps,
        "squared_errors": squared_errors,
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

    observation_weighted = {
        "nlpd": float(sum(result["total_nlpd"] for result in object_results) / n_total),
        "crps": float(sum(result["total_crps"] for result in object_results) / n_total),
        "rmse": float(np.sqrt(sum(result["sse"] for result in object_results) / n_total)),
        "nrmse": float(np.sqrt(np.average(
            [result["nrmse"] ** 2 for result in object_results],
            weights=[result["n_heldout"] for result in object_results],
        ))),
    }
    object_weighted = {
        "nlpd": float(np.mean([result["mean_nlpd"] for result in object_results])),
        "crps": float(np.mean([result["mean_crps"] for result in object_results])),
        "rmse": float(np.mean([result["rmse"] for result in object_results])),
        "nrmse": float(np.mean([result["nrmse"] for result in object_results])),
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
        "observation_weighted": observation_weighted,
        "object_weighted": object_weighted,
    }


def _scalar_from_result_value(value):
    arr = np.asarray(value, dtype=object)
    if arr.ndim == 0:
        return arr.item()
    if len(arr) == 0:
        return None
    return arr.reshape(-1)[0]


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


def _object_metric_row(result):
    n_test_object = int(result.get("n_heldout", len(result.get("y_true", []))))
    if n_test_object < 1:
        object_id = _scalar_from_result_value(result.get("object_id", None))
        raise ValueError(f"Object {object_id!r} must have at least one held-out prediction.")

    y_true = np.asarray(result["y_true"], dtype=float)
    y_pred = np.asarray(result["y_pred"], dtype=float)
    y_std = np.maximum(np.asarray(result["y_std"], dtype=float), PREDICTIVE_STD_EPSILON)
    if not np.all(y_std > 0):
        raise AssertionError("sigma_pred must be positive after clipping.")

    z = (y_true - y_pred) / y_std
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

    return {
        "object_id": _scalar_from_result_value(result.get("object_id", None)),
        "class": _require_valid_class_label(result),
        "rmse_object": rmse_object,
        "nrmse_object": float(nrmse_object) if nrmse_object is not None else np.nan,
        "nlpd_object": nlpd_object,
        "crps_object": crps_object,
        "coverage_1sigma_object": float(coverage.get("coverage_1sigma", np.mean(abs_z <= 1))),
        "coverage_2sigma_object": float(coverage.get("coverage_2sigma", np.mean(abs_z <= 2))),
        "coverage_3sigma_object": float(coverage.get("coverage_3sigma", np.mean(abs_z <= 3))),
        "z_mean_object": float(np.mean(z)),
        "z_std_object": float(np.std(z)),
        "n_test_object": n_test_object,
        # n_target_train statistics describe target-band data availability for the single-band GP.
        "n_target_train_object": int(n_target_train_object) if n_target_train_object is not None else np.nan,
    }


def single_band_gp_object_metric_table(object_results):
    """
    Build per-object single-band GP held-out metrics before class aggregation.
    """
    if len(object_results) == 0:
        raise ValueError("object_results must contain at least one result.")
    return pd.DataFrame([_object_metric_row(result) for result in object_results])


def _mean_std(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    return float(np.mean(values)), float(np.std(values))


def summarize_single_band_gp_class_metrics(
        object_results,
        min_test_for_zstd=5,
        sparse_threshold=5,
        output_path="single_band_gp_class_summary.csv",
        print_table=True,
):
    """
    Summarize single-band GP evaluation metrics by object class.

    Class-level metrics are object-weighted: metrics are computed per object
    first, then summarized across objects within each class.
    """
    object_table = single_band_gp_object_metric_table(object_results)
    rows = []
    performance_metrics = [
        ("rmse", "rmse_object"),
        ("nrmse", "nrmse_object"),
        ("nlpd", "nlpd_object"),
        ("crps", "crps_object"),
        ("coverage_1sigma", "coverage_1sigma_object"),
        ("coverage_2sigma", "coverage_2sigma_object"),
        ("coverage_3sigma", "coverage_3sigma_object"),
        ("z_mean", "z_mean_object"),
        ("z_std", "z_std_object"),
    ]

    for class_label, group in object_table.groupby("class", sort=True, dropna=False):
        n_objects = int(len(group))
        if n_objects < 1:
            raise AssertionError("Classes with very few objects should be included, not dropped.")

        n_target_train = np.asarray(group["n_target_train_object"], dtype=float)
        sparse_mask = n_target_train < sparse_threshold
        row = {
            "class": class_label,
            "n_objects": n_objects,
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
        }

        for metric_name, column in performance_metrics:
            metric_mean, metric_std = _mean_std(group[column])
            row[f"{metric_name}_mean"] = metric_mean
            row[f"{metric_name}_std"] = metric_std

        # Within-object z_std is unstable for very small n_test_object, so the filtered columns are included for interpretation.
        zstd_group = group[group["n_test_object"] >= min_test_for_zstd]
        row["n_objects_zstd_n_test_ge_5"] = int(len(zstd_group))
        if len(zstd_group) == 0:
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


def collect_heldout_predictions(object_results):
    """
    Concatenate held-out y_true, y_pred, and y_std arrays across objects.
    """
    if len(object_results) == 0:
        raise ValueError("object_results must contain at least one result.")

    y_true = np.concatenate([np.asarray(result["y_true"]) for result in object_results])
    y_pred = np.concatenate([np.asarray(result["y_pred"]) for result in object_results])
    y_std = np.concatenate([np.asarray(result["y_std"]) for result in object_results])

    return y_true, y_pred, y_std


def standardized_residual_statistics(object_results):
    """
    Compute pooled standardized residual statistics across all held-out points.
    """
    y_true, y_pred, y_std = collect_heldout_predictions(object_results)
    y_std = np.maximum(y_std, 1e-12)
    z = (y_true - y_pred) / y_std
    abs_z = np.abs(z)

    return {
        "z": z,
        "mean_z": float(np.mean(z)),
        "std_z": float(np.std(z)),
        "coverage_1sigma": float(np.mean(abs_z <= 1)),
        "coverage_2sigma": float(np.mean(abs_z <= 2)),
        "coverage_3sigma": float(np.mean(abs_z <= 3)),
        "max_abs_z": float(np.max(abs_z)),
        "p95_abs_z": float(np.percentile(abs_z, 95)),
        "p99_abs_z": float(np.percentile(abs_z, 99)),
    }


def yerr_statistics(object_results):
    """
    Summarize held-out measurement errors and their relationship to residuals.
    """
    if len(object_results) == 0:
        raise ValueError("object_results must contain at least one result.")

    yerr = np.concatenate([np.asarray(result["yerr"]) for result in object_results])
    y_true, y_pred, y_std = collect_heldout_predictions(object_results)
    residual = y_true - y_pred
    z = residual / np.maximum(y_std, 1e-12)
    abs_z = np.abs(z)

    return {
        "min_yerr": float(np.min(yerr)),
        "median_yerr": float(np.median(yerr)),
        "mean_yerr": float(np.mean(yerr)),
        "p95_yerr": float(np.percentile(yerr, 95)),
        "max_yerr": float(np.max(yerr)),
        "median_yerr_top_1pct_abs_z": float(np.median(yerr[abs_z >= np.percentile(abs_z, 99)])),
        "median_yerr_top_5pct_abs_z": float(np.median(yerr[abs_z >= np.percentile(abs_z, 95)])),
    }


def largest_standardized_residual_cases(object_results, top_n=20):
    """
    Return the held-out points with the largest absolute standardized residuals.
    """
    if len(object_results) == 0:
        raise ValueError("object_results must contain at least one result.")

    rows = []
    for result_idx, result in enumerate(object_results):
        y_true = np.asarray(result["y_true"])
        y_pred = np.asarray(result["y_pred"])
        y_std = np.maximum(np.asarray(result["y_std"]), 1e-12)
        z = (y_true - y_pred) / y_std
        abs_z = np.abs(z)

        for point_idx in range(len(y_true)):
            rows.append({
                "result_idx": result_idx,
                "point_idx": point_idx,
                "object_id": result["object_id"][point_idx],
                "band": result["band"][point_idx],
                "time": result["time"][point_idx],
                "X_test": result["X_test"][point_idx],
                "y": y_true[point_idx],
                "mean": y_pred[point_idx],
                "std": y_std[point_idx],
                "z": z[point_idx],
                "abs_z": abs_z[point_idx],
                "yerr": result["yerr"][point_idx],
                "outside_train_range": result["outside_train_range"][point_idx],
                "distance_to_train_range": result["distance_to_train_range"][point_idx],
                "near_peak": result["near_peak"][point_idx],
                "n_train": result["n_train"],
                "n_heldout": result["n_heldout"],
                "train_time_min": result["train_time_min"],
                "train_time_max": result["train_time_max"],
            })

    # Sort by absolute standardized residual and return the top cases
    rows = sorted(rows, key=lambda row: row["abs_z"], reverse=True)
    return rows[:top_n]


def print_largest_standardized_residual_cases(object_results, top_n=20):
    """
    Print the largest standardized residual cases in a notebook-friendly format.
    """
    rows = largest_standardized_residual_cases(object_results, top_n=top_n)
    for row in rows:
        print(
            "object_id:", row["object_id"],
            "band:", row["band"],
            "time:", row["time"],
            "X_test:", row["X_test"],
            "y:", row["y"],
            "mean:", row["mean"],
            "std:", row["std"],
            "z:", row["z"],
            "yerr:", row["yerr"],
            "outside_train_range:", row["outside_train_range"],
            "near_peak:", row["near_peak"],
            "n_train:", row["n_train"],
        )
    return rows
