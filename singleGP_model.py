from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern
import numpy as np

def fit_basic_gp(
        data,
        kernel_type="matern",
        length_scale=0.3,
        length_scale_bounds=(0.1, 5.0),
        constant_value=1.0,
        constant_value_bounds=(1e-2, 1e2),
        yerr_scale=1.0,
        noise_floor=0.0,
        jitter=1e-8,
        n_restarts_optimizer=5,
        random_state=0,
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
    scale = float(data["flux_scale"])
    background = float(data.get("background_flux", 0.0))
    y_norm = np.asarray(data["y"])
    yerr_norm = np.asarray(data["yerr"])

    y_raw = np.asarray(data.get("y_raw", y_norm * scale + background))
    yerr_raw = np.asarray(data.get("yerr_raw", yerr_norm * scale))

    return y_raw, yerr_raw, scale, background


def negative_log_predictive_density(y_true, mean, variance):
    """
    Compute pointwise negative log predictive density under a Gaussian.
    """
    y_true = np.asarray(y_true)
    mean = np.asarray(mean)
    variance = np.maximum(np.asarray(variance), 1e-12)

    return 0.5 * (np.log(2 * np.pi * variance) + ((y_true - mean) ** 2) / variance)   # Gaussian NLPD formula

def RMSE(y_true, y_pred):
    """Compute root mean squared error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def evaluate_heldout_rmse(gp, heldout_data, evaluate_raw_metrics=True):
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

    return {
        "rmse": float(rmse),
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
    y_true_norm = np.asarray(heldout_data["y"])
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
    std_raw = np.sqrt(variance_raw)

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
    per_point_nlpd = negative_log_predictive_density(y_metric, mean_metric, variance_metric)

    n_heldout = len(y_metric)
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
        "rmse": float(np.sqrt(np.mean(squared_errors))),
        "sse": float(np.sum(squared_errors)),
        "coverage": coverage,
        "coverage_counts": coverage_counts,
        "per_point_nlpd": per_point_nlpd,
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

    coverage_keys = sorted(object_results[0]["coverage"].keys())

    observation_weighted = {
        "nlpd": float(sum(result["total_nlpd"] for result in object_results) / n_total),
        "rmse": float(np.sqrt(sum(result["sse"] for result in object_results) / n_total)),
    }
    object_weighted = {
        "nlpd": float(np.mean([result["mean_nlpd"] for result in object_results])),
        "rmse": float(np.mean([result["rmse"] for result in object_results])),
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
        "observation_weighted": observation_weighted,
        "object_weighted": object_weighted,
    }


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
