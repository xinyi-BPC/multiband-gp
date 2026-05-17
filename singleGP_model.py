from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern
import numpy as np

def fit_basic_gp(
        data,
        kernel_type="matern",
        length_scale=0.3,
        length_scale_bounds=(0.05, 5.0),
        constant_value=1.0,
        constant_value_bounds=(1e-2, 1e2),
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
        alpha=data["yerr"]**2 + 0.03**2,    # Use the squared errors as the noise level
        normalize_y=True,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=random_state,
    )

    gp.fit(X, y)
    
    return gp


def predict_observation_distribution(gp, data, include_yerr=True, extra_noise=0):
    """
    Predict the distribution for observed flux values at data['X'].

    sklearn's GP predictive std is for the latent function. For held-out
    observations, add the test measurement error and any modeled observation noise.
    """
    mean, latent_std = gp.predict(data["X"], return_std=True)
    variance = latent_std ** 2

    if include_yerr:
        variance = variance + np.asarray(data["yerr"]) ** 2
    if extra_noise is not None and extra_noise > 0:
        variance = variance + extra_noise ** 2

    return mean, np.sqrt(variance), variance


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

def evaluate_heldout_rmse(gp, heldout_data):
    """
    Compare held-out observations with the GP predictive mean using RMSE.
    """
    mean, _, _ = predict_observation_distribution(gp, heldout_data, include_yerr=False)
    rmse = RMSE(heldout_data["y"], mean)

    return {
        "rmse": float(rmse),
        "y_pred": mean,
    }

def evaluate_heldout_nlpd(gp, heldout_data, include_yerr=True, extra_noise=0):
    """
    Compare held-out observations with the GP predictive distribution using NLPD.
    """
    mean, std, variance = predict_observation_distribution(
        gp,
        heldout_data,
        include_yerr=include_yerr,
        extra_noise=extra_noise,
    )
    per_point_nlpd = negative_log_predictive_density(
        heldout_data["y"],
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
        coverage_sigmas=(1.0, 2.0, 3.0),
        include_yerr=True,
        extra_noise=0,
):
    """
    Evaluate NLPD, RMSE, and coverage for one object's held-out observations.
    """
    mean, std, variance = predict_observation_distribution(
        gp,
        heldout_data,
        include_yerr=include_yerr,
        extra_noise=extra_noise,
    )
    y_true = np.asarray(heldout_data["y"])
    errors = y_true - mean
    squared_errors = errors ** 2
    per_point_nlpd = negative_log_predictive_density(y_true, mean, variance)

    coverage = {}
    coverage_counts = {}
    for sigma in coverage_sigmas:
        covered = np.abs(errors) <= sigma * std
        key = f"coverage_{sigma:g}sigma"
        coverage[key] = float(np.mean(covered))
        coverage_counts[key] = int(np.sum(covered))

    n_heldout = len(y_true)

    return {
        "n_heldout": n_heldout,
        "mean_nlpd": float(np.mean(per_point_nlpd)),
        "total_nlpd": float(np.sum(per_point_nlpd)),
        "rmse": float(np.sqrt(np.mean(squared_errors))),
        "sse": float(np.sum(squared_errors)),
        "coverage": coverage,
        "coverage_counts": coverage_counts,
        "per_point_nlpd": per_point_nlpd,
        "squared_errors": squared_errors,
        "y_true": y_true,
        "y_pred": mean,
        "y_std": std,
        "predictive_variance": variance,
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
