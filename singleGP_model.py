from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern
import numpy as np

def fit_basic_gp(data, kernel_type="matern"):
    X = data['X']
    y = data['y']

    if kernel_type == "rbf":
        kernel = ConstantKernel(1.0) * RBF(length_scale=50.0)
    elif kernel_type == "matern":
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=0.3, nu=1.5)
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")
    
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=data["yerr"]**2 + 0.03**2,    # Use the squared errors as the noise level
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=0,
    )

    gp.fit(X, y)
    
    return gp


def predict_observation_distribution(gp, data, include_yerr=True, extra_noise=0.03):
    """
    Predict the distribution for observed flux values at data['X'].

    sklearn's GP predictive std is for the latent function. For held-out
    observations, NLPD should include measurement error and any extra noise
    term used during training.
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

    return 0.5 * (np.log(2 * np.pi * variance) + ((y_true - mean) ** 2) / variance)


def evaluate_heldout_nlpd(gp, heldout_data, include_yerr=True, extra_noise=0.03):
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
