from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern
import numpy as np

def fit_basic_gp(data, kernel_type="matern"):
    X = data['X']
    y = data['y']
    yerr = data['yerr']

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