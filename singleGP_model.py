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

    # if no example is provided, or if the example does not contain a "lightcurve" key, then we fall back to looking for raw flux arrays in the datasets.
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
