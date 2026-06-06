import matplotlib.pyplot as plt
import numpy as np

from MOGP_model import DEFAULT_BAND_TO_WAVELENGTH, _resolve_wavelength

def plot_gp_fit(gp, data, heldout_data=None, n_grid=300):
    """
    Plots the Gaussian Process fit along with the original data points.
    
    Args:
        gp: The fitted GaussianProcessRegressor model.
        data: A dictionary containing training 't', 'y', and 'yerr' arrays.
        heldout_data: Optional held-out observations to overlay.
        n_grid: Number of points in the grid for prediction.
    """
    t = data['t']
    y = data['y']
    yerr = data['yerr']
    
    # Create a grid of time points for prediction
    t_min = t.min()
    t_max = t.max()
    t_grid = np.linspace(t_min, t_max, n_grid).reshape(-1, 1)
    
    # Predict mean and standard deviation from the GP
    y_pred, y_std = gp.predict(t_grid, return_std=True)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot original data points with error bars
    plt.errorbar(t.flatten(), y.flatten(), yerr=yerr.flatten(), fmt='o', label='Train', alpha=0.5)

    if heldout_data is not None:
        plt.errorbar(
            np.asarray(heldout_data['t']).flatten(),
            np.asarray(heldout_data['y']).flatten(),
            yerr=np.asarray(heldout_data['yerr']).flatten(),
            fmt='s',
            label='Held-out',
            color='black',
            alpha=0.8,
        )
    
    # Plot GP mean prediction
    plt.plot(t_grid.flatten(), y_pred.flatten(), label='GP Mean', color='red')
    
    # Fill between mean ± std
    plt.fill_between(t_grid.flatten(), (y_pred - y_std).flatten(), (y_pred + y_std).flatten(), 
                     color='red', alpha=0.3, label='GP Latent Std Dev')
    
    plt.title(
        f"{data['obj_type']} | band {data['band']} | object {data['obj_id']}"
    )
    plt.xlabel('Time (days)')
    plt.ylabel('Normalized Flux')
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_largest_standardized_residual_cases(object_results, cases, max_plots=10, n_grid=300):
    """
    Plot objects for the largest standardized residual cases.

    object_results must contain "gp", "train_data", and "heldout_data" entries
    for each object, as added in the notebook after evaluate_heldout_metrics.
    """
    plotted = set()
    n_plotted = 0

    for case in cases:
        result_idx = case["result_idx"]
        point_idx = case["point_idx"]
        if (result_idx, point_idx) in plotted:
            continue

        result = object_results[result_idx]
        gp = result["gp"]
        train_data = result["train_data"]
        heldout_data = result["heldout_data"]

        t_all = np.concatenate([
            np.asarray(train_data["t"]),
            np.asarray(heldout_data["t"]),
        ])
        t_grid = np.linspace(t_all.min(), t_all.max(), n_grid).reshape(-1, 1)
        y_pred, y_std = gp.predict(t_grid, return_std=True)

        plt.figure(figsize=(10, 6))
        plt.errorbar(
            np.asarray(train_data["t"]).flatten(),
            np.asarray(train_data["y"]).flatten(),
            yerr=np.asarray(train_data["yerr"]).flatten(),
            fmt="o",
            label="Train",
            alpha=0.5,
        )
        plt.errorbar(
            np.asarray(heldout_data["t"]).flatten(),
            np.asarray(heldout_data["y"]).flatten(),
            yerr=np.asarray(heldout_data["yerr"]).flatten(),
            fmt="s",
            label="Held-out",
            color="black",
            alpha=0.65,
        )
        plt.scatter(
            [case["time"]],
            [case["y"]],
            s=140,
            facecolors="none",
            edgecolors="orange",
            linewidths=2.5,
            label="Highlighted |z| case",
            zorder=5,
        )
        plt.plot(t_grid.flatten(), y_pred.flatten(), label="GP Mean", color="red")
        plt.fill_between(
            t_grid.flatten(),
            (y_pred - y_std).flatten(),
            (y_pred + y_std).flatten(),
            color="red",
            alpha=0.25,
            label="GP Latent Std Dev",
        )
        plt.axvline(case["train_time_min"], color="gray", linestyle="--", alpha=0.4)
        plt.axvline(case["train_time_max"], color="gray", linestyle="--", alpha=0.4)
        plt.title(
            f"{case['object_id']} | band {case['band']} | "
            f"z={case['z']:.2f} | edge={case['outside_train_range']} | peak={case['near_peak']}"
        )
        plt.xlabel("Time (standardized, peak-aligned)")
        plt.ylabel("Normalized Flux")
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        plt.grid()
        plt.tight_layout()
        plt.show()

        plotted.add((result_idx, point_idx))
        n_plotted += 1
        if n_plotted >= max_plots:
            break


def _column(data, key, default=None):
    if data is None:
        return default
    if isinstance(data, dict):
        return data.get(key, default)
    if hasattr(data, "columns") and key in data.columns:
        return data[key].to_numpy()
    return getattr(data, key, default)


def _band_mask(data, selected_band):
    band = np.asarray(_column(data, "band"), dtype=object)
    if band.size == 0:
        return np.array([], dtype=bool)

    direct = band == selected_band
    as_str = band.astype(str) == str(selected_band)
    try:
        selected_int = int(selected_band)
    except (TypeError, ValueError):
        selected_int = None
    if selected_int is None:
        return direct | as_str

    numeric = np.array([
        False if value is None else _matches_int_band(value, selected_int)
        for value in band
    ])
    return direct | as_str | numeric


def _matches_int_band(value, selected_int):
    try:
        return int(value) == selected_int
    except (TypeError, ValueError):
        return False


def _selected_band_background(data, selected_band, mask):
    backgrounds = _column(data, "background_by_band", None)
    if isinstance(backgrounds, dict):
        if selected_band in backgrounds:
            return float(backgrounds[selected_band])
        selected_str = str(selected_band)
        if selected_str in backgrounds:
            return float(backgrounds[selected_str])
        try:
            selected_int = int(selected_band)
        except (TypeError, ValueError):
            selected_int = None
        if selected_int in backgrounds:
            return float(backgrounds[selected_int])

    background_flux = _column(data, "background_flux", 0.0)
    background_flux = np.asarray(background_flux, dtype=float)
    if background_flux.ndim == 0:
        return float(background_flux)
    if len(background_flux) == len(mask) and np.any(mask):
        return float(np.median(background_flux[mask]))
    return 0.0


def _raw_time_from_processed(data, t_norm):
    t_scale = float(_column(data, "t_scale", 1.0))
    alignment_peak_time = float(_column(data, "alignment_peak_time", 0.0))
    return np.asarray(t_norm, dtype=float) * t_scale + alignment_peak_time


def plot_multiband_gp_single_band_prediction(
        gp,
        object_data,
        selected_band,
        train_data,
        heldout_data,
        band_to_wavelength=None,
        n_grid=300,
        output_path=None,
        ax=None,
        plot_raw_flux=True,
        plot_raw_time=False,
):
    """
    Plot one selected band's prediction from a fitted multiband GP.

    The GP is not refit.  The prediction grid is built by varying normalized
    time over the object's observed range and fixing wavelength to the selected
    band's central wavelength.

    Example:
        fig, ax = plot_multiband_gp_single_band_prediction(
            gp,
            object_data=train_data["obj_id"],
            selected_band="r",
            train_data=train_data,
            heldout_data=heldout_data,
        )
    """
    band_to_wavelength = band_to_wavelength or DEFAULT_BAND_TO_WAVELENGTH
    wavelength = _resolve_wavelength(selected_band, band_to_wavelength)

    train_mask = _band_mask(train_data, selected_band)
    heldout_mask = _band_mask(heldout_data, selected_band)
    if not np.any(train_mask) and not np.any(heldout_mask):
        raise ValueError(f"No train or held-out rows found for selected_band={selected_band!r}.")

    t_all = np.concatenate([
        np.asarray(_column(train_data, "t"), dtype=float),
        np.asarray(_column(heldout_data, "t"), dtype=float),
    ])
    t_grid = np.linspace(np.min(t_all), np.max(t_all), n_grid)
    X_grid = np.column_stack([t_grid, np.full(n_grid, wavelength, dtype=float)])
    mean_norm, std_norm = gp.predict(X_grid, return_std=True)

    flux_scale = float(_column(train_data, "flux_scale", _column(heldout_data, "flux_scale", 1.0)))
    background = _selected_band_background(train_data, selected_band, train_mask)

    if plot_raw_time:
        x_grid = _raw_time_from_processed(train_data, t_grid)
        train_x = _raw_time_from_processed(train_data, np.asarray(_column(train_data, "t"), dtype=float)[train_mask])
        heldout_x = _raw_time_from_processed(
            heldout_data,
            np.asarray(_column(heldout_data, "t"), dtype=float)[heldout_mask],
        )
        xlabel = "Time"
    else:
        x_grid = t_grid
        train_x = np.asarray(_column(train_data, "t"), dtype=float)[train_mask]
        heldout_x = np.asarray(_column(heldout_data, "t"), dtype=float)[heldout_mask]
        xlabel = "Time (standardized, peak-aligned)"

    if plot_raw_flux:
        mean = mean_norm * flux_scale + background
        std = std_norm * flux_scale
        train_y = np.asarray(_column(train_data, "y_raw"), dtype=float)[train_mask]
        heldout_y = np.asarray(_column(heldout_data, "y_raw"), dtype=float)[heldout_mask]
        train_yerr = np.asarray(_column(train_data, "yerr_raw", np.nan), dtype=float)
        heldout_yerr = np.asarray(_column(heldout_data, "yerr_raw", np.nan), dtype=float)
        ylabel = "Flux"
    else:
        mean = mean_norm
        std = std_norm
        train_y = np.asarray(_column(train_data, "y"), dtype=float)[train_mask]
        heldout_y = np.asarray(_column(heldout_data, "y"), dtype=float)[heldout_mask]
        train_yerr = np.asarray(_column(train_data, "yerr", np.nan), dtype=float)
        heldout_yerr = np.asarray(_column(heldout_data, "yerr", np.nan), dtype=float)
        ylabel = "Normalized Flux"

    train_yerr = train_yerr[train_mask] if train_yerr.ndim > 0 and len(train_yerr) == len(train_mask) else None
    heldout_yerr = (
        heldout_yerr[heldout_mask]
        if heldout_yerr.ndim > 0 and len(heldout_yerr) == len(heldout_mask)
        else None
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    ax.plot(x_grid, mean, color="tab:red", label="MOGP mean")
    ax.fill_between(
        x_grid,
        mean - std,
        mean + std,
        color="tab:red",
        alpha=0.25,
        label="MOGP +/- 1 sigma",
    )
    if np.any(train_mask):
        ax.errorbar(
            train_x,
            train_y,
            yerr=train_yerr,
            fmt="o",
            color="tab:blue",
            ecolor="tab:blue",
            alpha=0.75,
            label="Train",
        )
    if np.any(heldout_mask):
        ax.errorbar(
            heldout_x,
            heldout_y,
            yerr=heldout_yerr,
            fmt="s",
            color="black",
            ecolor="black",
            alpha=0.75,
            label="Held-out",
        )

    if isinstance(object_data, dict):
        object_label = object_data.get("object_id", object_data.get("obj_id", "unknown object"))
    else:
        object_label = object_data
    ax.set_title(f"Object {object_label} | band {selected_band} ({wavelength:.1f} nm)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")

    return fig, ax
