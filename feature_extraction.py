import numpy as np

def extract_lightcurve_features(time, flux, flux_err=None, t_peak=None):
    time = np.asarray(time)
    flux = np.asarray(flux)

    valid = np.isfinite(time) & np.isfinite(flux)
    time = time[valid]
    flux = flux[valid]

    if flux_err is not None:
        flux_err = np.asarray(flux_err)[valid]
    
    if len(time) < 3:
        return None # Not enough data points to extract features
    
    flux_scale = np.quantile(np.abs(flux), 0.95)
    if flux_scale <= 1e-6:
        return None # Flux values are too small to extract meaningful features
    
    flux_norm = flux / flux_scale
    
    if t_peak is None:
        t_peak = time[np.argmax(np.abs(flux_norm))]
    
    t = time - t_peak

    order = np.argsort(t)
    t = t[order]
    f = flux_norm[order]

    dt = np.diff(t)
    df = np.diff(f)
    valid_dt = dt > 0

    if np.any(valid_dt):
        cadence = dt[valid_dt]
        slope = df[valid_dt] / cadence
    else:
        cadence = np.array([np.nan])
        slope = np.array([np.nan])  # No valid slopes if no valid dt

    amp_abs = np.nanmax(np.abs(f))

    near_peak_50 = np.abs(t) <= 50
    near_peak_100 = np.abs(t) <= 100

    above_20 = np.abs(f) >= 0.2
    above_50 = np.abs(f) >= 0.5

    if np.any(above_20):
        width_20 = np.nanmax(t[above_20]) - np.nanmin(t[above_20])
    else:
        width_20 = np.nan

    if np.any(above_50):
        width_50 = np.nanmax(t[above_50]) - np.nanmin(t[above_50])
    else:
        width_50 = np.nan

    flat_fraction = np.mean(np.abs(f) < 0.05 * amp_abs)

    sign_changes = np.sum(np.diff(np.sign(f)) != 0) / max(len(f) - 1, 1)

    # Functuation degree calculation 
    flux_std = np.nanstd(f)
    flux_iqr = np.nanpercentile(f, 75) - np.nanpercentile(f, 25)
    flux_range = np.nanmax(f) - np.nanmin(f)
    
    total_variation = np.nansum(np.abs(np.diff(f)))
    mean_abs_slope = np.nanmean(np.abs(np.diff(f) / np.diff(t)))
    slope_std = np.nanstd(np.diff(f) / np.diff(t))
    sign_change_rate = np.mean(np.diff(np.sign(f - np.nanmedian(f))) != 0)  # measures how often the curve crosses its median/baseline.

    features = {
        "n_points": len(t),

        "median_cadence": np.nanmedian(cadence),
        "mean_cadence": np.nanmean(cadence),
        "max_gap": np.nanmax(cadence),
        "gap_90": np.nanquantile(cadence, 0.90),

        "peak_amplitude": amp_abs,
        "width_20": width_20,
        "width_50": width_50,
        "flat_fraction": flat_fraction,

        "n_near_peak_50": np.sum(near_peak_50),
        "n_near_peak_100": np.sum(near_peak_100),
        "distance_peak_to_nearest_point": np.nanmin(np.abs(t)),

        "total_variation": np.nansum(np.abs(df)),
        "mean_abs_slope": np.nanmean(np.abs(slope)),
        "max_abs_slope": np.nanmax(np.abs(slope)),
        "slope_std": np.nanstd(slope),
        "sign_change_rate": sign_changes,
    }

    if flux_err is not None:
        e = flux_err[order]
        median_err = np.nanmedian(e)
        features.update({
            "median_flux_err": median_err,
            "noise_to_signal": median_err / flux_scale,
            "peak_snr": np.nanmax(np.abs(flux)) / median_err,
            "median_snr": np.nanmedian(np.abs(flux) / e),
        })

    return features