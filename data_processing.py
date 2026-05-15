import numpy as np


def _subset_processed_data(data, indices):
    subset = data.copy()
    for key in ("X", "t", "y", "yerr"):
        subset[key] = np.asarray(data[key])[indices]
    return subset


def split_train_heldout_observations(
        data,
        heldout_fraction=0.2,
        min_train_points=3,
        min_heldout_points=1,
        random_state=0,
        strategy="random",
):
    """
    Split one processed light curve into train and held-out observations.

    Args:
        data: Processed dictionary returned by process_one_obj_one_band.
        heldout_fraction: Fraction of observations to hold out.
        min_train_points: Minimum number of observations kept for fitting.
        min_heldout_points: Minimum number of observations held out.
        random_state: Seed used when strategy="random".
        strategy: only support "random"
    Returns:
        (train_data, heldout_data), each with the same keys as data plus
        train_indices/heldout_indices metadata.
    """
    n_obs = len(data["y"])
    if n_obs < min_train_points + min_heldout_points:
        raise ValueError(
            "Not enough observations to make the requested split: "
            f"{n_obs} available, {min_train_points + min_heldout_points} required."
        )

    n_heldout = int(np.ceil(n_obs * heldout_fraction))
    n_heldout = max(min_heldout_points, n_heldout)
    n_heldout = min(n_heldout, n_obs - min_train_points)

    if strategy == "random":
        rng = np.random.default_rng(random_state)
        heldout_indices = np.sort(rng.choice(n_obs, size=n_heldout, replace=False))
    else:
        raise ValueError(f"Unsupported split strategy: {strategy}")

    train_indices = np.setdiff1d(np.arange(n_obs), heldout_indices)
    train_indices = train_indices[np.argsort(np.asarray(data["t"])[train_indices])]
    heldout_indices = heldout_indices[np.argsort(np.asarray(data["t"])[heldout_indices])]

    train_data = _subset_processed_data(data, train_indices)
    heldout_data = _subset_processed_data(data, heldout_indices)
    train_data["train_indices"] = train_indices
    heldout_data["heldout_indices"] = heldout_indices

    return train_data, heldout_data


def process_one_obj_one_band(
        example,
        target_band='r',
        align_peak=True,
        normalize_flux=True,
        min_points=5,
):
    """
    Process one object in one band.
    Args:
        example: a dictionary containing the light curve data for one object.
        target_band: the band to process (e.g., 'r').
        align_peak: whether to align the peak of the light curve to time zero.
        normalize_flux: whether to normalize the flux values by the peak flux.
        min_points: minimum number of data points required to keep the example.
    Returns:
        A dictionary with processed time and flux arrays, or None if the example is discarded.
    """
    # Extract time and flux
    lc = example["lightcurve"]
    time = np.array(lc['time'])
    flux = np.array(lc['flux'])
    flux_err = np.array(lc["flux_err"])
    band = np.array(lc["band"])
    
    # Extract entries from the target band and remove invalid entries
    valid = (band == target_band) * (time > 0) * np.isfinite(time) * np.isfinite(flux) * np.isfinite(flux_err)

    t = time[valid]
    y = flux[valid]
    yerr = flux_err[valid]

    # Check if there are enough points
    if len(t) < min_points:
        return None
    
    # sort by time
    order = np.argsort(t)
    t = t[order]
    y = y[order]
    yerr = yerr[order]

    # Align peak if required
    if align_peak:
        peak_index = np.argmax(y)
        t = t - t[peak_index]
        
    # Normalize time by its standard deviation to help with GP fitting
    t_scale = np.std(t)
    if t_scale <= 0:
        return None

    t = t / t_scale

    # Normalize flux if required
    if normalize_flux:
        peak_flux = np.max(np.abs(y))   # Use absolute value to handle negative peaks
        if peak_flux <= 0:
            return None
        
        y = y / peak_flux
        yerr = yerr / peak_flux
    else:
        # If not normalizing, we can still filter out examples with non-positive peak flux
        if np.max(np.abs(y)) <= 0:
            return None
        peak_flux = 1.0
    
    # sklearn expects X shape = (n_samples, n_features)
    X = t.reshape(-1, 1)

    return {
        'X': X,
        't': t,
        'y': y,
        'yerr': yerr,
        'band': target_band,
        'obj_type': example['obj_type'],
        'obj_id': example['object_id'],
        't_peak': t[np.argmax(y)],
        'peak_flux': peak_flux,
    }
