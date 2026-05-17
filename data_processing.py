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

def _extract_valid_band_observations(example, target_band):
    lc = example["lightcurve"]
    time = np.array(lc['time'])
    flux = np.array(lc['flux'])
    flux_err = np.array(lc["flux_err"])
    band = np.array(lc["band"])

    valid = (band == target_band) * (time > 0) * np.isfinite(time) * np.isfinite(flux) * np.isfinite(flux_err)

    t = time[valid]
    y = flux[valid]
    yerr = flux_err[valid]

    order = np.argsort(t)

    return t[order], y[order], yerr[order]


def _is_usable_one_band_example(example, target_band, min_points):
    t, _, _ = _extract_valid_band_observations(example, target_band)

    return len(t) >= min_points and np.std(t) > 0


def find_global_percentile_flux_peak(examples, target_band, percentile=99, min_points=5):
    """
    Find a global percentile-based flux scale from an iterable of examples.

    Args:
        examples: Iterable of object examples.
        target_band: The band to filter the data (e.g., 'r').
        percentile: Percentile of absolute flux values to use as the scale.
        min_points: Minimum valid points required for an object to contribute.
    Returns:
        percentile_based_peak_flux: The flux value at the specified percentile.
    """
    flux_values = []
    for example in examples:
        t, y, _ = _extract_valid_band_observations(example, target_band)
        if len(t) >= min_points and np.std(t) > 0:
            flux_values.append(np.abs(y))

    if len(flux_values) == 0:
        return None

    flux_values = np.concatenate(flux_values)
    percentile_based_peak_flux = np.percentile(flux_values, percentile)
    if percentile_based_peak_flux <= 0:
        raise ValueError("Computed flux scale for this band is zero or negative.")

    return percentile_based_peak_flux


def select_examples_and_global_flux_scale(
        examples,
        target_band='r',
        n_objects=300,
        max_examples_to_scan=3000,
        percentile=99,
        min_points=8,
):
    """
    Select usable examples and compute a global flux scale from the same objects.

    This is useful for streaming datasets: the selected examples are exactly the
    examples that should later be used by the training/evaluation loop.
    """
    selected_examples = []
    scanned_examples = 0

    for stream_idx, example in enumerate(examples):
        if len(selected_examples) >= n_objects or stream_idx >= max_examples_to_scan:
            break

        scanned_examples = stream_idx + 1
        if _is_usable_one_band_example(example, target_band, min_points):
            selected_examples.append(example)

    flux_scale = find_global_percentile_flux_peak(
        selected_examples,
        target_band=target_band,
        percentile=percentile,
        min_points=min_points,
    )

    return selected_examples, flux_scale, scanned_examples

def process_one_obj_one_band(
        example,
        flux_scale=None,
        target_band='r',
        align_peak=True,
        normalize_flux=True,
        min_points=5,
):
    """
    Process one object in one band.
    Args:
        example: a dictionary containing the light curve data for <one object>.
        flux_scale: a global flux scale to use for normalization, or None to use the peak flux of this example.
        target_band: the band to process (e.g., 'r').
        align_peak: whether to align the peak of the light curve to time zero.
        normalize_flux: whether to normalize the flux values by the peak flux.
        min_points: minimum number of data points required to keep the example.
    Returns:
        A dictionary with processed time and flux arrays, or None if the example is discarded.
    """
    t, y, yerr = _extract_valid_band_observations(example, target_band)

    # Check if there are enough points
    if len(t) < min_points:
        return None

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
        if flux_scale is None:
            peak_flux = np.max(np.abs(y))
            if peak_flux <= 0:
                return None
            flux_scale = peak_flux
        else:
            peak_flux = flux_scale
        y = y / flux_scale
        yerr = yerr / flux_scale
    else:
        # If not normalizing, we can still filter out examples with non-positive peak flux
        if np.max(np.abs(y)) <= 0:
            return None
        peak_flux = 1.0
        flux_scale = 1.0
    
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
        'flux_scale': flux_scale,
    }
