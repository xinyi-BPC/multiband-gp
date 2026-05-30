import numpy as np


def _subset_processed_data(data, indices):
    subset = data.copy()
    for key in ("X", "t", "y", "yerr"):
        subset[key] = np.asarray(data[key])[indices]
    return subset


def _split_indices(
        n_obs,
        heldout_fraction=0.2,
        min_train_points=3,
        min_heldout_points=1,
        random_state=0,
        strategy="random",
        force_train_indices=None,
):
    force_train_indices = (
        np.array([], dtype=int)
        if force_train_indices is None
        else np.unique(np.asarray(force_train_indices, dtype=int))
    )
    if np.any((force_train_indices < 0) | (force_train_indices >= n_obs)):
        raise ValueError("force_train_indices contains indices outside the observation range.")

    if n_obs < min_train_points + min_heldout_points:
        raise ValueError(
            "Not enough observations to make the requested split: "
            f"{n_obs} available, {min_train_points + min_heldout_points} required."
        )

    candidate_heldout_indices = np.setdiff1d(np.arange(n_obs), force_train_indices)
    if len(candidate_heldout_indices) < min_heldout_points:
        raise ValueError(
            "Not enough non-forced observations to make the requested held-out split: "
            f"{len(candidate_heldout_indices)} available, {min_heldout_points} required."
        )

    n_heldout = int(np.ceil(n_obs * heldout_fraction))
    n_heldout = max(min_heldout_points, n_heldout)
    n_heldout = min(n_heldout, n_obs - min_train_points)
    n_heldout = min(n_heldout, len(candidate_heldout_indices))

    if strategy == "random":
        rng = np.random.default_rng(random_state)
        heldout_indices = np.sort(
            rng.choice(candidate_heldout_indices, size=n_heldout, replace=False)
        )
    else:
        raise ValueError(f"Unsupported split strategy: {strategy}")

    train_indices = np.setdiff1d(np.arange(n_obs), heldout_indices)

    return train_indices, heldout_indices


def split_train_heldout_observations(
        data,
        heldout_fraction=0.2,
        min_train_points=3,
        min_heldout_points=1,
        random_state=0,
        strategy="random",
        force_peak_in_train=False,
        peak_mode="absolute",
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
        force_peak_in_train: If True, keep the peak-flux observation out of held-out data.
        peak_mode: "absolute" or "positive"; used when force_peak_in_train=True.
    Returns:
        (train_data, heldout_data), each with the same keys as data plus
        train_indices/heldout_indices metadata.
    """
    n_obs = len(data["y"])
    force_train_indices = (
        _find_peak_indices(data["y"], peak_mode=peak_mode)
        if force_peak_in_train
        else None
    )
    train_indices, heldout_indices = _split_indices(
        n_obs,
        heldout_fraction=heldout_fraction,
        min_train_points=min_train_points,
        min_heldout_points=min_heldout_points,
        random_state=random_state,
        strategy=strategy,
        force_train_indices=force_train_indices,
    )
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


def _find_peak_indices(y, peak_mode="absolute"):
    y = np.asarray(y)
    if len(y) == 0:
        return np.array([], dtype=int)

    if peak_mode == "absolute":
        peak_values = np.abs(y)
    elif peak_mode == "positive":
        peak_values = y
    else:
        raise ValueError(f"Unsupported peak_mode: {peak_mode}")

    return np.flatnonzero(np.isclose(peak_values, np.max(peak_values)))


def _find_alignment_peak_time(example, target_band, t, y, peak_alignment):
    if peak_alignment == "target_peak":
        return t[np.argmax(y)]
    if peak_alignment == "target_abs_peak":
        return t[np.argmax(np.abs(y))]
    if peak_alignment in ("global_peak", "global_abs_peak"):
        lc = example["lightcurve"]
        time = np.array(lc["time"])
        flux = np.array(lc["flux"])
        flux_err = np.array(lc["flux_err"])

        valid = (time > 0) * np.isfinite(time) * np.isfinite(flux) * np.isfinite(flux_err)
        if not np.any(valid):
            return None

        global_time = time[valid]
        global_flux = flux[valid]
        if peak_alignment == "global_peak":
            return global_time[np.argmax(global_flux)]
        return global_time[np.argmax(np.abs(global_flux))]

    raise ValueError(f"Unsupported peak_alignment: {peak_alignment}")


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

def bitweight_location(y, c=6.0, eps=1e-12, max_iter=50, tol=1e-8):
    """
    Estimate robust background location and scale.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]

    if len(y) == 0:
        return np.nan, np.nan
    
    # initial robust center and scale
    T = np.median(y)  
    S = np.median(np.abs(y - T)) * 1.4826   # scale estimate based on MAD, multiplied by 1.4826 for consistency with std under normal distribution

    if S < eps:
        q25, q75 = np.percentile(y, [25, 75])
        S = (q75 - q25) / 1.349
    if S < eps: 
        return T, S
    
    for _ in range(max_iter):
        u = (y - T) / (c * S)
        
        mask = np.abs(u) < 1   # only consider points within the cutoff for weighting; others get zero weight
        if not np.any(mask):
            return T, S
        
        # compute weights
        w = np.zeros_like(y)
        w[mask] = (1 - u[mask]**2)**2  # Tukey's biweight function,
        # It gives high weight to central points and low weight to moderately far points.

        T_new = np.sum(w * y) / np.sum(w)
        if abs(T_new - T) < tol:   # convergence check
            return T_new, S

        T = T_new

    return T, S


def _resolve_background_and_scale(
        y,
        flux_scale=None,
        subtract_background=False,
        background_flux=None,
        background_estimator=bitweight_location,
):
    if not subtract_background:
        return 0.0, flux_scale

    estimated_background, estimated_scale = background_estimator(y)
    if background_flux is not None:
        resolved_background = float(background_flux)
    else:
        resolved_background = float(estimated_background)
    if not np.isfinite(resolved_background):
        raise ValueError("Estimated background_flux is not finite.")

    resolved_scale = flux_scale
    if resolved_scale is None:
        resolved_scale = float(estimated_scale)
        if not np.isfinite(resolved_scale) or resolved_scale <= 0:
            raise ValueError("Estimated background scale is not finite and positive.")

    return resolved_background, resolved_scale


def inverse_transform_flux_from_gp(y_gp, data):
    """
    Transform GP-space flux back to raw flux.
    """
    return np.asarray(y_gp) * data["flux_scale"] + data.get("background_flux", 0.0)


def inverse_transform_flux_err_from_gp(yerr_gp, data):
    """
    Transform GP-space flux uncertainty back to raw flux uncertainty.
    """
    return np.asarray(yerr_gp) * data["flux_scale"]

    
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
        peak_alignment="target_peak",
        subtract_background=False,
        background_flux=None,
        background_estimator=bitweight_location,
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
        peak_alignment: "target_peak", "target_abs_peak", "global_peak", or "global_abs_peak".
        subtract_background: whether to subtract an estimated base flux before fitting the GP.
        background_flux: optional fixed base flux. If None and subtract_background=True, estimate it from the object's flux.
        background_estimator: callable used to estimate the base flux from raw flux values.
        normalize_flux: whether to normalize the flux values by the peak flux.
        min_points: minimum number of data points required to keep the example.
    Returns:
        A dictionary with processed time and flux arrays, or None if the example is discarded.
    """
    t, y, yerr = _extract_valid_band_observations(example, target_band)

    # Check if there are enough points
    if len(t) < min_points:
        return None

    background_flux, background_scale = _resolve_background_and_scale(
        y,
        flux_scale=flux_scale,
        subtract_background=subtract_background,
        background_flux=background_flux,
        background_estimator=background_estimator,
    )
    y_gp = y - background_flux

    # Align peak if required
    if align_peak:
        alignment_peak_time = _find_alignment_peak_time(
            example,
            target_band,
            t,
            y,
            peak_alignment,
        )
        if alignment_peak_time is None:
            return None
        t = t - alignment_peak_time
    else:
        alignment_peak_time = np.nan
        
    # Normalize time by its standard deviation to help with GP fitting
    t_scale = np.std(t)
    if t_scale <= 0:
        return None

    t = t / t_scale

    # Normalize flux if required
    if normalize_flux:
        if flux_scale is None:
            if subtract_background:
                if background_scale is None:
                    return None
                peak_flux = float(background_scale)
            else:
                peak_flux = float(np.max(np.abs(y_gp)))
            if peak_flux <= 0:
                return None
        else:
            peak_flux = float(flux_scale)
        resolved_flux_scale = peak_flux
        y = y_gp / resolved_flux_scale
        yerr = yerr / resolved_flux_scale
    else:
        # If not normalizing, we can still filter out examples with non-positive peak flux
        if np.max(np.abs(y_gp)) <= 0:
            return None
        peak_flux = 1.0
        resolved_flux_scale = 1.0
        y = y_gp
    
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
        't_peak': 0.0 if align_peak else t[np.argmax(y)],
        'target_peak_time': t[np.argmax(y)],
        'alignment_peak_time': alignment_peak_time,
        'peak_alignment': peak_alignment,
        'peak_flux': peak_flux,
        'flux_scale': resolved_flux_scale,
        'background_flux': background_flux,
        'subtract_background': subtract_background,
    }


def _make_processed_subset(
        example,
        target_band,
        t_raw,
        y_raw,
        yerr_raw,
        alignment_peak_time,
        t_scale,
        flux_scale,
        background_flux,
        subtract_background,
        peak_flux,
        peak_alignment,
        train_indices=None,
        heldout_indices=None,
):
    t = (t_raw - alignment_peak_time) / t_scale
    y = (y_raw - background_flux) / flux_scale
    yerr = yerr_raw / flux_scale
    X = t.reshape(-1, 1)

    data = {
        'X': X,
        't': t,
        'y': y,
        'yerr': yerr,
        'band': target_band,
        'obj_type': example['obj_type'],
        'obj_id': example['object_id'],
        't_peak': 0.0,
        'target_peak_time': (t_raw[np.argmax(y_raw)] - alignment_peak_time) / t_scale,
        'alignment_peak_time': alignment_peak_time,
        'peak_alignment': peak_alignment,
        'peak_flux': peak_flux,
        'flux_scale': flux_scale,
        'background_flux': background_flux,
        'subtract_background': subtract_background,
        't_scale': t_scale,
    }
    if train_indices is not None:
        data["train_indices"] = train_indices
    if heldout_indices is not None:
        data["heldout_indices"] = heldout_indices

    return data


def process_one_obj_one_band_train_heldout(
        example,
        flux_scale=None,
        target_band='r',
        align_peak=True,
        peak_alignment="target_peak",
        peak_mode="absolute",
        force_peak_in_train=True,
        subtract_background=False,
        background_flux=None,
        background_estimator=bitweight_location,
        normalize_flux=True,
        min_points=8,
        heldout_fraction=0.2,
        min_train_points=5,
        min_heldout_points=1,
        random_state=0,
        strategy="random",
):
    """
    Split raw observations first, then fit preprocessing from training data only.

    This avoids leakage from held-out points into peak alignment and time scaling.
    By default, the object's peak-flux observation is forced into train_data and
    excluded from heldout_data.
    Background and scale are fit from training data only, then reused for held-out data.
    """
    t_raw, y_raw, yerr_raw = _extract_valid_band_observations(example, target_band)
    if len(t_raw) < min_points:
        return None, None

    force_train_indices = (
        _find_peak_indices(y_raw, peak_mode=peak_mode)
        if force_peak_in_train
        else None
    )

    train_indices, heldout_indices = _split_indices(
        len(t_raw),
        heldout_fraction=heldout_fraction,
        min_train_points=min_train_points,
        min_heldout_points=min_heldout_points,
        random_state=random_state,
        strategy=strategy,
        force_train_indices=force_train_indices,
    )
    train_indices = train_indices[np.argsort(t_raw[train_indices])]
    heldout_indices = heldout_indices[np.argsort(t_raw[heldout_indices])]

    train_t_raw = t_raw[train_indices]
    train_y_raw = y_raw[train_indices]
    train_yerr_raw = yerr_raw[train_indices]
    heldout_t_raw = t_raw[heldout_indices]
    heldout_y_raw = y_raw[heldout_indices]
    heldout_yerr_raw = yerr_raw[heldout_indices]

    background_flux, background_scale = _resolve_background_and_scale(
        train_y_raw,
        flux_scale=flux_scale,
        subtract_background=subtract_background,
        background_flux=background_flux,
        background_estimator=background_estimator,
    )
    print(f"Resolved background_flux={background_flux}, background_scale={background_scale}")
    train_y_gp_raw = train_y_raw - background_flux

    if align_peak:
        # Use only target-band training observations to avoid leakage.
        alignment_peak_time = _find_alignment_peak_time(
            example,
            target_band,
            train_t_raw,
            train_y_raw,
            peak_alignment if not peak_alignment.startswith("global") else "target_abs_peak",
        )
        if alignment_peak_time is None:
            return None, None
    else:
        alignment_peak_time = 0.0

    train_t_centered = train_t_raw - alignment_peak_time
    t_scale = np.std(train_t_centered)
    if t_scale <= 0:
        return None, None

    if normalize_flux:
        if flux_scale is None:
            if subtract_background:
                if background_scale is None:
                    return None, None
                peak_flux = float(background_scale)
            else:
                peak_flux = float(np.max(np.abs(train_y_gp_raw)))
            if peak_flux <= 0:
                return None, None
        else:
            peak_flux = float(flux_scale)
        resolved_flux_scale = peak_flux
    else:
        if np.max(np.abs(train_y_gp_raw)) <= 0:
            return None, None
        peak_flux = 1.0
        resolved_flux_scale = 1.0

    train_data = _make_processed_subset(
        example,
        target_band,
        train_t_raw,
        train_y_raw,
        train_yerr_raw,
        alignment_peak_time,
        t_scale,
        resolved_flux_scale,
        background_flux,
        subtract_background,
        peak_flux,
        peak_alignment,
        train_indices=train_indices,
    )
    heldout_data = _make_processed_subset(
        example,
        target_band,
        heldout_t_raw,
        heldout_y_raw,
        heldout_yerr_raw,
        alignment_peak_time,
        t_scale,
        resolved_flux_scale,
        background_flux,
        subtract_background,
        peak_flux,
        peak_alignment,
        heldout_indices=heldout_indices,
    )

    return train_data, heldout_data
