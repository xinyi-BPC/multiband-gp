from itertools import islice
from pathlib import Path
from typing import Any, cast
import warnings

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from data_processing import (
    _extract_valid_band_observations,
    _find_peak_indices,
    _resolve_background_and_scale,
    _split_indices,
    bitweight_location,
    process_one_obj_one_band_train_heldout,
)
from singleGP_model import (
    evaluate_heldout_metrics,
    fit_basic_gp,
    inverse_transform_predictions,
    negative_log_predictive_density,
    summarize_object_metric_results,
)


# LSST-like effective wavelengths in nanometers. PLAsTiCC often stores bands
# as 0..5, so both string and integer aliases are accepted by the resolver.
DEFAULT_BAND_TO_WAVELENGTH = {
    "u": 367.1,
    "g": 482.7,
    "r": 622.3,
    "i": 754.6,
    "z": 869.1,
    0: 367.1,
    1: 482.7,
    2: 622.3,
    3: 754.6,
    4: 869.1,
}


def _resolve_wavelength(band, band_to_wavelength):
    """Resolve a band's configured effective wavelength."""
    if band in band_to_wavelength:
        return float(band_to_wavelength[band])

    band_str = str(band)
    if band_str in band_to_wavelength:
        return float(band_to_wavelength[band_str])

    try:
        band_int = int(band)
    except (TypeError, ValueError):
        band_int = None
    if band_int in band_to_wavelength:
        return float(band_to_wavelength[band_int])

    raise KeyError(f"No wavelength configured for band {band!r}.")


def _available_bands(example, bands, band_to_wavelength):
    if bands is None:
        lc_bands = np.asarray(example["lightcurve"]["band"])
        bands = list(dict.fromkeys(lc_bands.tolist()))

    usable = []
    for band in bands:
        try:
            _resolve_wavelength(band, band_to_wavelength)
        except KeyError:
            continue

        t, _, _ = _extract_valid_band_observations(example, band)
        if len(t) > 0:
            usable.append(band)

    return usable


def _rng_from_seed_or_rng(random_state=None, rng=None):
    if rng is not None:
        return rng
    return np.random.default_rng(random_state)


def _data_n_rows(data):
    """Determine the number of rows in the data based on the 'band' key."""
    return len(np.asarray(data["band"]))


def _subset_rows(data, indices):
    indices = np.asarray(indices, dtype=int)
    n_rows = _data_n_rows(data)
    subset = data.copy()
    for key, value in data.items():
        if isinstance(value, np.ndarray) and len(value) == n_rows:
            subset[key] = value[indices]
        elif isinstance(value, list) and len(value) == n_rows:
            subset[key] = [value[i] for i in indices]
    return subset


def _append_rows(first, second):
    """Append rows from second to first, concatenating arrays/lists of matching length."""
    first_n = _data_n_rows(first)
    merged = first.copy()
    for key, first_value in first.items():
        if key not in second:
            continue
        second_value = second[key]
        if isinstance(first_value, np.ndarray) and len(first_value) == first_n:
            merged[key] = np.concatenate([first_value, np.asarray(second_value)])
        elif isinstance(first_value, list) and len(first_value) == first_n:
            merged[key] = first_value + list(second_value)
    for key, second_value in second.items():
        if key not in merged:
            merged[key] = second_value
    return merged


def _stack_raw_mogp_subset(example, bands, split_by_band, index_key):
    """
    Stack the raw data across bands for the given subset (train or heldout) defined by index_key.
    """
    t_parts = []
    y_parts = []
    yerr_parts = []
    band_parts = []
    index_parts = []
    row_id_parts = []

    for band in bands:
        split = split_by_band[band]  
        indices = np.asarray(split[index_key], dtype=int)
        if len(indices) == 0:
            continue

        # subsampling will be done after stacking to preserve the original multiband alignment before reduction
        t_parts.append(split["t_raw"][indices])
        y_parts.append(split["y_raw"][indices])
        yerr_parts.append(split["yerr_raw"][indices])
        band_parts.append(np.full(len(indices), band, dtype=object))
        index_parts.append(indices)
        # Create row IDs in the format "band:index" to track original rows across bands and splits
        row_id_parts.append(np.asarray([f"{band}:{idx}" for idx in indices], dtype=object))

    if len(t_parts) == 0:
        return None

    # Stack the raw data across all bands
    t_raw = np.concatenate(t_parts)
    y_raw = np.concatenate(y_parts)
    yerr_raw = np.concatenate(yerr_parts)
    band = np.concatenate(band_parts)
    source_indices = np.concatenate(index_parts)
    row_id = np.concatenate(row_id_parts)

    order = np.lexsort((band.astype(str), t_raw))
    return {
        "t_raw": t_raw[order],
        "y_raw": y_raw[order],
        "yerr_raw": yerr_raw[order],
        "band": band[order],
        "source_indices": source_indices[order],
        "row_id": row_id[order],
        "obj_type": example["obj_type"],
        "obj_id": example["object_id"],
    }


def subsample_multiband_train_to_reference_band(
        train_data,
        test_data,
        reference_band,
        random_state=0,
        rng=None,
        preserve_peak=True,
):
    """
    Reduce multiband train rows after the original split for fair comparison.

    The reduced train size equals the number of original training rows in
    reference_band. Non-selected original train rows are appended to test_data.
    If preserve_peak=True, exactly one row with the maximum raw training flux
    across all bands is forced into the reduced train set.
    """

    "Count how many training observations are in the reference band. "
    "Use that number as the target size for the reduced multiband training set."
    train_band = np.asarray(train_data["band"], dtype=object)
    n_reference_train = int(np.sum(train_band == reference_band))
    if n_reference_train <= 0:
        raise ValueError(f"reference_band={reference_band!r} has no training rows.")

    n_train = _data_n_rows(train_data)
    if n_reference_train > n_train:
        raise ValueError("Reference-band training count cannot exceed total training count.")

    rng = _rng_from_seed_or_rng(random_state=random_state, rng=rng)
    all_indices = np.arange(n_train)

    if preserve_peak:
        peak_index = int(np.argmax(np.asarray(train_data["y_raw"], dtype=float)))
        remaining_needed = n_reference_train - 1
        candidate_indices = np.setdiff1d(all_indices, [peak_index], assume_unique=True)
        if remaining_needed > len(candidate_indices):
            raise ValueError("Not enough non-peak rows to complete the reduced training set.")
        sampled = (
            np.array([], dtype=int)
            if remaining_needed == 0
            else rng.choice(candidate_indices, size=remaining_needed, replace=False)
        )
        selected_indices = np.sort(np.concatenate([[peak_index], sampled]))
    else:
        selected_indices = np.sort(rng.choice(all_indices, size=n_reference_train, replace=False))
        peak_index = None

    moved_indices = np.setdiff1d(all_indices, selected_indices, assume_unique=True)
    reduced_train = _subset_rows(train_data, selected_indices)
    moved_to_test = _subset_rows(train_data, moved_indices)
    reduced_test = _append_rows(test_data, moved_to_test)

    selected_row_ids = set(np.asarray(reduced_train.get("row_id", []), dtype=object).tolist())
    moved_row_ids = set(np.asarray(moved_to_test.get("row_id", []), dtype=object).tolist())
    if selected_row_ids and selected_row_ids.intersection(moved_row_ids):
        raise AssertionError("Reduced train and moved-to-test rows overlap.")

    peak_row_id = None if peak_index is None else train_data.get("row_id", [None] * n_train)[peak_index]
    reduced_train["subsample_metadata"] = {
        "reference_band": reference_band,
        "n_reference_train": n_reference_train,
        "selected_indices": selected_indices,
        "moved_indices": moved_indices,
        "preserve_peak": preserve_peak,
        "peak_index": peak_index,
        "peak_row_id": peak_row_id,
    }
    reduced_test["subsample_metadata"] = reduced_train["subsample_metadata"]

    # Sanity checks to ensure the subsampling logic is correct and consistent
    if _data_n_rows(reduced_train) != n_reference_train:
        raise AssertionError("Reduced multiband train size does not match reference-band train size.")
    if preserve_peak and peak_row_id not in set(np.asarray(reduced_train.get("row_id", []), dtype=object).tolist()):
        raise AssertionError("Preserved peak row is missing from reduced train data.")

    return reduced_train, reduced_test


def _background_for_training_flux(
        train_by_band,
        subtract_background=False,
        background_flux=None,
        background_estimator=bitweight_location,
        background_mode="object",
):
    if not subtract_background:
        return {band: 0.0 for band in train_by_band}

    # If background_mode is "band", estimate a separate background_flux for each band using the provided estimator.
    if background_mode == "band":
        backgrounds = {}
        for band, arrays in train_by_band.items():
            if background_flux is None:
                background = float(background_estimator(arrays["y_raw"])[0])
            else:
                background = float(background_flux)
            if not np.isfinite(background):
                raise ValueError("Estimated band background_flux is not finite.")
            backgrounds[band] = background
        return backgrounds

    if background_mode != "object":
        raise ValueError("background_mode must be 'object' or 'band'.")

    # Estimate a single background_flux for the entire object using all training fluxes across bands.
    y_train = np.concatenate([arrays["y_raw"] for arrays in train_by_band.values()])
    if background_flux is None:
        background = float(background_estimator(y_train)[0])
    else:
        background = float(background_flux)
    if not np.isfinite(background):
        raise ValueError("Estimated object background_flux is not finite.")

    # return the same background for all bands if background_mode is "object"
    return {band: background for band in train_by_band}


def _stack_mogp_subset(
        example,
        bands,
        split_by_band,
        index_key,
        band_to_wavelength,
        alignment_peak_time,
        t_scale,
        flux_scale,
        backgrounds,
        subtract_background,
        background_mode,
        scale_mode,
        peak_alignment,
):
    t_parts = []
    wavelength_parts = []
    y_parts = []
    yerr_parts = []
    y_raw_parts = []
    yerr_raw_parts = []
    band_parts = []
    index_parts = []

    for band in bands:
        split = split_by_band[band]
        indices = np.asarray(split[index_key], dtype=int)
        if len(indices) == 0:
            continue

        t_raw = split["t_raw"][indices]
        y_raw = split["y_raw"][indices]
        yerr_raw = split["yerr_raw"][indices]
        background = backgrounds[band]
        wavelength = _resolve_wavelength(band, band_to_wavelength)

        t = (t_raw - alignment_peak_time) / t_scale
        y = (y_raw - background) / flux_scale
        yerr = yerr_raw / flux_scale

        t_parts.append(t)
        wavelength_parts.append(np.full(len(t), wavelength, dtype=float))
        y_parts.append(y)
        yerr_parts.append(yerr)
        y_raw_parts.append(y_raw)
        yerr_raw_parts.append(yerr_raw)
        band_parts.append(np.full(len(t), band, dtype=object))
        index_parts.append(indices)

    if len(t_parts) == 0:
        return None

    t = np.concatenate(t_parts)
    wavelength = np.concatenate(wavelength_parts)
    y = np.concatenate(y_parts)
    yerr = np.concatenate(yerr_parts)
    y_raw = np.concatenate(y_raw_parts)
    yerr_raw = np.concatenate(yerr_raw_parts)
    band = np.concatenate(band_parts)
    source_indices = np.concatenate(index_parts)

    order = np.lexsort((wavelength, t))
    t = t[order]
    wavelength = wavelength[order]
    y = y[order]
    yerr = yerr[order]
    y_raw = y_raw[order]
    yerr_raw = yerr_raw[order]
    band = band[order]
    source_indices = source_indices[order]

    X = np.column_stack([t, wavelength])
    point_background = np.asarray([backgrounds[b] for b in band], dtype=float)

    return {
        "X": X,
        "t": t,
        "wavelength": wavelength,
        "y": y,
        "yerr": yerr,
        "y_raw": y_raw,
        "yerr_raw": yerr_raw,
        "band": band,
        "source_indices": source_indices,
        "obj_type": example["obj_type"],
        "obj_id": example["object_id"],
        "alignment_peak_time": alignment_peak_time,
        "peak_alignment": peak_alignment,
        "flux_scale": flux_scale,
        "scale": flux_scale,
        "scale_mode": scale_mode,
        "background_flux": point_background,
        "background_by_band": backgrounds,
        "subtract_background": subtract_background,
        "background_mode": background_mode,
        "t_scale": t_scale,
    }


def process_one_obj_mogp_train_heldout(
        example,
        bands=("u", "g", "r", "i", "z", 0, 1, 2, 3, 4),
        band_to_wavelength=None,
        flux_scale=None,
        align_peak=True,
        peak_alignment="global_abs_peak",
        peak_mode="absolute",
        force_peak_in_train=True,
        subtract_background=False,
        background_flux=None,
        background_estimator=bitweight_location,
        background_mode="band",
        scale_mode="local_peak",
        local_flux_percentile=95,
        scale_eps=1e-12,
        normalize_flux=True,
        min_points_per_band=0,
        min_total_points=8,
        heldout_fraction=0.2,
        min_train_points=5,
        min_heldout_points=1,
        random_state=0,
        strategy="random",
        subsample_reference_band=None,
        preserve_peak=True,
        subsample_random_state=None,
):
    """
    Split each band with the same helper used by the single-band GP, optionally
    reduce the multiband train set after that split, then fit preprocessing from
    training data only. This is a continuous time+wavelength GP baseline, not a
    full ICM/LMC coregionalization model.
    """
    band_to_wavelength = band_to_wavelength or DEFAULT_BAND_TO_WAVELENGTH
    usable_bands = _available_bands(example, bands, band_to_wavelength)

    split_by_band = {}
    for band in usable_bands:
        t_raw, y_raw, yerr_raw = _extract_valid_band_observations(example, band)
        if len(t_raw) < max(min_points_per_band, min_train_points + min_heldout_points):
            continue

        finite = np.isfinite(t_raw) & np.isfinite(y_raw) & np.isfinite(yerr_raw) & (yerr_raw >= 0)
        t_raw = t_raw[finite]
        y_raw = y_raw[finite]
        yerr_raw = yerr_raw[finite]
        if len(t_raw) < max(min_points_per_band, min_train_points + min_heldout_points):
            continue

        force_train_indices = (
            _find_peak_indices(y_raw, peak_mode=peak_mode)
            if force_peak_in_train
            else None
        )
        try:
            train_indices, heldout_indices = _split_indices(
                len(t_raw),
                heldout_fraction=heldout_fraction,
                min_train_points=min_train_points,
                min_heldout_points=min_heldout_points,
                random_state=random_state,
                strategy=strategy,
                force_train_indices=force_train_indices,
            )
        except ValueError:
            continue

        train_indices = train_indices[np.argsort(t_raw[train_indices])]
        heldout_indices = heldout_indices[np.argsort(t_raw[heldout_indices])]
        split_by_band[band] = {  
            "t_raw": t_raw,
            "y_raw": y_raw,
            "yerr_raw": yerr_raw,
            "train_indices": train_indices,
            "heldout_indices": heldout_indices,
        }

    if len(split_by_band) == 0:
        return None, None

    n_train_total = sum(len(v["train_indices"]) for v in split_by_band.values())
    n_heldout_total = sum(len(v["heldout_indices"]) for v in split_by_band.values())
    if n_train_total < min_train_points or n_heldout_total < min_heldout_points:
        return None, None
    if n_train_total + n_heldout_total < min_total_points:
        return None, None

    if scale_mode is None:
        scale_mode = "background_scale" if subtract_background else "local_peak"

    # Subsample the training and heldout data to match the reference band
    if subsample_reference_band is not None:
        raw_train_data = _stack_raw_mogp_subset(
            example,
            list(split_by_band.keys()),
            split_by_band,
            "train_indices",
        )
        raw_heldout_data = _stack_raw_mogp_subset(
            example,
            list(split_by_band.keys()),
            split_by_band,
            "heldout_indices",
        )
        if raw_train_data is None or raw_heldout_data is None:
            return None, None

        reduced_raw_train, reduced_raw_heldout = subsample_multiband_train_to_reference_band(
            raw_train_data,
            raw_heldout_data,
            reference_band=subsample_reference_band,
            random_state=random_state if subsample_random_state is None else subsample_random_state,
            preserve_peak=preserve_peak,
        )

        for band, split in split_by_band.items():
            train_mask = np.asarray(reduced_raw_train["band"], dtype=object) == band
            heldout_mask = np.asarray(reduced_raw_heldout["band"], dtype=object) == band
            split["train_indices"] = np.sort(
                np.asarray(reduced_raw_train["source_indices"], dtype=int)[train_mask]
            )
            split["heldout_indices"] = np.sort(
                np.asarray(reduced_raw_heldout["source_indices"], dtype=int)[heldout_mask]
            )

    train_by_band = {
        band: {
            "t_raw": split["t_raw"][split["train_indices"]],
            "y_raw": split["y_raw"][split["train_indices"]],
            "yerr_raw": split["yerr_raw"][split["train_indices"]],
        }
        for band, split in split_by_band.items()
        if len(split["train_indices"]) > 0
    }
    if len(train_by_band) == 0:
        return None, None
    n_train_total = sum(len(v["train_indices"]) for v in split_by_band.values())
    n_heldout_total = sum(len(v["heldout_indices"]) for v in split_by_band.values())
    if n_train_total < min_train_points or n_heldout_total < min_heldout_points:
        return None, None

    # Estimate backgrounds for each band based on the training fluxes.
    backgrounds = _background_for_training_flux(
        train_by_band,
        subtract_background=subtract_background,
        background_flux=background_flux,
        background_estimator=background_estimator,
        background_mode=background_mode,
    )
    # Determine fallback background based on available estimates
    if background_flux is not None:
        fallback_background = float(background_flux)
    else:
        fallback_background = 0.0
    for band in split_by_band:
        if band not in backgrounds:
            backgrounds[band] = fallback_background if background_mode == "object" else 0.0

    train_t_raw_all = np.concatenate([v["t_raw"] for v in train_by_band.values()])
    train_y_centered_all = np.concatenate([
        v["y_raw"] - backgrounds[band]
        for band, v in train_by_band.items()
    ])

    if align_peak:
        peak_idx = int(np.argmax(np.abs(train_y_centered_all)))
        alignment_peak_time = float(train_t_raw_all[peak_idx])
    else:
        alignment_peak_time = 0.0

    train_t_centered = train_t_raw_all - alignment_peak_time
    t_scale = float(np.std(train_t_centered))
    if not np.isfinite(t_scale) or t_scale <= 0:
        return None, None

    if normalize_flux:
        if scale_mode == "global":   # Use the provided flux_scale directly as the global scale for all bands.
            if flux_scale is None:
                raise ValueError("scale_mode='global' requires flux_scale for MOGP.")
            resolved_flux_scale = float(flux_scale)
        elif scale_mode in ("local_peak", "background_scale"):
            _, resolved_flux_scale = _resolve_background_and_scale(
                train_y_centered_all,
                flux_scale=flux_scale,
                subtract_background=False,
                background_flux=None,
                background_estimator=background_estimator,
                scale_mode=scale_mode,
                local_flux_percentile=local_flux_percentile,
                eps=scale_eps,
            )
        else:
            raise ValueError(f"Unsupported scale_mode for MOGP: {scale_mode}")

        if not np.isfinite(resolved_flux_scale) or resolved_flux_scale <= 0:
            return None, None
    else:
        resolved_flux_scale = 1.0

    train_data = _stack_mogp_subset(
        example,
        list(split_by_band.keys()),
        split_by_band,
        "train_indices",
        band_to_wavelength,
        alignment_peak_time,
        t_scale,
        resolved_flux_scale,
        backgrounds,
        subtract_background,
        background_mode,
        scale_mode,
        peak_alignment,
    )
    heldout_data = _stack_mogp_subset(
        example,
        list(split_by_band.keys()),
        split_by_band,
        "heldout_indices",
        band_to_wavelength,
        alignment_peak_time,
        t_scale,
        resolved_flux_scale,
        backgrounds,
        subtract_background,
        background_mode,
        scale_mode,
        peak_alignment,
    )
    if train_data is None or heldout_data is None:
        return None, None

    train_data["train_indices_by_band"] = {
        band: split["train_indices"] for band, split in split_by_band.items()
    }
    heldout_data["heldout_indices_by_band"] = {
        band: split["heldout_indices"] for band, split in split_by_band.items()
    }
    if subsample_reference_band is not None:
        train_data["subsample_reference_band"] = subsample_reference_band
        heldout_data["subsample_reference_band"] = subsample_reference_band

    return train_data, heldout_data


def find_reduced_mogp_train_global_percentile_flux_peak(
        examples,
        reference_band,
        bands=("u", "g", "r", "i", "z", 0, 1, 2, 3, 4),
        percentile=95,
        n_objects=None,
        max_examples_to_scan=3000,
        random_state=0,
        preserve_peak=True,
        subtract_background=False,
        background_flux=None,
        background_estimator=bitweight_location,
        background_mode="object",
        **process_kwargs,
):
    """
    Compute a MOGP global flux scale from reduced multiband training rows only.

    This is the multiband counterpart to the single-band train-only global
    scale helper. It first performs the normal per-band split, then reduces
    each object's multiband train set to the reference-band train count, and
    finally pools only those reduced training fluxes.
    """
    flux_values = []
    scanned = 0
    used = 0
    process_kwargs = {
        key: value
        for key, value in process_kwargs.items()
        if key not in {
            "random_state",
            "subsample_reference_band",
            "preserve_peak",
            "subtract_background",
            "background_flux",
            "background_estimator",
            "background_mode",
            "normalize_flux",
            "scale_mode",
            "flux_scale",
        }
    }

    for object_idx, example in enumerate(examples):
        if object_idx >= max_examples_to_scan:
            break
        if n_objects is not None and used >= n_objects:
            break
        scanned = object_idx + 1

        train_data, _ = process_one_obj_mogp_train_heldout(
            example,
            bands=bands,
            random_state=random_state + object_idx,
            subsample_reference_band=reference_band,
            preserve_peak=preserve_peak,
            subtract_background=subtract_background,
            background_flux=background_flux,
            background_estimator=background_estimator,
            background_mode=background_mode,
            normalize_flux=False,   # disable internal normalization to get raw flux values for percentile calculation
            scale_mode="local_peak",
            **process_kwargs,
        )
        if train_data is None:
            continue

        centered_flux = np.asarray(train_data["y_raw"], dtype=float) - np.asarray(
            train_data.get("background_flux", 0.0),
            dtype=float,
        )
        flux_values.append(np.abs(centered_flux))
        used += 1

    if len(flux_values) == 0:
        return None

    flux_values = np.concatenate(flux_values)
    scale = float(np.percentile(flux_values, percentile))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Computed reduced MOGP train-only global scale is not finite and positive.")

    return {
        "flux_scale": scale,
        "percentile": percentile,
        "n_objects": used,
        "scanned_examples": scanned,
        "reference_band": reference_band,
    }


def fit_mogp_gp(
        data,
        time_length_scale=0.3,
        wavelength_length_scale=600.0,
        time_length_scale_bounds=(0.05, 10.0),
        wavelength_length_scale_bounds=(20.0, 2000.0),
        constant_value=1.0,
        constant_value_bounds=(1e-2, 1e2),
        yerr_scale=1.0,
        noise_floor=0.0,
        jitter=1e-8,
        n_restarts_optimizer=2,
        random_state=0,
        print_kernel=True,
):
    """Fit the minimal continuous time+wavelength GP baseline."""
    kernel = ConstantKernel(
        constant_value,
        constant_value_bounds,
    ) * Matern(
        length_scale=[time_length_scale, wavelength_length_scale],
        length_scale_bounds=cast(
            Any,
            [time_length_scale_bounds, wavelength_length_scale_bounds],
        ),
        nu=1.5,
    )

    yerr_scaled = np.asarray(data["yerr"], dtype=float)
    alpha = (yerr_scale * yerr_scaled) ** 2 + noise_floor ** 2 + jitter
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=False,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=random_state,
    )
    gp.fit(np.asarray(data["X"], dtype=float), np.asarray(data["y"], dtype=float))
    if print_kernel:
        print(f"MOGP learned kernel: {gp.kernel_}")
    return gp


def predict_mogp_observation_distribution(
        gp,
        data,
        include_yerr=True,
        yerr_scale=1.0,
        noise_floor=0.0,
        return_raw_flux=False,
):
    mean_norm, latent_std_norm = gp.predict(data["X"], return_std=True)
    variance_norm = np.maximum(latent_std_norm ** 2, 0.0)
    if include_yerr:
        variance_norm = variance_norm + (yerr_scale * np.asarray(data["yerr"])) ** 2
    if noise_floor is not None and noise_floor > 0:
        variance_norm = variance_norm + noise_floor ** 2

    if not return_raw_flux:
        return mean_norm, np.sqrt(np.maximum(variance_norm, 0.0)), variance_norm

    flux_scale = float(data["flux_scale"])
    background = np.asarray(data.get("background_flux", 0.0), dtype=float)
    mean_raw = mean_norm * flux_scale + background
    variance_raw = variance_norm * flux_scale ** 2

    return mean_raw, np.sqrt(np.maximum(variance_raw, 0.0)), variance_raw


def _assert_mogp_z_score_invariance(y_norm, mean_norm, std_norm, y_raw, mean_raw, std_raw, scale):
    z_norm = (np.asarray(y_norm) - np.asarray(mean_norm)) / np.maximum(std_norm, 1e-12)
    z_raw = (np.asarray(y_raw) - np.asarray(mean_raw)) / np.maximum(std_raw, scale * 1e-12)
    if not np.allclose(z_norm, z_raw, rtol=1e-5, atol=1e-5):
        max_diff = float(np.max(np.abs(z_norm - z_raw)))
        raise AssertionError(f"MOGP raw and normalized z-scores differ: max_abs_diff={max_diff:g}")


def evaluate_mogp_heldout_metrics(
        gp,
        heldout_data,
        train_data=None,
        coverage_sigmas=(1.0, 2.0, 3.0),
        include_yerr=True,
        yerr_scale=1.0,
        noise_floor=0.0,
        evaluate_raw_metrics=True,
        assert_z_invariance=True,
):
    mean_norm, std_norm, variance_norm = predict_mogp_observation_distribution(
        gp,
        heldout_data,
        include_yerr=include_yerr,
        yerr_scale=yerr_scale,
        noise_floor=noise_floor,
        return_raw_flux=False,
    )
    y_norm = np.asarray(heldout_data["y"])
    errors_norm = y_norm - mean_norm

    coverage = {}
    coverage_counts = {}
    for sigma in coverage_sigmas:
        covered = np.abs(errors_norm) <= sigma * std_norm
        key = f"coverage_{sigma:g}sigma"
        coverage[key] = float(np.mean(covered))
        coverage_counts[key] = int(np.sum(covered))

    scale = float(heldout_data["flux_scale"])
    background = np.asarray(heldout_data.get("background_flux", 0.0), dtype=float)
    mean_raw, variance_raw = inverse_transform_predictions(mean_norm, variance_norm, scale, 0.0)
    mean_raw = mean_raw + background
    std_raw = np.sqrt(np.maximum(variance_raw, 0.0))
    y_raw = np.asarray(heldout_data["y_raw"])
    yerr_raw = np.asarray(heldout_data["yerr_raw"])

    if assert_z_invariance:
        _assert_mogp_z_score_invariance(
            y_norm,
            mean_norm,
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
        y_metric = y_norm
        mean_metric = mean_norm
        std_metric = std_norm
        variance_metric = variance_norm
        yerr_metric = np.asarray(heldout_data["yerr"])
        metric_space = "normalized"

    squared_errors = (y_metric - mean_metric) ** 2
    per_point_nlpd = negative_log_predictive_density(y_metric, mean_metric, variance_metric)

    n_heldout = len(y_metric)
    if train_data is not None:
        train_t = np.asarray(train_data["t"])
        train_time_min = float(np.min(train_t))
        train_time_max = float(np.max(train_t))
        outside_train_range = (heldout_data["t"] < train_time_min) | (heldout_data["t"] > train_time_max)
        distance_to_train_range = np.maximum.reduce([
            train_time_min - heldout_data["t"],
            heldout_data["t"] - train_time_max,
            np.zeros_like(heldout_data["t"]),
        ])
        n_train = len(train_t)
        all_t = np.concatenate([train_data["t"], heldout_data["t"]])
        all_y = np.concatenate([train_data["y"], heldout_data["y"]])
        peak_time = all_t[np.argmax(np.abs(all_y))]
    else:
        train_time_min = np.nan
        train_time_max = np.nan
        outside_train_range = np.full(n_heldout, False)
        distance_to_train_range = np.full(n_heldout, np.nan)
        n_train = None
        peak_time = heldout_data["t"][np.argmax(np.abs(y_norm))]
    near_peak = np.abs(heldout_data["t"] - peak_time) <= 0.25

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
        "y_true_norm": y_norm,
        "y_pred_norm": mean_norm,
        "y_std_norm": std_norm,
        "predictive_variance_norm": variance_norm,
        "y_true_raw": y_raw,
        "y_pred_raw": mean_raw,
        "y_std_raw": std_raw,
        "yerr_raw": yerr_raw,
        "time": np.asarray(heldout_data["t"]),
        "X_test": np.asarray(heldout_data["X"]).reshape(n_heldout, -1),
        "object_id": np.repeat(heldout_data.get("obj_id", None), n_heldout),
        "band": np.asarray(heldout_data["band"], dtype=object),
        "outside_train_range": outside_train_range,
        "distance_to_train_range": distance_to_train_range,
        "near_peak": near_peak,
        "predictive_variance": variance_metric,
        "flux_scale": scale,
        "background_flux": background,
    }


def _band_equal_mask(values, band):
    values = np.asarray(values, dtype=object)
    mask = values == band
    mask = mask | (values.astype(str) == str(band))
    try:
        band_int = int(band)
    except (TypeError, ValueError):
        return mask
    numeric_mask = []
    for value in values:
        try:
            numeric_mask.append(int(value) == band_int)
        except (TypeError, ValueError):
            numeric_mask.append(False)
    return mask | np.asarray(numeric_mask)


def _split_raw_band_for_ablation(
        example,
        band,
        heldout_fraction=0.2,
        min_train_points=5,
        min_heldout_points=1,
        random_state=0,
        strategy="random",
        force_peak_in_train=True,
        peak_mode="absolute",
):
    """Split the raw observations of a single band into training and heldout sets for ablation."""
    t_raw, y_raw, yerr_raw = _extract_valid_band_observations(example, band)
    if len(t_raw) < min_train_points + min_heldout_points:
        return None

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

    return {
        "band": band,
        "t_raw": t_raw,
        "y_raw": y_raw,
        "yerr_raw": yerr_raw,
        "train_indices": train_indices,
        "heldout_indices": heldout_indices,
    }


def _raw_rows_from_split(split, indices):
    """Extract raw rows corresponding to the specified indices from a band split dict."""
    indices = np.asarray(indices, dtype=int)
    band = split["band"]
    return {
        "t_raw": split["t_raw"][indices],
        "y_raw": split["y_raw"][indices],
        "yerr_raw": split["yerr_raw"][indices],
        "band": np.full(len(indices), band, dtype=object),
        "source_indices": indices,
        "row_id": np.asarray([f"{band}:{idx}" for idx in indices], dtype=object),
    }


def _concat_raw_row_groups(groups):
    groups = [group for group in groups if group is not None and len(group["t_raw"]) > 0]
    if len(groups) == 0:
        return None

    data = {
        "t_raw": np.concatenate([group["t_raw"] for group in groups]),
        "y_raw": np.concatenate([group["y_raw"] for group in groups]),
        "yerr_raw": np.concatenate([group["yerr_raw"] for group in groups]),
        "band": np.concatenate([group["band"] for group in groups]),
        "source_indices": np.concatenate([group["source_indices"] for group in groups]),
        "row_id": np.concatenate([group["row_id"] for group in groups]),
    }
    order = np.lexsort((data["band"].astype(str), data["t_raw"]))
    return {key: value[order] for key, value in data.items()}


def _sample_auxiliary_train_indices(split, ratio, rng):
    """Sample a subset of the auxiliary band train indices based on the specified ratio."""
    available = np.asarray(split["train_indices"], dtype=int)
    if ratio <= 0 or len(available) == 0:
        return np.array([], dtype=int)
    if ratio >= 1:
        return available
    n_select = int(np.floor(len(available) * ratio))
    if n_select <= 0:
        return np.array([], dtype=int)
    return np.sort(rng.choice(available, size=n_select, replace=False))


def _build_mogp_data_from_raw_rows(
        example,
        raw_rows,
        reference_processed_data,
        band_to_wavelength,
        wavelength_override=None,
):
    """
    Build data dicts for MOGP training or evaluation from raw rows and reference processed data.
    example: the original example dict for the object, used for metadata like obj_id and obj_type.
    raw_rows: dict with keys "t_raw", "y_raw", "yerr_raw",
                "band", "source_indices", "row_id", each containing arrays of the same length.
    reference_processed_data: the processed data dict from the reference split, used for metadata and scaling factors.
    band_to_wavelength: dict mapping band identifiers to wavelengths, used to compute the wavelength feature for MOGP.
    wavelength_override: optional dict mapping band identifiers to wavelengths, used to override the band_to_wavelength mapping for specific bands.
    """
    if raw_rows is None or len(raw_rows["t_raw"]) == 0:
        return None

    band_to_wavelength = band_to_wavelength or DEFAULT_BAND_TO_WAVELENGTH
    bands = np.asarray(raw_rows["band"], dtype=object)
    wavelength = np.asarray([
        wavelength_override[i]
        if wavelength_override is not None
        else _resolve_wavelength(band, band_to_wavelength)
        for i, band in enumerate(bands)
    ], dtype=float)

    t_scale = float(reference_processed_data.get("t_scale", 1.0))
    alignment_peak_time = float(reference_processed_data.get("alignment_peak_time", 0.0))
    flux_scale = float(reference_processed_data["flux_scale"])
    background = float(np.asarray(reference_processed_data.get("background_flux", 0.0)).reshape(-1)[0])
    t = (np.asarray(raw_rows["t_raw"], dtype=float) - alignment_peak_time) / t_scale
    y = (np.asarray(raw_rows["y_raw"], dtype=float) - background) / flux_scale
    yerr = np.asarray(raw_rows["yerr_raw"], dtype=float) / flux_scale

    return {
        "X": np.column_stack([t, wavelength]),
        "t": t,
        "wavelength": wavelength,
        "y": y,
        "yerr": yerr,
        "y_raw": np.asarray(raw_rows["y_raw"], dtype=float),
        "yerr_raw": np.asarray(raw_rows["yerr_raw"], dtype=float),
        "band": bands,
        "source_indices": np.asarray(raw_rows["source_indices"], dtype=int),
        "row_id": np.asarray(raw_rows["row_id"], dtype=object),
        "obj_type": example["obj_type"],
        "obj_id": example["object_id"],
        "alignment_peak_time": alignment_peak_time,
        "peak_alignment": reference_processed_data.get("peak_alignment", "target_peak"),
        "flux_scale": flux_scale,
        "scale": flux_scale,
        "scale_mode": reference_processed_data.get("scale_mode", "target_reference"),
        "background_flux": np.full(len(t), background, dtype=float),
        "background_by_band": {band: background for band in np.unique(bands)},
        "subtract_background": reference_processed_data.get("subtract_background", False),
        "background_mode": "target_reference",
        "t_scale": t_scale,
    }


def _metrics_row_from_result(model_name, metrics, train_data, target_band, heldout_indices, notes=None):
    y_true = np.asarray(metrics["y_true"], dtype=float)
    y_pred = np.asarray(metrics["y_pred"], dtype=float)
    y_std = np.maximum(np.asarray(metrics["y_std"], dtype=float), 1e-12)
    z = (y_true - y_pred) / y_std
    train_band = np.asarray(train_data["band"], dtype=object)
    if train_band.ndim == 0:
        train_band = np.repeat(train_band.item(), len(train_data["y"]))
    target_mask = _band_equal_mask(train_band, target_band)
    coverage = metrics["coverage"]
    return {
        "model": model_name,
        "object_id": metrics["object_id"][0] if len(metrics["object_id"]) else None,
        "target_band": target_band,
        "n_train_target_band": int(np.sum(target_mask)),
        "n_train_other_bands": int(len(train_band) - np.sum(target_mask)),
        "n_heldout_target_band": int(metrics["n_heldout"]),
        "heldout_indices": np.asarray(heldout_indices, dtype=int),
        "rmse": float(metrics["rmse"]),
        "nlpd": float(metrics["mean_nlpd"]),
        "coverage_1sigma": float(coverage["coverage_1sigma"]),
        "coverage_2sigma": float(coverage["coverage_2sigma"]),
        "coverage_3sigma": float(coverage["coverage_3sigma"]),
        "z_score_mean": float(np.mean(z)),
        "z_score_std": float(np.std(z)),
        "notes": notes,
    }


def _evaluate_mogp_on_target_heldout(gp, target_heldout_data, train_data):
    metrics = evaluate_mogp_heldout_metrics(
        gp,
        target_heldout_data,
        train_data=train_data,
        assert_z_invariance=True,
    )
    if not np.all(_band_equal_mask(metrics["band"], target_heldout_data["band"][0])):
        raise AssertionError("MOGP evaluation contains non-target held-out rows.")
    return metrics


def run_target_band_ablation_study(
        example,
        target_band="r",
        bands=("u", "g", "r", "i", "z", 0, 1, 2, 3, 4),
        aux_band_ratios=None,
        shuffle_repeats=3,
        random_state=0,
        heldout_fraction=0.2,
        min_train_points=5,
        min_heldout_points=1,
        force_peak_in_train=True,
        peak_mode="absolute",
        strategy="random",
        band_to_wavelength=None,
        include_same_total_train_budget=True,
        single_gp_kwargs=None,
        mogp_gp_kwargs=None,
):
    """
    Run target-band ablations separating training-set size from cross-band covariance.

    All models use one fixed target-band held-out split.  Models B-E evaluate
    only those same target-band held-out points through the MOGP input format.
    Model D is implemented as the small-wavelength-length-scale control, which
    approximates a block-diagonal independent-band kernel for well-separated
    band wavelengths.
    """
    band_to_wavelength = band_to_wavelength or DEFAULT_BAND_TO_WAVELENGTH
    aux_band_ratios = aux_band_ratios or {}
    single_gp_kwargs = single_gp_kwargs or {}
    mogp_gp_kwargs = mogp_gp_kwargs or {}
    rng = np.random.default_rng(random_state)

    # Create the single-band GP dictionaries for the target band
    target_train_s, target_heldout_s = process_one_obj_one_band_train_heldout(
        example,
        target_band=target_band,
        subtract_background=False,   # disable background subtraction for single-band GP to avoid confounding effects of background estimation with the ablation of auxiliary bands
        heldout_fraction=heldout_fraction,
        min_train_points=min_train_points,
        min_heldout_points=min_heldout_points,
        random_state=random_state,
        strategy=strategy,
        force_peak_in_train=force_peak_in_train,
        peak_mode=peak_mode,
    )
    if target_train_s is None or target_heldout_s is None:
        raise ValueError(f"Could not create target-band split for {target_band!r}.")

    # Recreate the same target-band split in raw format for MOGP
    target_split = _split_raw_band_for_ablation(
        example,
        target_band,
        heldout_fraction=heldout_fraction,
        min_train_points=min_train_points,
        min_heldout_points=min_heldout_points,
        random_state=random_state,
        strategy=strategy,
        force_peak_in_train=force_peak_in_train,
        peak_mode=peak_mode,
    )
    if target_split is None:
        raise ValueError(f"Could not recreate raw target-band split for {target_band!r}.")
    # Assert that the MOGP ablation uses exactly the same indices as the single-band GP
    if not np.array_equal(target_train_s["train_indices"], target_split["train_indices"]):
        raise AssertionError("Single-GP and ablation target train indices differ.")
    if not np.array_equal(target_heldout_s["heldout_indices"], target_split["heldout_indices"]):
        raise AssertionError("Single-GP and ablation target held-out indices differ.")

    target_train_rows = _raw_rows_from_split(target_split, target_split["train_indices"])
    target_heldout_rows = _raw_rows_from_split(target_split, target_split["heldout_indices"])
    # Convert the same target-band held-out rows to MOGP format for evaluation of B\C\D\E\F models on the same target-band held-out points
    target_heldout_m = _build_mogp_data_from_raw_rows(
        example,
        target_heldout_rows,
        target_train_s,
        band_to_wavelength,
    )
    if target_heldout_m is None:
        raise ValueError("Could not build MOGP target held-out data.")
    target_wavelength = _resolve_wavelength(target_band, band_to_wavelength)

    aux_splits = {}
    aux_selected_groups = []
    aux_selected_indices_by_band = {}
    # Iterate over all available bands except the target band to create auxiliary training splits and select points based on the specified ratios.
    for band in _available_bands(example, bands, band_to_wavelength):
        if _resolve_wavelength(band, band_to_wavelength) == target_wavelength and str(band) == str(target_band):
            continue
        if band == target_band:
            continue
        ratio = float(aux_band_ratios.get(band, aux_band_ratios.get(str(band), 0.0)))
        split = _split_raw_band_for_ablation(
            example,
            band,
            heldout_fraction=heldout_fraction,
            min_train_points=min_train_points,
            min_heldout_points=min_heldout_points,
            random_state=random_state,
            strategy=strategy,
            force_peak_in_train=force_peak_in_train,
            peak_mode=peak_mode,
        )
        if split is None:
            continue
        aux_splits[band] = split
        selected = _sample_auxiliary_train_indices(split, ratio, rng)
        aux_selected_indices_by_band[band] = selected
        aux_selected_groups.append(_raw_rows_from_split(split, selected))

    real_train_rows = _concat_raw_row_groups([target_train_rows] + aux_selected_groups)
    rows = []
    artifacts = {}

    # A. Existing single-band GP.
    single_gp = fit_basic_gp(target_train_s, kernel_type="matern", **single_gp_kwargs)
    single_metrics = evaluate_heldout_metrics(
        single_gp,
        target_heldout_s,
        train_data=target_train_s,
    )
    rows.append(_metrics_row_from_result(
        "single_band_gp",
        single_metrics,
        target_train_s,
        target_band,
        target_heldout_s["heldout_indices"],
    ))
    artifacts["single_band_gp"] = {
        "gp": single_gp,
        "train_data": target_train_s,
        "heldout_data": target_heldout_s,
    }

    # B. MOGP target-only, fixed wavelength.
    target_only_train_m = _build_mogp_data_from_raw_rows(
        example,
        target_train_rows,
        target_train_s,
        band_to_wavelength,
    )
    if target_only_train_m is None:
        raise ValueError("Could not build MOGP target-only training data.")
    target_only_gp = fit_mogp_gp(target_only_train_m, **mogp_gp_kwargs)
    target_only_metrics = _evaluate_mogp_on_target_heldout(
        target_only_gp,
        target_heldout_m,
        target_only_train_m,
    )
    rows.append(_metrics_row_from_result(
        "mogp_target_only",
        target_only_metrics,
        target_only_train_m,
        target_band,
        target_heldout_s["heldout_indices"],
    ))
    artifacts["mogp_target_only"] = {
        "gp": target_only_gp,
        "train_data": target_only_train_m,
        "heldout_data": target_heldout_m,
    }

    # C. Real wavelengths with selected auxiliary bands.
    real_train_m = _build_mogp_data_from_raw_rows(
        example,
        real_train_rows,
        target_train_s,
        band_to_wavelength,
    )
    if real_train_m is None:
        raise ValueError("Could not build MOGP real-wavelength training data.")
    real_gp = fit_mogp_gp(real_train_m, **mogp_gp_kwargs)
    real_metrics = _evaluate_mogp_on_target_heldout(real_gp, target_heldout_m, real_train_m)
    rows.append(_metrics_row_from_result(
        "mogp_real_wavelength",
        real_metrics,
        real_train_m,
        target_band,
        target_heldout_s["heldout_indices"],
    ))
    artifacts["mogp_real_wavelength"] = {
        "gp": real_gp,
        "train_data": real_train_m,
        "heldout_data": target_heldout_m,
        "aux_selected_indices_by_band": aux_selected_indices_by_band,
    }

    # D. Same points as C, approximate independent-band control.
    independent_kwargs = dict(mogp_gp_kwargs)
    independent_kwargs.setdefault("wavelength_length_scale", 1e-6)
    independent_kwargs.setdefault("wavelength_length_scale_bounds", (1e-6, 1e-6))
    independent_kwargs.setdefault("n_restarts_optimizer", 0)
    independent_gp = fit_mogp_gp(real_train_m, **independent_kwargs)
    independent_metrics = _evaluate_mogp_on_target_heldout(
        independent_gp,
        target_heldout_m,
        real_train_m,
    )
    rows.append(_metrics_row_from_result(
        "mogp_independent_band_control",
        independent_metrics,
        real_train_m,
        target_band,
        target_heldout_s["heldout_indices"],
        notes="Approximate independent-band control: wavelength length scale set near zero.",
    ))
    artifacts["mogp_independent_band_control"] = {
        "gp": independent_gp,
        "train_data": real_train_m,
        "heldout_data": target_heldout_m,
    }

    # E. Same points as C, shuffled non-target wavelengths.
    shuffled_rows = []
    non_target_mask = ~_band_equal_mask(real_train_m["band"], target_band)
    non_target_wavelengths = np.asarray(real_train_m["wavelength"], dtype=float)[non_target_mask]
    for repeat_idx in range(shuffle_repeats):
        shuffled_wavelengths = np.asarray(real_train_m["wavelength"], dtype=float).copy()
        if len(non_target_wavelengths) > 1:
            shuffled_wavelengths[non_target_mask] = rng.permutation(non_target_wavelengths)
        shuffled_train_m = real_train_m.copy()
        shuffled_train_m["wavelength"] = shuffled_wavelengths
        shuffled_train_m["X"] = np.column_stack([shuffled_train_m["t"], shuffled_wavelengths])
        shuffled_gp = fit_mogp_gp(shuffled_train_m, **mogp_gp_kwargs)
        shuffled_metrics = _evaluate_mogp_on_target_heldout(
            shuffled_gp,
            target_heldout_m,
            shuffled_train_m,
        )
        shuffled_row = _metrics_row_from_result(
            f"mogp_shuffled_wavelength_control_seed{repeat_idx}",
            shuffled_metrics,
            shuffled_train_m,
            target_band,
            target_heldout_s["heldout_indices"],
        )
        shuffled_row["shuffle_repeat"] = repeat_idx
        rows.append(shuffled_row)
        shuffled_rows.append({
            "gp": shuffled_gp,
            "train_data": shuffled_train_m,
            "heldout_data": target_heldout_m,
        })
    artifacts["mogp_shuffled_wavelength_control"] = shuffled_rows

    # F. Existing fixed-total-budget comparison: N total rows sampled from all bands.
    if include_same_total_train_budget:
        all_train_groups = [target_train_rows] + [
            _raw_rows_from_split(split, split["train_indices"])
            for split in aux_splits.values()
        ]
        all_train_rows = _concat_raw_row_groups(all_train_groups)
        n_budget = len(target_train_rows["t_raw"])
        if all_train_rows is not None and len(all_train_rows["t_raw"]) >= n_budget:
            budget_indices = np.sort(rng.choice(np.arange(len(all_train_rows["t_raw"])), size=n_budget, replace=False))
            budget_rows = {key: value[budget_indices] for key, value in all_train_rows.items()}
            budget_train_m = _build_mogp_data_from_raw_rows(
                example,
                budget_rows,
                target_train_s,
                band_to_wavelength,
            )
            if budget_train_m is None:
                raise ValueError("Could not build fixed-total-budget MOGP training data.")
            budget_gp = fit_mogp_gp(budget_train_m, **mogp_gp_kwargs)
            budget_metrics = _evaluate_mogp_on_target_heldout(
                budget_gp,
                target_heldout_m,
                budget_train_m,
            )
            rows.append(_metrics_row_from_result(
                "same_total_train_budget_existing",
                budget_metrics,
                budget_train_m,
                target_band,
                target_heldout_s["heldout_indices"],
                notes="Fixed-total-budget comparison, not an isolated cross-band covariance test.",
            ))
            artifacts["same_total_train_budget_existing"] = {
                "gp": budget_gp,
                "train_data": budget_train_m,
                "heldout_data": target_heldout_m,
            }

    expected_heldout = np.asarray(target_heldout_s["heldout_indices"], dtype=int)
    for row in rows:
        if not np.array_equal(np.asarray(row["heldout_indices"], dtype=int), expected_heldout):
            raise AssertionError(f"Held-out indices differ for model {row['model']}.")

    return {
        "object_id": example["object_id"],
        "target_band": target_band,
        "target_heldout_indices": expected_heldout,
        "aux_band_ratios": aux_band_ratios,
        "rows": rows,
        "artifacts": artifacts,
    }


def summarize_metrics_by_band(object_results):
    """
    Aggregate per-observation metrics into per-band summaries across objects.
    """
    rows = []
    for result in object_results:
        for band in np.unique(result["band"]):
            mask = np.asarray(result["band"]) == band
            y = np.asarray(result["y_true"])[mask]
            pred = np.asarray(result["y_pred"])[mask]
            std = np.maximum(np.asarray(result["y_std"])[mask], 1e-12)
            per_point_nlpd = negative_log_predictive_density(y, pred, std ** 2)
            z = (y - pred) / std
            rows.append({
                "band": band,
                "n_heldout": int(np.sum(mask)),
                "total_nlpd": float(np.sum(per_point_nlpd)),
                "sse": float(np.sum((y - pred) ** 2)),
                "coverage_1sigma_count": int(np.sum(np.abs(z) <= 1)),
                "coverage_2sigma_count": int(np.sum(np.abs(z) <= 2)),
                "coverage_3sigma_count": int(np.sum(np.abs(z) <= 3)),
            })

    summary = {}
    for band in sorted({row["band"] for row in rows}, key=str):
        band_rows = [row for row in rows if row["band"] == band]
        n = sum(row["n_heldout"] for row in band_rows)
        summary[band] = {
            "n_heldout": int(n),
            "nlpd": float(sum(row["total_nlpd"] for row in band_rows) / n),
            "rmse": float(np.sqrt(sum(row["sse"] for row in band_rows) / n)),
            "coverage_1sigma": float(sum(row["coverage_1sigma_count"] for row in band_rows) / n),
            "coverage_2sigma": float(sum(row["coverage_2sigma_count"] for row in band_rows) / n),
            "coverage_3sigma": float(sum(row["coverage_3sigma_count"] for row in band_rows) / n),
        }
    return summary


def _debug_object_summary(train_data, heldout_data, gp=None):
    alpha = np.asarray(train_data["yerr"]) ** 2
    summary = {
        "object_id": train_data["obj_id"],
        "bands_used": sorted({str(b) for b in train_data["band"]}),
        "n_train_by_band": {
            str(b): int(np.sum(train_data["band"] == b)) for b in np.unique(train_data["band"])
        },
        "n_heldout_by_band": {
            str(b): int(np.sum(heldout_data["band"] == b)) for b in np.unique(heldout_data["band"])
        },
        "y_range": (float(np.min(train_data["y"])), float(np.max(train_data["y"]))),
        "yerr_range": (float(np.min(train_data["yerr"])), float(np.max(train_data["yerr"]))),
        "flux_scale": float(train_data["flux_scale"]),
        "background_by_band": {str(k): float(v) for k, v in train_data["background_by_band"].items()},
        "alpha_quantiles_from_yerr_only": tuple(float(x) for x in np.percentile(alpha, [0, 50, 90, 100])),
    }
    if gp is not None:
        summary["learned_kernel"] = str(gp.kernel_)
    return summary


def run_mogp_evaluation(
        examples,
        bands=("u", "g", "r", "i", "z", "Y", 0, 1, 2, 3, 4, 5),
        n_objects=20,
        max_examples_to_scan=3000,
        debug_n_objects=3,
        random_state=0,
        **kwargs,
):
    """
    Run a comparison-friendly MOGP evaluation and return object-level,
    aggregate, and per-band metrics.
    """
    if isinstance(examples, dict) and "lightcurve" in examples:
        examples = [examples]

    object_results = []
    processed_objects = []
    debug = []
    scanned = 0

    process_keys = {
        "band_to_wavelength",
        "flux_scale",
        "align_peak",
        "peak_alignment",
        "peak_mode",
        "force_peak_in_train",
        "subtract_background",
        "background_flux",
        "background_estimator",
        "background_mode",
        "scale_mode",
        "local_flux_percentile",
        "scale_eps",
        "normalize_flux",
        "min_points_per_band",
        "min_total_points",
        "heldout_fraction",
        "min_train_points",
        "min_heldout_points",
        "strategy",
        "subsample_reference_band",
        "preserve_peak",
        "subsample_random_state",
    }
    fit_keys = {
        "time_length_scale",
        "wavelength_length_scale",
        "time_length_scale_bounds",
        "wavelength_length_scale_bounds",
        "constant_value",
        "constant_value_bounds",
        "yerr_scale",
        "noise_floor",
        "jitter",
        "n_restarts_optimizer",
    }
    eval_keys = {"include_yerr", "yerr_scale", "noise_floor", "evaluate_raw_metrics"}

    process_kwargs = {k: v for k, v in kwargs.items() if k in process_keys}
    fit_kwargs = {k: v for k, v in kwargs.items() if k in fit_keys}
    eval_kwargs = {k: v for k, v in kwargs.items() if k in eval_keys}

    if process_kwargs.get("scale_mode") == "global" and process_kwargs.get("flux_scale") is None:
        reference_band = process_kwargs.get("subsample_reference_band")
        if reference_band is None:
            raise ValueError(
                "MOGP scale_mode='global' with automatic scale requires "
                "subsample_reference_band so the reduced training set is well defined."
            )
        if not isinstance(examples, list):
            examples = list(islice(examples, max_examples_to_scan))
        global_scale_process_kwargs = {
            key: value
            for key, value in process_kwargs.items()
            if key not in {
                "subsample_reference_band",
                "preserve_peak",
                "subtract_background",
                "background_flux",
                "background_estimator",
                "background_mode",
                "flux_scale",
                "scale_mode",
            }
        }
        global_scale = find_reduced_mogp_train_global_percentile_flux_peak(
            examples,
            reference_band=reference_band,
            bands=bands,
            percentile=kwargs.get("global_flux_percentile", 95),
            n_objects=n_objects,
            max_examples_to_scan=max_examples_to_scan,
            random_state=random_state,
            preserve_peak=process_kwargs.get("preserve_peak", True),
            subtract_background=process_kwargs.get("subtract_background", False),
            background_flux=process_kwargs.get("background_flux"),
            background_estimator=process_kwargs.get("background_estimator", bitweight_location),
            background_mode=process_kwargs.get("background_mode", "object"),
            **global_scale_process_kwargs,
        )
        if global_scale is None:
            raise ValueError("Could not compute reduced MOGP global scale from any usable object.")
        process_kwargs["flux_scale"] = global_scale["flux_scale"]

    for object_idx, example in enumerate(examples):
        if len(object_results) >= n_objects or object_idx >= max_examples_to_scan:
            break
        scanned = object_idx + 1

        train_data, heldout_data = process_one_obj_mogp_train_heldout(
            example,
            bands=bands,
            random_state=random_state + object_idx,
            **process_kwargs,
        )
        if train_data is None or heldout_data is None:
            continue

        gp = fit_mogp_gp(
            train_data,
            random_state=random_state + object_idx,
            print_kernel=len(debug) < debug_n_objects,
            **fit_kwargs,
        )
        metrics = evaluate_mogp_heldout_metrics(
            gp,
            heldout_data,
            train_data=train_data,
            **eval_kwargs,
        )
        object_results.append(metrics)
        processed_objects.append({
            "object_idx": object_idx,
            "example": example,
            "train_data": train_data,
            "heldout_data": heldout_data,
            "gp": gp,
            "metrics": metrics,
        })
        if len(debug) < debug_n_objects:
            debug.append(_debug_object_summary(train_data, heldout_data, gp=gp))

    if len(object_results) == 0:
        raise ValueError("No usable objects found for MOGP evaluation.")

    return {
        "object_results": object_results,
        "processed_objects": processed_objects,
        "aggregate": summarize_object_metric_results(object_results),
        "per_band": summarize_metrics_by_band(object_results),
        "debug": debug,
        "scanned_examples": scanned,
        "bands": bands,
    }


def _canonical_ablation_model_name(model):
    model = str(model)
    if model.startswith("mogp_shuffled_wavelength_control_seed"):
        return "mogp_shuffled_wavelength_control"
    return model


def collapse_shuffled_wavelength_controls(
        results_df,
        keep_seed_level=False,
):
    """
    Collapse shuffled wavelength seed rows to one object-band row per model.

    C vs shuffled controls tests whether real wavelength structure matters
    beyond generic cross-band information sharing.  Collapsing prevents one
    object with many shuffle seeds from being overrepresented in aggregates.
    """
    df = results_df.copy()
    if "model" not in df.columns:
        raise ValueError("results_df must contain a 'model' column.")

    df["model_original"] = df["model"]
    df["model"] = df["model"].map(_canonical_ablation_model_name)
    if keep_seed_level:
        return df

    group_cols = ["model", "object_id", "target_band"]
    missing = [col for col in group_cols if col not in df.columns]
    if missing:
        raise ValueError(f"results_df missing required columns for shuffled collapse: {missing}")

    numeric_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in {"shuffle_repeat"}
    ]
    first_cols = [
        col for col in df.columns
        if col not in set(group_cols + numeric_cols)
    ]
    agg_spec = {col: "mean" for col in numeric_cols}
    agg_spec.update({col: "first" for col in first_cols})

    collapsed = df.groupby(group_cols, as_index=False, dropna=False).agg(agg_spec)
    return collapsed


def _prepare_ablation_results_df(
        results_df,
        collapse_shuffled=True,
        keep_seed_level=False,
):
    df = results_df.copy()
    required = {
        "model",
        "object_id",
        "target_band",
        "n_heldout_target_band",
        "rmse",
        "nlpd",
        "coverage_1sigma",
        "coverage_2sigma",
        "coverage_3sigma",
        "z_score_mean",
        "z_score_std",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"results_df is missing required columns: {missing}")

    df = df[pd.to_numeric(df["n_heldout_target_band"], errors="coerce") > 0].copy()
    if len(df) == 0:
        raise ValueError("No rows remain after excluding n_heldout_target_band <= 0.")

    if collapse_shuffled:
        df = collapse_shuffled_wavelength_controls(df, keep_seed_level=keep_seed_level)

    return df


def _standard_error(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= 1:
        return np.nan
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.nan
    return float(np.sum(weights[valid] * values[valid]) / np.sum(weights[valid]))


def _aggregate_residual_level_if_available(group):
    """
    Prefer residual-level arrays for z/RMSE when the dataframe carries them.
    Expected optional columns are y_true/y_pred and/or z_scores.
    """
    y_true_values = []
    y_pred_values = []
    z_values = []
    if {"y_true", "y_pred"}.issubset(group.columns):
        for _, row in group.iterrows():
            y_true = row.get("y_true")
            y_pred = row.get("y_pred")
            if y_true is not None and y_pred is not None:
                y_true_values.append(np.asarray(y_true, dtype=float).reshape(-1))
                y_pred_values.append(np.asarray(y_pred, dtype=float).reshape(-1))
    if "z_scores" in group.columns:
        for _, row in group.iterrows():
            z = row.get("z_scores")
            if z is not None:
                z_values.append(np.asarray(z, dtype=float).reshape(-1))

    out = {}
    if len(y_true_values) > 0:
        y_true_all = np.concatenate(y_true_values)
        y_pred_all = np.concatenate(y_pred_values)
        out["rmse_obs_weighted"] = float(np.sqrt(np.mean((y_true_all - y_pred_all) ** 2)))
    if len(z_values) > 0:
        z_all = np.concatenate(z_values)
        out["z_score_mean_obs_weighted"] = float(np.mean(z_all))
        out["z_score_std_obs_weighted"] = float(np.std(z_all))
    return out


def aggregate_gp_ablation_results(
        results_df,
        group_cols=None,
        residual_level_available=False,
        collapse_shuffled=True,
        keep_seed_level=False,
        min_cases_warn=5,
):
    """
    Aggregate GP ablation rows using observation- and object-weighted summaries.

    Observation-weighted metrics describe performance over all held-out
    observations.  Object-weighted metrics describe performance for a typical
    object-band case and prevent high-cadence objects from dominating.
    """
    if group_cols is None:
        group_cols = ["model"]
    df = _prepare_ablation_results_df(
        results_df,
        collapse_shuffled=collapse_shuffled,
        keep_seed_level=keep_seed_level,
    )

    rows = []
    for group_key, group in df.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = {col: value for col, value in zip(group_cols, group_key)}

        n = group["n_heldout_target_band"].astype(float).to_numpy()
        row["n_object_band_cases"] = int(len(group))
        row["n_unique_objects"] = int(group["object_id"].nunique())
        row["n_total_heldout"] = int(np.sum(n))

        if row["n_object_band_cases"] < min_cases_warn:
            warnings.warn(
                f"Group {row} has only {row['n_object_band_cases']} object-band cases; "
                "do not treat this as global evidence.",
                RuntimeWarning,
            )

        row["rmse_obs_weighted"] = float(
            np.sqrt(np.sum(n * group["rmse"].astype(float).to_numpy() ** 2) / np.sum(n))
        )
        row["nlpd_obs_weighted"] = _weighted_mean(group["nlpd"], n)
        for cov in ("coverage_1sigma", "coverage_2sigma", "coverage_3sigma"):
            row[f"{cov}_obs_weighted"] = _weighted_mean(group[cov], n)

        z_mean_obs = _weighted_mean(group["z_score_mean"], n)
        z_second = _weighted_mean(
            group["z_score_std"].astype(float) ** 2 + group["z_score_mean"].astype(float) ** 2,
            n,
        )
        row["z_score_mean_obs_weighted"] = z_mean_obs
        row["z_score_std_obs_weighted"] = float(
            np.sqrt(max(z_second - z_mean_obs ** 2, 0.0))
        )

        if residual_level_available:
            row.update(_aggregate_residual_level_if_available(group))

        object_metrics = {
            "rmse": "rmse_object_weighted",
            "nlpd": "nlpd_object_weighted",
            "coverage_1sigma": "coverage_1sigma_object_weighted",
            "coverage_2sigma": "coverage_2sigma_object_weighted",
            "coverage_3sigma": "coverage_3sigma_object_weighted",
            "z_score_mean": "z_score_mean_object_weighted",
            "z_score_std": "z_score_std_object_weighted",
        }
        for src, dest in object_metrics.items():
            values = group[src].astype(float).to_numpy()
            row[dest] = float(np.mean(values))
            row[f"{src}_object_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            row[f"{src}_object_se"] = _standard_error(values)

        rows.append(row)

    return pd.DataFrame(rows)


def compare_models_aggregated(
        results_df,
        model_pairs=None,
        collapse_shuffled=True,
        keep_seed_level=False,
):
    """
    Paired model comparisons matched exactly by object_id and target_band.

    C vs D, mogp_real_wavelength vs mogp_independent_band_control, is the key
    comparison for whether cross-band covariance helps beyond merely adding
    more data points.  A/B matching is a preprocessing sanity check.
    """
    if model_pairs is None:
        model_pairs = [
            ("mogp_real_wavelength", "single_band_gp"),
            ("mogp_real_wavelength", "mogp_independent_band_control"),
            ("mogp_real_wavelength", "mogp_shuffled_wavelength_control"),
            ("same_total_train_budget_existing", "single_band_gp"),
            ("mogp_target_only", "single_band_gp"),
        ]
    df = _prepare_ablation_results_df(
        results_df,
        collapse_shuffled=collapse_shuffled,
        keep_seed_level=keep_seed_level,
    )

    out_rows = []
    for model_a, model_b in model_pairs:
        a = df[df["model"] == model_a].copy()
        b = df[df["model"] == model_b].copy()
        paired = a.merge(
            b,
            on=["object_id", "target_band"],
            suffixes=("_a", "_b"),
            how="inner",
        )
        if len(paired) == 0:
            warnings.warn(
                f"No matched object-band pairs for {model_a} vs {model_b}.",
                RuntimeWarning,
            )
            continue

        delta_rmse = paired["rmse_a"].astype(float) - paired["rmse_b"].astype(float)
        delta_nlpd = paired["nlpd_a"].astype(float) - paired["nlpd_b"].astype(float)
        delta_cov1 = paired["coverage_1sigma_a"].astype(float) - paired["coverage_1sigma_b"].astype(float)
        delta_z_std = paired["z_score_std_a"].astype(float) - paired["z_score_std_b"].astype(float)
        better_z = (
            np.abs(paired["z_score_std_a"].astype(float) - 1.0)
            < np.abs(paired["z_score_std_b"].astype(float) - 1.0)
        )

        row = {
            "model_a": model_a,
            "model_b": model_b,
            "n_matched_object_band_pairs": int(len(paired)),
            "delta_rmse_mean": float(np.mean(delta_rmse)),
            "delta_rmse_median": float(np.median(delta_rmse)),
            "delta_rmse_se": _standard_error(delta_rmse),
            "delta_nlpd_mean": float(np.mean(delta_nlpd)),
            "delta_nlpd_median": float(np.median(delta_nlpd)),
            "delta_nlpd_se": _standard_error(delta_nlpd),
            "delta_coverage_1sigma_mean": float(np.mean(delta_cov1)),
            "delta_coverage_1sigma_median": float(np.median(delta_cov1)),
            "delta_coverage_1sigma_se": _standard_error(delta_cov1),
            "delta_z_score_std_mean": float(np.mean(delta_z_std)),
            "delta_z_score_std_median": float(np.median(delta_z_std)),
            "delta_z_score_std_se": _standard_error(delta_z_std),
            "fraction_improved_rmse": float(np.mean(delta_rmse < 0)),
            "fraction_improved_nlpd": float(np.mean(delta_nlpd < 0)),
            "fraction_better_calibrated_z_std": float(np.mean(better_z)),
        }
        out_rows.append(row)

    return pd.DataFrame(out_rows)


def save_gp_ablation_reports(
        results_df,
        output_dir=".",
        collapse_shuffled=True,
        keep_seed_level=False,
):
    """
    Save standard ablation aggregate CSV reports.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_model = aggregate_gp_ablation_results(
        results_df,
        group_cols=["model"],
        collapse_shuffled=collapse_shuffled,
        keep_seed_level=keep_seed_level,
    )
    by_model_band = aggregate_gp_ablation_results(
        results_df,
        group_cols=["model", "target_band"],
        collapse_shuffled=collapse_shuffled,
        keep_seed_level=keep_seed_level,
    )
    comparisons = compare_models_aggregated(
        results_df,
        collapse_shuffled=collapse_shuffled,
        keep_seed_level=keep_seed_level,
    )

    paths = {
        "by_model": output_dir / "gp_ablation_aggregated_by_model.csv",
        "by_model_and_band": output_dir / "gp_ablation_aggregated_by_model_and_band.csv",
        "paired_model_comparisons": output_dir / "gp_ablation_paired_model_comparisons.csv",
    }
    by_model.to_csv(paths["by_model"], index=False)
    by_model_band.to_csv(paths["by_model_and_band"], index=False)
    comparisons.to_csv(paths["paired_model_comparisons"], index=False)

    return {
        "aggregated_by_model": by_model,
        "aggregated_by_model_and_band": by_model_band,
        "paired_model_comparisons": comparisons,
        "paths": paths,
    }


def plot_gp_ablation_summary(
        results_df,
        output_dir=None,
        collapse_shuffled=True,
):
    """
    Create optional summary plots for ablation aggregates.
    """
    import matplotlib.pyplot as plt

    output_dir = None if output_dir is None else Path(output_dir)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    agg = aggregate_gp_ablation_results(
        results_df,
        group_cols=["model"],
        collapse_shuffled=collapse_shuffled,
    )
    prepared = _prepare_ablation_results_df(results_df, collapse_shuffled=collapse_shuffled)

    figures = {}
    for metric, title, filename in [
        ("nlpd_obs_weighted", "Observation-weighted NLPD by model", "nlpd_observation_weighted_by_model.png"),
        ("nlpd_object_weighted", "Object-weighted NLPD by model", "nlpd_object_weighted_by_model.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        ordered = agg.sort_values(metric)
        ax.bar(ordered["model"], ordered[metric])
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        if output_dir is not None:
            fig.savefig(output_dir / filename, bbox_inches="tight")
        figures[metric] = fig

    paired = prepared.pivot_table(
        index=["object_id", "target_band"],
        columns="model",
        values="nlpd",
        aggfunc="mean",
    )
    if {"mogp_real_wavelength", "mogp_independent_band_control"}.issubset(paired.columns):
        delta = paired["mogp_real_wavelength"] - paired["mogp_independent_band_control"]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(delta.dropna(), bins=20, alpha=0.8)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title("Paired delta NLPD: real wavelength - independent-band control")
        ax.set_xlabel("Delta NLPD")
        ax.set_ylabel("Object-band count")
        fig.tight_layout()
        if output_dir is not None:
            fig.savefig(output_dir / "delta_nlpd_real_vs_independent.png", bbox_inches="tight")
        figures["delta_nlpd_real_vs_independent"] = fig

    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = agg.sort_values("z_score_std_object_weighted")
    ax.bar(ordered["model"], ordered["z_score_std_object_weighted"])
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="ideal")
    ax.set_ylabel("Object-weighted z-score std")
    ax.set_title("z-score std by model")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    if output_dir is not None:
        fig.savefig(output_dir / "z_score_std_by_model.png", bbox_inches="tight")
    figures["z_score_std_by_model"] = fig

    return figures
