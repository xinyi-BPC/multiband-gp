from itertools import islice
from typing import Any, cast

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
from evaluation_metrics import (
    _metrics_row_from_result,
    evaluate_heldout_metrics,
    evaluate_mogp_heldout_metrics,
    summarize_metrics_by_band,
    summarize_object_metric_results,
)
from singleGP_model import (
    fit_basic_gp,
    object_level_empirical_flux_scale,
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
    """
    Stack the data from a single example into a format suitable for MOGP training, including the scales.
    """
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

        # This is Model F: fixed total budget per object 
        # thus we want to reduce the multiband train set to the reference-band train count while preserving the original multiband alignment before reduction. 
        # The non-selected original train rows are appended to the heldout set to maintain a consistent total number of observations.
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
        print_kernel=False,
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


def _band_equal_mask(values, band):
    """
    Create a boolean mask for values that match a specific band.
    """
    values = np.asarray(values, dtype=object)
    # First check for direct equality, then check for string equality to handle cases where bands might be stored as strings or integers.
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
    """
    Concatenate raw row from multiple bands into a single dict, ensuring the rows are sorted by time and then by band.
    """
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


def _sample_auxiliary_train_indices(split, ratio, rng, n_target_train):
    """Sample a subset of the auxiliary band train indices based on the specified ratio relative to target band training points."""
    available = np.asarray(split["train_indices"], dtype=int)
    if ratio <= 0 or n_target_train <= 0 or len(available) == 0:
        return np.array([], dtype=int)
    n_select = np.floor(ratio * n_target_train).astype(int)   # The required number of auxiliary training points
    n_available = len(available)
    n_select = min(n_select, n_available)
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


def _evaluate_mogp_on_target_heldout(
        gp,
        target_heldout_data,
        train_data,
        object_data=None,
        object_flux_scale=None,
        nrmse_quantile=0.95,
        nrmse_epsilon=1e-8,
):
    metrics = evaluate_mogp_heldout_metrics(
        gp,
        target_heldout_data,
        train_data=train_data,
        object_data=object_data,
        object_flux_scale=object_flux_scale,
        nrmse_quantile=nrmse_quantile,
        nrmse_epsilon=nrmse_epsilon,
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
        nrmse_quantile=0.95,
        nrmse_epsilon=1e-8,
        single_gp_kwargs=None,
        mogp_gp_kwargs=None,
        output_csv_path=None,
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
    object_flux_scale = object_level_empirical_flux_scale(
        example=example,
        q=nrmse_quantile,
        epsilon=nrmse_epsilon,
    )

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
    n_target_train = len(target_train_rows["t_raw"])
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
        selected = _sample_auxiliary_train_indices(split, ratio, rng, n_target_train)
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
        object_data=example,
        object_flux_scale=object_flux_scale,
        nrmse_quantile=nrmse_quantile,
        nrmse_epsilon=nrmse_epsilon,
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
        object_data=example,
        object_flux_scale=object_flux_scale,
        nrmse_quantile=nrmse_quantile,
        nrmse_epsilon=nrmse_epsilon,
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
    real_metrics = _evaluate_mogp_on_target_heldout(
        real_gp,
        target_heldout_m,
        real_train_m,
        object_data=example,
        object_flux_scale=object_flux_scale,
        nrmse_quantile=nrmse_quantile,
        nrmse_epsilon=nrmse_epsilon,
    )
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
        object_data=example,
        object_flux_scale=object_flux_scale,
        nrmse_quantile=nrmse_quantile,
        nrmse_epsilon=nrmse_epsilon,
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
            object_data=example,
            object_flux_scale=object_flux_scale,
            nrmse_quantile=nrmse_quantile,
            nrmse_epsilon=nrmse_epsilon,
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
                object_data=example,
                object_flux_scale=object_flux_scale,
                nrmse_quantile=nrmse_quantile,
                nrmse_epsilon=nrmse_epsilon,
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

    if output_csv_path is not None:
        pd.DataFrame(rows).to_csv(output_csv_path, index=False)

    return {
        "object_id": example["object_id"],
        "target_band": target_band,
        "target_heldout_indices": expected_heldout,
        "aux_band_ratios": aux_band_ratios,
        "rows": rows,
        "artifacts": artifacts,
    }


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
    eval_keys = {
        "include_yerr",
        "yerr_scale",
        "noise_floor",
        "evaluate_raw_metrics",
        "nrmse_quantile",
        "nrmse_epsilon",
    }

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
            object_data=example,
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
