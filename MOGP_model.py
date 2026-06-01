from typing import Any, cast

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from data_processing import (
    _extract_valid_band_observations,
    _find_peak_indices,
    _resolve_background_and_scale,
    _split_indices,
    bitweight_location,
)
from singleGP_model import (
    inverse_transform_predictions,
    negative_log_predictive_density,
    summarize_object_metric_results,
)


# LSST-like effective wavelengths in nanometers.  PLAsTiCC often stores bands
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
    """
    Resolve the wavelength for a given band using the provided mapping, with flexible key types.
    """
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


def _background_for_training_flux(
        train_by_band,
        subtract_background=False,
        background_flux=None,
        background_estimator=bitweight_location,
        background_mode="object",
):
    if not subtract_background:
        return {band: 0.0 for band in train_by_band}

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

    y_train = np.concatenate([arrays["y_raw"] for arrays in train_by_band.values()])
    if background_flux is None:
        background = float(background_estimator(y_train)[0])
    else:
        background = float(background_flux)
    if not np.isfinite(background):
        raise ValueError("Estimated object background_flux is not finite.")

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
        indices = split[index_key]
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
        background_mode="object",
        scale_mode=None,
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
):
    """
    Split each band with the same helper used by the single-band GP, then fit
    one continuous GP over [time, wavelength].  This is not an ICM/LMC
    coregionalization model
    """
    band_to_wavelength = band_to_wavelength or DEFAULT_BAND_TO_WAVELENGTH
    usable_bands = _available_bands(example, bands, band_to_wavelength)

    split_by_band = {}
    train_by_band = {}
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
        train_by_band[band] = {
            "t_raw": t_raw[train_indices],
            "y_raw": y_raw[train_indices],
            "yerr_raw": yerr_raw[train_indices],
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

    backgrounds = _background_for_training_flux(
        train_by_band,
        subtract_background=subtract_background,
        background_flux=background_flux,
        background_estimator=background_estimator,
        background_mode=background_mode,
    )

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
        if scale_mode == "global":
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
                scale_mode="local_peak",
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

    train_data["train_indices_by_band"] = {
        band: split["train_indices"] for band, split in split_by_band.items()
    }
    heldout_data["heldout_indices_by_band"] = {
        band: split["heldout_indices"] for band, split in split_by_band.items()
    }

    return train_data, heldout_data


def fit_mogp_gp(
        data,
        time_length_scale=0.3,
        wavelength_length_scale=200.0,
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
    """
    Fit the minimal continuous time+wavelength GP baseline.
    """
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


def summarize_metrics_by_band(object_results):
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
    Run a small comparison-friendly MOGP evaluation and return object-level,
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
        
        print("total train:", len(train_data["y"]))
        print("total heldout:", len(heldout_data["y"]))

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
