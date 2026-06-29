"""Feature extraction for single-band GP calibration analysis.

The functions in this module operate at the object-band level: one object,
one photometric band, one single-band GP fit/evaluation split.  Feature groups
are intentionally separated by what data they are allowed to use, so analyses
can distinguish intrinsic observed light-curve shape from GP training coverage,
test geometry, and raw signal-to-noise.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EPS = 1e-12
FeatureDict = dict[str, Any]
GP_METRIC_COLUMNS = (
    "rmse",
    "nrmse",
    "nlpd",
    "crps",
    "ncrps",
    "z_std",
    "z_mean",
    "coverage_1sigma",
    "coverage_2sigma",
    "coverage_3sigma",
    "ks_pit",
    "object_flux_scale",
)
LIGHTCURVE_ARRAY_COLUMNS = (
    "t_all",
    "flux_all",
    "flux_err_all",
    "t_train",
    "flux_train",
    "flux_err_train",
    "t_test",
    "flux_test",
)


def _as_float_array(values: Any) -> np.ndarray:
    """Return values as a flattened float array; invalid inputs become empty."""
    if values is None:
        return np.array([], dtype=float)
    try:
        return np.asarray(values, dtype=float).ravel()
    except (TypeError, ValueError):
        return np.array([], dtype=float)


def _finite_pair(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return finite, aligned arrays for paired measurements."""
    x_arr = _as_float_array(x)
    y_arr = _as_float_array(y)
    n = min(x_arr.size, y_arr.size)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    x_arr = x_arr[:n]
    y_arr = y_arr[:n]
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    return x_arr[mask], y_arr[mask]


def _finite_values(values: Any) -> np.ndarray:
    """Return finite values from a one-dimensional numeric input."""
    arr = _as_float_array(values)
    return arr[np.isfinite(arr)]


def _nanpercentile(values: Any, q: float) -> float:
    """Return a percentile or NaN if no finite values are available."""
    arr = _finite_values(values)
    if arr.size == 0:
        return np.nan
    return float(np.percentile(arr, q))


def _nan_stat(values: Any, func: Any) -> float:
    """Return a scalar statistic or NaN if no finite values are available."""
    arr = _finite_values(values)
    if arr.size == 0:
        return np.nan
    return float(func(arr))


def _fraction(mask: np.ndarray, denominator: int) -> float:
    """Return a safe fraction for boolean masks."""
    if denominator <= 0:
        return np.nan
    return float(np.sum(mask) / denominator)


def _time_span(times: Any) -> float:
    """Return max(time) - min(time) for finite times, or NaN."""
    t = _finite_values(times)
    if t.size == 0:
        return np.nan
    return float(np.max(t) - np.min(t))


def robust_flux_scale(flux: Any, *, percentile: float = 95, min_scale: float = EPS) -> float:
    """Estimate a robust flux scale from original flux values.

    Uses complete or subset flux values exactly as supplied by the caller.  By
    default the scale is the 95th percentile of absolute original flux values.
    Returns NaN if the scale is unavailable, non-finite, or too small.
    """
    scale = _nanpercentile(np.abs(_finite_values(flux)), percentile)
    if not np.isfinite(scale) or scale <= min_scale:
        return np.nan
    return float(scale)


def _determine_peak_time(t_all: Any, flux_all: Any) -> float:
    """Return peak time from complete data using normalized complete flux."""
    t, f = _finite_pair(t_all, flux_all)
    if t.size == 0:
        return np.nan

    scale = robust_flux_scale(f)
    if not np.isfinite(scale):
        return np.nan

    f_norm = f / scale
    valid = np.isfinite(f_norm)
    if not np.any(valid):
        return np.nan
    valid_indices = np.flatnonzero(valid)
    peak_index = valid_indices[int(np.argmax(np.abs(f_norm[valid])))]
    return float(t[peak_index])


def _width_above(t_aligned: np.ndarray, flux: np.ndarray, threshold: float, mode: str) -> float:
    """Return time span above an amplitude threshold for normalized flux."""
    valid = np.isfinite(t_aligned) & np.isfinite(flux)
    if not np.any(valid):
        return np.nan

    t = t_aligned[valid]
    f = flux[valid]
    if mode == "abs":
        amp = _nan_stat(np.abs(f), np.max)
        mask = np.greater_equal(np.abs(f), threshold * amp) if np.isfinite(amp) and amp > EPS else None
    elif mode == "pos":
        amp = _nan_stat(f, np.max)
        mask = np.greater_equal(f, threshold * amp) if np.isfinite(amp) and amp > EPS else None
    else:
        raise ValueError(f"Unsupported width mode: {mode}")

    if mask is None or not np.any(mask):
        return np.nan
    return float(np.max(t[mask]) - np.min(t[mask]))


def _crossing_rate(values: np.ndarray, level: float) -> float:
    """Return adjacent crossing rate around a scalar level."""
    v = _finite_values(values)
    if v.size < 2 or not np.isfinite(level):
        return np.nan
    centered = v - level
    left = centered[:-1]
    right = centered[1:]
    crossings = (
        np.less_equal(left, 0) & np.greater(right, 0)
    ) | (
        np.greater_equal(left, 0) & np.less(right, 0)
    )
    return _fraction(crossings, centered.size - 1)


def extract_morphology_features(
    t_all: Any,
    flux_all: Any,
    *,
    normalize: bool = True,
    t_peak: float | None = None,
    near_peak_window: float = 50,
    baseline_window: float = 200,
) -> FeatureDict:
    """Extract morphology features from the complete observed curve.

    Uses complete observed times and complete observed flux.  If
    ``normalize=True``, morphology is computed on flux divided by the robust
    complete-curve flux scale.  If ``t_peak`` is not supplied, it is inferred
    from the complete observed curve as the time of maximum absolute normalized
    flux.  ``near_peak_window`` is accepted for API symmetry; morphology width
    features use amplitude thresholds rather than this window.
    """
    del near_peak_window

    keys = (
        "flux_scale",
        "flux_range",
        "flux_range_robust",
        "amp_abs",
        "amp_pos",
        "amp_neg",
        "flux_std",
        "flux_iqr",
        "baseline_level",
        "baseline_std",
        "flat_fraction",
        "width_20_abs",
        "width_50_abs",
        "width_20_pos",
        "width_50_pos",
        "time_span_all",
        "n_points_all",
    )
    features: FeatureDict = dict.fromkeys(keys, np.nan)

    t, f_original = _finite_pair(t_all, flux_all)
    features["n_points_all"] = int(t.size)
    features["time_span_all"] = _time_span(t)
    features["flux_scale"] = robust_flux_scale(f_original)
    if t.size == 0:
        return features

    if normalize:
        if not np.isfinite(features["flux_scale"]):
            return features
        f = f_original / features["flux_scale"]
    else:
        f = f_original

    if t_peak is None:
        t_peak = _determine_peak_time(t, f_original)
    t_aligned = t - t_peak if np.isfinite(t_peak) else np.full(t.shape, np.nan)

    features.update(
        {
            "flux_range": _nan_stat(f, np.max) - _nan_stat(f, np.min),
            "flux_range_robust": _nanpercentile(f, 95) - _nanpercentile(f, 5),
            "amp_abs": _nan_stat(np.abs(f), np.max),
            "amp_pos": _nan_stat(f, np.max),
            "amp_neg": _nan_stat(f, np.min),
            "flux_std": _nan_stat(f, np.std),
            "flux_iqr": _nanpercentile(f, 75) - _nanpercentile(f, 25),
        }
    )

    baseline_mask = np.isfinite(t_aligned) & np.greater(np.abs(t_aligned), baseline_window)
    baseline_flux = f[baseline_mask] if np.any(baseline_mask) else f
    features["baseline_level"] = _nan_stat(baseline_flux, np.median)
    features["baseline_std"] = _nan_stat(baseline_flux, np.std)

    amp_abs = features["amp_abs"]
    if np.isfinite(amp_abs) and amp_abs > EPS:
        features["flat_fraction"] = _fraction(np.less(np.abs(f), 0.05 * amp_abs), f.size)

    features["width_20_abs"] = _width_above(t_aligned, f, 0.20, "abs")
    features["width_50_abs"] = _width_above(t_aligned, f, 0.50, "abs")
    features["width_20_pos"] = _width_above(t_aligned, f, 0.20, "pos")
    features["width_50_pos"] = _width_above(t_aligned, f, 0.50, "pos")
    return features


def extract_fluctuation_features(
    t_all: Any,
    flux_all: Any,
    *,
    normalize: bool = True,
) -> FeatureDict:
    """Extract fluctuation and roughness features from the complete curve.

    Uses complete observed times and complete observed flux.  By default the
    flux is normalized by the robust complete-curve flux scale before roughness,
    slope, curvature, and crossing features are computed.
    """
    keys = (
        "total_variation",
        "mean_abs_slope",
        "median_abs_slope",
        "max_abs_slope",
        "slope_std",
        "curvature_mean_abs",
        "curvature_std",
        "sign_change_rate",
        "median_crossing_rate",
    )
    features: FeatureDict = dict.fromkeys(keys, np.nan)

    t, f_original = _finite_pair(t_all, flux_all)
    if t.size < 2:
        return features

    if normalize:
        scale = robust_flux_scale(f_original)
        if not np.isfinite(scale):
            return features
        f = f_original / scale
    else:
        f = f_original

    order = np.argsort(t)
    t = t[order]
    f = f[order]
    dt = np.diff(t)
    df = np.diff(f)
    valid_gap = np.greater(dt, 0)

    if np.any(valid_gap):
        slopes = df[valid_gap] / dt[valid_gap]
        midpoint_times = (t[:-1][valid_gap] + t[1:][valid_gap]) / 2.0
        features["total_variation"] = float(np.sum(np.abs(df[valid_gap])))
        features["mean_abs_slope"] = _nan_stat(np.abs(slopes), np.mean)
        features["median_abs_slope"] = _nan_stat(np.abs(slopes), np.median)
        features["max_abs_slope"] = _nan_stat(np.abs(slopes), np.max)
        features["slope_std"] = _nan_stat(slopes, np.std)

        if slopes.size >= 2:
            d_mid = np.diff(midpoint_times)
            d_slope = np.diff(slopes)
            valid_mid = np.greater(d_mid, 0)
            if np.any(valid_mid):
                curvature = d_slope[valid_mid] / d_mid[valid_mid]
                features["curvature_mean_abs"] = _nan_stat(np.abs(curvature), np.mean)
                features["curvature_std"] = _nan_stat(curvature, np.std)

    features["sign_change_rate"] = _crossing_rate(f, 0.0)
    features["median_crossing_rate"] = _crossing_rate(f, _nan_stat(f, np.median))
    return features


def extract_snr_features(flux_all: Any, flux_err_all: Any | None = None) -> FeatureDict:
    """Extract SNR/noise features from original, unnormalized flux.

    Uses original complete observed flux and original complete observed flux
    errors.  Flux is never normalized here.  If errors are missing or invalid,
    raw flux-scale features are still computed and error-based features remain
    NaN.
    """
    features: FeatureDict = {
        "raw_flux_scale": robust_flux_scale(flux_all),
        "median_flux_err": np.nan,
        "mean_flux_err": np.nan,
        "peak_snr": np.nan,
        "median_snr": np.nan,
        "mean_snr": np.nan,
        "noise_to_signal": np.nan,
        "relative_noise_p95": np.nan,
        "has_flux_err": bool(flux_err_all is not None),
    }

    flux = _finite_values(flux_all)
    if flux_err_all is None:
        return features

    err_raw = _as_float_array(flux_err_all)
    n = min(_as_float_array(flux_all).size, err_raw.size)
    if n == 0:
        features["has_flux_err"] = False
        return features

    flux_aligned = _as_float_array(flux_all)[:n]
    err_aligned = err_raw[:n]
    valid = np.isfinite(flux_aligned) & np.isfinite(err_aligned) & np.greater(err_aligned, 0)
    if not np.any(valid):
        features["has_flux_err"] = False
        return features

    flux_valid = flux_aligned[valid]
    err_valid = err_aligned[valid]
    median_err = _nan_stat(err_valid, np.median)
    raw_scale = features["raw_flux_scale"]
    snr = np.abs(flux_valid) / err_valid

    features.update(
        {
            "median_flux_err": median_err,
            "mean_flux_err": _nan_stat(err_valid, np.mean),
            "peak_snr": _nan_stat(np.abs(flux), np.max) / median_err
            if np.isfinite(median_err) and median_err > EPS
            else np.nan,
            "median_snr": _nan_stat(snr, np.median),
            "mean_snr": _nan_stat(snr, np.mean),
            "noise_to_signal": median_err / raw_scale
            if np.isfinite(raw_scale) and raw_scale > EPS and np.isfinite(median_err)
            else np.nan,
            "relative_noise_p95": _nanpercentile(err_valid, 95) / raw_scale
            if np.isfinite(raw_scale) and raw_scale > EPS
            else np.nan,
            "has_flux_err": True,
        }
    )
    return features


def extract_sampling_features(t_train: Any) -> FeatureDict:
    """Extract sampling features from training times only.

    Uses only the times seen by the single-band GP during training.  Flux,
    complete-curve points, and test points are intentionally ignored.
    """
    features: FeatureDict = {
        "n_train": 0,
        "train_time_span": np.nan,
        "median_train_cadence": np.nan,
        "mean_train_cadence": np.nan,
        "max_train_gap": np.nan,
        "gap_90_train": np.nan,
        "gap_std_train": np.nan,
    }

    t = np.sort(_finite_values(t_train))
    features["n_train"] = int(t.size)
    features["train_time_span"] = _time_span(t)
    if t.size < 2:
        return features

    gaps = np.diff(t)
    gaps = gaps[np.greater(gaps, 0)]
    if gaps.size == 0:
        return features

    features.update(
        {
            "median_train_cadence": _nan_stat(gaps, np.median),
            "mean_train_cadence": _nan_stat(gaps, np.mean),
            "max_train_gap": _nan_stat(gaps, np.max),
            "gap_90_train": _nanpercentile(gaps, 90),
            "gap_std_train": _nan_stat(gaps, np.std),
        }
    )
    return features


def _nearest_distances(query_times: np.ndarray, reference_times: np.ndarray) -> np.ndarray:
    """Return nearest absolute distance from each query time to reference times."""
    if query_times.size == 0 or reference_times.size == 0:
        return np.array([], dtype=float)

    ref = np.sort(reference_times)
    insert = np.searchsorted(ref, query_times)
    left_idx = np.clip(insert - 1, 0, ref.size - 1)
    right_idx = np.clip(insert, 0, ref.size - 1)
    left_dist = np.abs(query_times - ref[left_idx])
    right_dist = np.abs(query_times - ref[right_idx])
    return np.minimum(left_dist, right_dist)


def extract_train_test_geometry_features(
    t_train: Any,
    t_test: Any,
    *,
    t_peak: float | None = None,
    near_peak_window: float = 50,
) -> FeatureDict:
    """Extract geometry features describing training positions vs test positions.

    Uses training times and test times from the same split used for GP
    evaluation.  If ``t_peak`` is provided, it may come from the complete
    observed curve; otherwise peak-related geometry features are NaN.
    """
    features: FeatureDict = {
        "n_test": 0,
        "test_time_span": np.nan,
        "outside_train_frac": np.nan,
        "test_before_train_frac": np.nan,
        "test_after_train_frac": np.nan,
        "nearest_train_distance_mean": np.nan,
        "nearest_train_distance_median": np.nan,
        "nearest_train_distance_max": np.nan,
        "nearest_train_distance_90": np.nan,
        "distance_peak_to_nearest_train": np.nan,
        "distance_peak_to_nearest_test": np.nan,
        "n_train_near_peak": np.nan,
        "n_test_near_peak": np.nan,
        "fraction_train_near_peak": np.nan,
        "fraction_test_near_peak": np.nan,
    }

    train = np.sort(_finite_values(t_train))
    test = np.sort(_finite_values(t_test))
    features["n_test"] = int(test.size)
    features["test_time_span"] = _time_span(test)

    if train.size > 0 and test.size > 0:
        train_min = np.min(train)
        train_max = np.max(train)
        before = test < train_min
        after = test > train_max
        features["test_before_train_frac"] = _fraction(before, test.size)
        features["test_after_train_frac"] = _fraction(after, test.size)
        features["outside_train_frac"] = _fraction(before | after, test.size)

        distances = _nearest_distances(test, train)
        features["nearest_train_distance_mean"] = _nan_stat(distances, np.mean)
        features["nearest_train_distance_median"] = _nan_stat(distances, np.median)
        features["nearest_train_distance_max"] = _nan_stat(distances, np.max)
        features["nearest_train_distance_90"] = _nanpercentile(distances, 90)

    if t_peak is not None and np.isfinite(t_peak):
        if train.size > 0:
            peak_train_distance = _nearest_distances(np.array([t_peak], dtype=float), train)
            features["distance_peak_to_nearest_train"] = _nan_stat(peak_train_distance, np.min)
            train_near = np.abs(train - t_peak) <= near_peak_window
            features["n_train_near_peak"] = int(np.sum(train_near))
            features["fraction_train_near_peak"] = _fraction(train_near, train.size)

        if test.size > 0:
            peak_test_distance = _nearest_distances(np.array([t_peak], dtype=float), test)
            features["distance_peak_to_nearest_test"] = _nan_stat(peak_test_distance, np.min)
            test_near = np.abs(test - t_peak) <= near_peak_window
            features["n_test_near_peak"] = int(np.sum(test_near))
            features["fraction_test_near_peak"] = _fraction(test_near, test.size)

    return features


def extract_object_band_features(
    object_id: Any,
    obj_type: Any,
    band: Any,
    t_all: Any,
    flux_all: Any,
    t_train: Any,
    flux_train: Any,
    t_test: Any | None = None,
    flux_test: Any | None = None,
    flux_err_all: Any | None = None,
    flux_err_train: Any | None = None,
    near_peak_window: float = 50,
    baseline_window: float = 200,
) -> FeatureDict:
    """Build one flat feature row for a single object-band GP analysis unit.

    Determines ``t_peak`` from complete observed times and normalized complete
    observed flux.  Morphology and roughness use complete observed data,
    sampling uses training times only, SNR uses original complete flux and
    original complete flux errors, and geometry uses the provided train/test
    split.  ``flux_train``, ``flux_test``, and ``flux_err_train`` are accepted
    to preserve object-band record structure, but split-dependent features here
    are time-geometry features.
    """
    del flux_train, flux_test, flux_err_train

    t_peak = _determine_peak_time(t_all, flux_all)
    features: FeatureDict = {
        "object_id": object_id,
        "obj_type": obj_type,
        "band": band,
        "t_peak": t_peak,
    }
    features.update(
        extract_morphology_features(
            t_all,
            flux_all,
            normalize=True,
            t_peak=t_peak,
            near_peak_window=near_peak_window,
            baseline_window=baseline_window,
        )
    )
    features.update(extract_fluctuation_features(t_all, flux_all, normalize=True))
    features.update(extract_snr_features(flux_all, flux_err_all))
    features.update(extract_sampling_features(t_train))
    features.update(
        extract_train_test_geometry_features(
            t_train,
            t_test,
            t_peak=t_peak if np.isfinite(t_peak) else None,
            near_peak_window=near_peak_window,
        )
    )
    return features


def _scalar_metadata(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first scalar-like metadata value found in a processed data dict."""
    for key in keys:
        if key not in data:
            continue
        value = data[key]
        arr = np.asarray(value, dtype=object)
        if arr.ndim == 0:
            return arr.item()
        if arr.size > 0:
            return arr.reshape(-1)[0]
    return default


def _raw_observation_arrays_from_processed(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return raw flux and raw flux errors from processed GP data.

    Uses ``singleGP_model._raw_observation_arrays`` when that module is
    importable.  If optional modeling dependencies such as scikit-learn are not
    available, falls back to the same raw-array reconstruction logic.
    """
    try:
        from singleGP_model import _raw_observation_arrays

        y_raw, yerr_raw, _, _ = _raw_observation_arrays(data)
        return _as_float_array(y_raw), _as_float_array(yerr_raw)
    except ModuleNotFoundError:
        scale = float(data.get("flux_scale", data.get("scale", 1.0)))
        background = float(data.get("background_flux", 0.0))
        y_norm = _as_float_array(data.get("y"))
        yerr_norm = _as_float_array(data.get("yerr"))
        y_raw = _as_float_array(data.get("y_raw", y_norm * scale + background))
        yerr_raw = _as_float_array(data.get("yerr_raw", yerr_norm * scale))
        return y_raw, yerr_raw


def _time_array_from_processed_data(data: dict[str, Any], *, use_raw_time: bool) -> np.ndarray:
    """Return raw/recovered or GP-space time from processed split data."""
    if use_raw_time:
        if "t_raw" in data:
            return _as_float_array(data["t_raw"])
        if "t_scale" in data and "alignment_peak_time" in data:
            t_scale = float(_scalar_metadata(data, "t_scale", default=1.0))
            alignment_peak_time = float(_scalar_metadata(data, "alignment_peak_time", default=0.0))
            return _as_float_array(data.get("t")) * t_scale + alignment_peak_time
    return _as_float_array(data.get("t"))


def _coverage_value(metrics: dict[str, Any], key: str) -> float:
    """Extract a coverage metric from flat or nested evaluation output."""
    if key in metrics:
        return metrics[key]
    coverage = metrics.get("coverage", {})
    if isinstance(coverage, dict) and key in coverage:
        return coverage[key]
    return np.nan


def _metrics_for_feature_row(metrics: dict[str, Any] | None) -> FeatureDict:
    """Normalize evaluate_heldout_metrics-style output to feature-table metric names."""
    if metrics is None:
        return {}

    z_values = _finite_values(metrics.get("z_values"))
    metric_row: FeatureDict = {
        "rmse": metrics.get("rmse", np.nan),
        "nrmse": metrics.get("nrmse", np.nan),
        "nlpd": metrics.get("nlpd", metrics.get("mean_nlpd", np.nan)),
        "crps": metrics.get("crps", metrics.get("mean_crps", np.nan)),
        "ncrps": metrics.get("ncrps", np.nan),
        "coverage_1sigma": _coverage_value(metrics, "coverage_1sigma"),
        "coverage_2sigma": _coverage_value(metrics, "coverage_2sigma"),
        "coverage_3sigma": _coverage_value(metrics, "coverage_3sigma"),
        "ks_pit": metrics.get("ks_pit", metrics.get("ks_pit_object", np.nan)),
        "object_flux_scale": metrics.get("object_flux_scale", np.nan),
        "z_mean": metrics.get("z_mean", np.mean(z_values) if z_values.size > 0 else np.nan),
        "z_std": metrics.get("z_std", np.std(z_values) if z_values.size > 0 else np.nan),
    }
    return metric_row


def make_feature_record_from_gp_split(
    train_data: dict[str, Any],
    heldout_data: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    *,
    use_raw_time: bool = True,
) -> FeatureDict:
    """Create a build_feature_table-compatible record from GP split artifacts.

    Uses processed GP split dictionaries such as those passed to and evaluated
    by ``evaluate_heldout_metrics``.  Raw flux and raw flux errors are recovered
    with ``singleGP_model._raw_observation_arrays``.  If ``use_raw_time=True``,
    times are recovered as original light-curve times when ``t_scale`` and
    ``alignment_peak_time`` are available; otherwise the GP-space ``t`` arrays
    are used.  The returned record is compatible with ``build_feature_table``.
    """
    heldout_data = {} if heldout_data is None else heldout_data

    train_flux_raw, train_err_raw = _raw_observation_arrays_from_processed(train_data)
    train_t = _time_array_from_processed_data(train_data, use_raw_time=use_raw_time)

    if heldout_data:
        test_flux_raw, test_err_raw = _raw_observation_arrays_from_processed(heldout_data)
        test_t = _time_array_from_processed_data(heldout_data, use_raw_time=use_raw_time)
    else:
        test_flux_raw = np.array([], dtype=float)
        test_err_raw = np.array([], dtype=float)
        test_t = np.array([], dtype=float)

    record: FeatureDict = {
        "object_id": _scalar_metadata(train_data, "obj_id", "object_id"),
        "obj_type": _scalar_metadata(train_data, "obj_type", "class"),
        "band": _scalar_metadata(train_data, "band"),
        "t_all": np.concatenate([train_t, test_t]),
        "flux_all": np.concatenate([_as_float_array(train_flux_raw), _as_float_array(test_flux_raw)]),
        "flux_err_all": np.concatenate([_as_float_array(train_err_raw), _as_float_array(test_err_raw)]),
        "t_train": train_t,
        "flux_train": train_flux_raw,
        "flux_err_train": train_err_raw,
        "t_test": test_t,
        "flux_test": test_flux_raw,
        "time_space": "raw" if use_raw_time else "gp",
    }
    if metrics is not None:
        record.update(_metrics_for_feature_row(metrics))
    return record


def _record_from_supported_input(record: dict[str, Any], record_index: int) -> FeatureDict:
    """Accept either a feature-ready record or a GP split/evaluation record."""
    required = ("t_all", "flux_all", "t_train", "flux_train")
    if all(key in record for key in required):
        return dict(record)

    if "train_data" in record:
        metrics = record.get("metrics", record.get("evaluation_metrics"))
        if metrics is None and any(key in record for key in ("rmse", "mean_nlpd", "coverage")):
            metrics = record

        converted = make_feature_record_from_gp_split(
            record["train_data"],
            record.get("heldout_data", record.get("test_data")),
            metrics,
            use_raw_time=record.get("use_raw_time", True),
        )
        for key, value in record.items():
            if key not in {"train_data", "heldout_data", "test_data", "metrics", "evaluation_metrics"}:
                converted.setdefault(key, value)
        return converted

    missing = [key for key in required if key not in record]
    raise KeyError(
        f"Record {record_index} is missing required keys {missing}. "
        "Provide raw feature arrays or train_data/heldout_data split artifacts."
    )


def _as_record_list(records: Any) -> list[Any]:
    """Allow callers to pass either one record dict or an iterable of record dicts."""
    if isinstance(records, pd.DataFrame):
        return records.to_dict("records")
    if isinstance(records, dict):
        return [records]
    return list(records)


def _array_to_json(values: Any) -> str:
    """Serialize a numeric array-like value for CSV storage."""
    arr = _as_float_array(values)
    return json.dumps(arr.tolist())


def _json_to_array(value: Any) -> np.ndarray:
    """Deserialize a numeric array stored by ``_array_to_json``."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.array([], dtype=float)
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return np.array([], dtype=float)
        return _as_float_array(json.loads(value))
    return _as_float_array(value)


def _deserialize_lightcurve_record(record: Any) -> FeatureDict:
    """Convert serialized array columns in a record back to NumPy arrays."""
    row = dict(record)
    for column in LIGHTCURVE_ARRAY_COLUMNS:
        if column in row:
            row[column] = _json_to_array(row[column])
    return row


def build_feature_table(records: Any) -> pd.DataFrame:
    """Build a feature table from object-band records.

    Accepts either a single record dictionary or an iterable of record
    dictionaries.  Records may already contain ``t_all``, ``flux_all``,
    ``t_train``, and ``flux_train``.  Alternatively, a record may contain
    ``train_data``, optional ``heldout_data``, and optional ``metrics`` from
    ``evaluate_heldout_metrics``; these are converted with
    ``make_feature_record_from_gp_split``.  Optional GP evaluation metrics such
    as RMSE, NLPD, CRPS, calibration coverage, z-std, and KS-PIT are copied into
    the returned rows.  No additional random split is introduced during feature
    extraction.
    """
    rows: list[dict[str, Any]] = []

    for i, record in enumerate(_as_record_list(records)):
        feature_record = _record_from_supported_input(record, i)

        row = extract_object_band_features(
            object_id=feature_record.get("object_id"),
            obj_type=feature_record.get("obj_type"),
            band=feature_record.get("band"),
            t_all=feature_record["t_all"],
            flux_all=feature_record["flux_all"],
            t_train=feature_record["t_train"],
            flux_train=feature_record["flux_train"],
            t_test=feature_record.get("t_test"),
            flux_test=feature_record.get("flux_test"),
            flux_err_all=feature_record.get("flux_err_all"),
            flux_err_train=feature_record.get("flux_err_train"),
            near_peak_window=feature_record.get("near_peak_window", 50),
            baseline_window=feature_record.get("baseline_window", 200),
        )

        if "time_space" in feature_record:
            row["time_space"] = feature_record["time_space"]
        for metric in GP_METRIC_COLUMNS:
            if metric in feature_record:
                row[metric] = feature_record[metric]
        rows.append(row)

    return pd.DataFrame(rows)


def build_lightcurve_record_table(records: Any) -> pd.DataFrame:
    """Build a CSV-friendly table that preserves light-curve arrays.

    This is different from ``build_feature_table``: it stores the complete,
    training, and held-out arrays as JSON strings so they can be loaded later
    for object-band inspection and plotting.
    """
    rows: list[dict[str, Any]] = []
    for i, record in enumerate(_as_record_list(records)):
        row = _record_from_supported_input(record, i)
        output: FeatureDict = {
            "object_id": row.get("object_id"),
            "obj_type": row.get("obj_type"),
            "band": row.get("band"),
            "time_space": row.get("time_space"),
        }
        for column in LIGHTCURVE_ARRAY_COLUMNS:
            if column in row:
                output[column] = _array_to_json(row[column])
        rows.append(output)
    return pd.DataFrame(rows)


def save_lightcurve_records_csv(
    records: Any,
    output_path: str | Path,
    *,
    index: bool = False,
) -> Path:
    """Save object-band light-curve records with arrays preserved."""
    table = build_lightcurve_record_table(records)
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=index)
    return path


def load_lightcurve_records_csv(input_path: str | Path) -> list[FeatureDict]:
    """Load light-curve records saved by ``save_lightcurve_records_csv``."""
    table = pd.read_csv(Path(input_path).expanduser())
    return [_deserialize_lightcurve_record(row) for row in table.to_dict("records")]


def build_records_lookup(records: Any) -> dict[tuple[Any, Any], FeatureDict]:
    """Build a lookup dictionary keyed by ``(object_id, band)``.

    Use this with the in-memory records produced by
    ``make_feature_record_from_gp_split``.  Unlike the compact feature table,
    these records keep the train and held-out arrays needed for inspecting a
    specific light curve.
    """
    lookup: dict[tuple[Any, Any], FeatureDict] = {}
    for i, record in enumerate(_as_record_list(records)):
        record = _deserialize_lightcurve_record(record)
        row = _record_from_supported_input(record, i)
        key = (row.get("object_id"), row.get("band"))
        lookup[key] = row
    return lookup


def get_object_band_lightcurve(
    records_or_lookup: Any,
    object_id: Any,
    band: Any,
    *,
    normalize: bool = False,
) -> FeatureDict:
    """Return stored complete/train/held-out data for one object and band.

    ``records_or_lookup`` can be either the original records list or the output
    of ``build_records_lookup``.  The returned dictionary contains ``t_all``,
    ``flux_all``, ``t_train``, ``flux_train``, ``t_test``, ``flux_test``, and
    ``flux_scale``.  If ``normalize=True``, normalized flux arrays are also
    included with ``_norm`` suffixes.
    """
    key = (object_id, band)
    if isinstance(records_or_lookup, dict) and key in records_or_lookup:
        row = records_or_lookup[key]
    else:
        row = build_records_lookup(records_or_lookup).get(key)

    if row is None:
        raise KeyError(f"No light curve found for object_id={object_id!r}, band={band!r}.")

    t_all = _as_float_array(row.get("t_all"))
    flux_all = _as_float_array(row.get("flux_all"))
    t_train = _as_float_array(row.get("t_train"))
    flux_train = _as_float_array(row.get("flux_train"))
    t_test = _as_float_array(row.get("t_test"))
    flux_test = _as_float_array(row.get("flux_test"))
    flux_scale = robust_flux_scale(flux_all)

    result: FeatureDict = {
        "object_id": row.get("object_id"),
        "obj_type": row.get("obj_type"),
        "band": row.get("band"),
        "t_all": t_all,
        "flux_all": flux_all,
        "t_train": t_train,
        "flux_train": flux_train,
        "t_test": t_test,
        "flux_test": flux_test,
        "flux_scale": flux_scale,
        "time_space": row.get("time_space"),
    }

    if normalize:
        if np.isfinite(flux_scale) and flux_scale > EPS:
            result["flux_all_norm"] = flux_all / flux_scale
            result["flux_train_norm"] = flux_train / flux_scale
            result["flux_test_norm"] = flux_test / flux_scale
        else:
            result["flux_all_norm"] = np.full(flux_all.shape, np.nan)
            result["flux_train_norm"] = np.full(flux_train.shape, np.nan)
            result["flux_test_norm"] = np.full(flux_test.shape, np.nan)

    return result


def plot_object_band_lightcurve(
    records_or_lookup: Any,
    object_id: Any,
    band: Any,
    *,
    normalize: bool = False,
    ax: Any = None,
    figsize: tuple[float, float] = (8, 4),
) -> Any:
    """Plot complete, training, and held-out points for one object and band."""
    curve = get_object_band_lightcurve(
        records_or_lookup,
        object_id,
        band,
        normalize=normalize,
    )

    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=figsize)

    flux_key = "flux_all_norm" if normalize else "flux_all"
    train_key = "flux_train_norm" if normalize else "flux_train"
    test_key = "flux_test_norm" if normalize else "flux_test"

    ax.scatter(curve["t_all"], curve[flux_key], color="0.75", s=28, label="all")
    ax.scatter(curve["t_train"], curve[train_key], color="tab:blue", s=42, label="train")
    if len(curve["t_test"]) > 0:
        ax.scatter(curve["t_test"], curve[test_key], color="tab:orange", s=48, label="held-out")

    ylabel = "normalized flux" if normalize else "flux"
    title = f"object_id={curve['object_id']}, band={curve['band']}"
    ax.set_title(title)
    ax.set_xlabel(f"time ({curve.get('time_space') or 'stored'})")
    ax.set_ylabel(ylabel)
    ax.legend()
    return ax


def save_feature_table_csv(
    data: pd.DataFrame,
    output_path: str | Path,
    *,
    index: bool = False,
) -> Path:
    """Save a feature table DataFrame to CSV.

    Takes the DataFrame returned by ``build_feature_table`` or any compatible
    feature/metric table, creates the parent directory when needed, writes a CSV
    file, and returns the resolved output path.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=index)
    return path


def load_feature_table_csv(
    input_path: str | Path,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Load a saved feature table CSV back into a pandas DataFrame.

    This is the inverse convenience helper for ``save_feature_table_csv``.
    Extra keyword arguments are passed directly to ``pd.read_csv`` for cases
    where callers want custom dtypes, NA handling, or parsing options.
    """
    path = Path(input_path).expanduser()
    return pd.read_csv(path, **read_csv_kwargs)


def add_quantile_groups(
    df: pd.DataFrame,
    columns: Iterable[str],
    q: int = 3,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Add quantile group columns for feature-stratified analysis.

    Uses ``pd.qcut`` on each requested column and writes ``"{column}_group"``.
    For q=3, default labels are ``["low", "medium", "high"]``.  Duplicate
    bin edges and too-few distinct values are handled gracefully by leaving
    unavailable groups as NaN.
    """
    result = df.copy()
    if labels is None and q == 3:
        labels = ["low", "medium", "high"]

    for column in columns:
        group_column = f"{column}_group"
        if column not in result:
            result[group_column] = np.nan
            continue

        values = pd.to_numeric(result[column], errors="coerce")
        valid = values.dropna()
        if valid.nunique() < 2:
            result[group_column] = np.nan
            continue

        try:
            # First compute the actual number of bins after duplicate edges are dropped.
            _, bins = pd.qcut(
                values,
                q=q,
                retbins=True,
                duplicates="drop"
            )

            n_bins = len(bins) - 1

            if n_bins < 1:
                result[group_column] = np.nan
                continue

            if labels is not None and len(labels) >= n_bins:
                use_labels = labels[:n_bins]
            else:
                use_labels = None

            result[group_column] = pd.qcut(
                values,
                q=q,
                labels=use_labels,
                duplicates="drop"
            )
        except ValueError:
            result[group_column] = np.nan

    return result


def summarize_metrics_by_group(
    df: pd.DataFrame,
    group_col: str,
    metric_cols: Iterable[str],
    feature_cols: Iterable[str] | None = None,
    object_col: str = "object_id",
) -> pd.DataFrame:
    """Summarize GP metrics by a feature group column.

    Parameters
    ----------
    df:
        DataFrame containing feature columns, group columns, object IDs, and metrics.

    group_col:
        Categorical group column, for example "total_variation_group".

    metric_cols:
        GP metric columns to summarize, for example
        ["rmse", "nlpd", "crps", "z_std", "ks_pit"].

    feature_cols:
        Original continuous feature columns to summarize inside each group.
        For example, if group_col is "total_variation_group",
        pass feature_cols=["total_variation"].

        If feature_cols is None and group_col ends with "_group",
        the function tries to infer the feature column automatically.

    object_col:
        Column used to count unique objects. Default is "object_id".

    Returns
    -------
    pd.DataFrame
        One row per group, including:
        - n_rows
        - n_objects, if object_col exists
        - feature means / medians / stds
        - metric means / medians / stds
    """

    if group_col not in df:
        raise KeyError(f"Group column not found: {group_col}")
    result = df.copy()
    # Infer feature column from group column, e.g.
    # "total_variation_group" -> "total_variation"
    if feature_cols is None:
        if group_col.endswith("_group"):
            inferred_feature = group_col.removesuffix("_group")
            if inferred_feature in result.columns:
                feature_cols = [inferred_feature]
            else:
                feature_cols = []
        else:
            feature_cols = []
    else:
        feature_cols = list(feature_cols)
    
    metric_cols = [col for col in metric_cols if col in df]
    grouped = df.groupby(group_col, observed=False, dropna=False)
    summary = grouped.size().to_frame("n_rows")

    if object_col in result.columns:
        summary["n_objects"] = grouped[object_col].nunique(dropna=True)

    # Summarize original feature values inside each group.
    for feature in feature_cols:
        numeric_feature = pd.to_numeric(result[feature], errors="coerce")

        feature_summary = (
            numeric_feature.to_frame(feature)
            .groupby(result[group_col], observed=False, dropna=False)
            .agg(
                mean=(feature, "mean"),
                std=(feature, "std"),
                median=(feature, "median"),
                q25=(feature, lambda x: _nanpercentile(x, 25)),
                q75=(feature, lambda x: _nanpercentile(x, 75)),
                min=(feature, "min"),
                max=(feature, "max"),
            )
        )

        feature_summary.columns = [
            f"{feature}_mean",
            f"{feature}_std",
            f"{feature}_median",
            f"{feature}_q25",
            f"{feature}_q75",
            f"{feature}_min",
            f"{feature}_max",
        ]

        summary = summary.join(feature_summary)

    # Summarize GP metrics inside each group.
    for metric in metric_cols:
        numeric_metric = pd.to_numeric(result[metric], errors="coerce")

        metric_summary = (
            numeric_metric.to_frame(metric)
            .groupby(result[group_col], observed=False, dropna=False)
            .agg(
                mean=(metric, "mean"),
                std=(metric, "std"),
                median=(metric, "median"),
                q25=(metric, lambda x: _nanpercentile(x, 25)),
                q75=(metric, lambda x: _nanpercentile(x, 75)),
                min=(metric, "min"),
                max=(metric, "max"),
            )
        )

        metric_summary.columns = [
            f"{metric}_mean",
            f"{metric}_std",
            f"{metric}_median",
            f"{metric}_q25",
            f"{metric}_q75",
            f"{metric}_min",
            f"{metric}_max",
        ]

        summary = summary.join(metric_summary)

    return summary.reset_index()


def summarize_feature_bins(
    df: pd.DataFrame,
    feature: str,
    metric_cols: Iterable[str],
    *,
    q: int = 3,
    labels: list[str] | None = None,
    object_col: str = "object_id",
) -> pd.DataFrame:
    """Group one feature into quantile bins and summarize metrics per bin.

    This is a convenience wrapper for the common analysis pattern:
    ``add_quantile_groups`` followed by ``summarize_metrics_by_group``.  For a
    feature such as ``"flat_fraction"``, the returned table includes
    ``"flat_fraction_group"``, ``"flat_fraction_mean"``, row/object counts, and
    summary statistics for each requested metric.
    """
    if feature not in df:
        raise KeyError(f"Feature column not found: {feature}")

    grouped_df = add_quantile_groups(df, [feature], q=q, labels=labels)
    group_col = f"{feature}_group"
    return summarize_metrics_by_group(
        grouped_df,
        group_col=group_col,
        metric_cols=metric_cols,
        feature_cols=[feature],
        object_col=object_col,
    )


def screen_features_by_spearman(df, feature_cols, target_col, min_n=30):
    rows = []

    for feature in feature_cols:
        if feature not in df or target_col not in df:
            continue

        tmp = df[[feature, target_col]].copy()
        tmp[feature] = pd.to_numeric(tmp[feature], errors="coerce")
        tmp[target_col] = pd.to_numeric(tmp[target_col], errors="coerce")
        tmp = tmp.dropna()

        if len(tmp) < min_n:
            continue

        if tmp[feature].nunique() < 2 or tmp[target_col].nunique() < 2:
            continue

        rho = tmp[[feature, target_col]].corr(method="spearman").iloc[0, 1]

        rows.append({
            "feature": feature,
            "target": target_col,
            "spearman_rho": rho,
            "abs_spearman_rho": abs(rho),
            "n": len(tmp),
        })

    result = pd.DataFrame(rows)
    return result.sort_values("abs_spearman_rho", ascending=False)
