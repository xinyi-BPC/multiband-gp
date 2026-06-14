import numpy as np
from singleGP_model import predict_observation_distribution, _raw_observation_arrays
PREDICTIVE_STD_EPSILON = 1e-12

def percentile_abs_difference(a, b, percentiles=(50, 90, 99)):
    """
    Summarize absolute differences between two same-shaped arrays.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"Array shapes must match: {a.shape} != {b.shape}")

    return np.percentile(np.abs(a - b), percentiles)


def compare_preprocessed_flux_arrays(
        data_no_filter,
        data_filter,
        percentiles=(50, 90, 99),
        print_summary=True,
):
    """
    Compare normalized and raw flux arrays from two preprocessing modes.

    The two data dictionaries should refer to the same object, band, and split.
    """
    y_no_filter_norm = np.asarray(data_no_filter["y"])
    y_filter_norm = np.asarray(data_filter["y"])
    y_no_filter_raw, yerr_no_filter_raw, _, _ = _raw_observation_arrays(data_no_filter)
    y_filter_raw, yerr_filter_raw, _, _ = _raw_observation_arrays(data_filter)

    summary = {
        "y_norm_abs_diff_percentiles": percentile_abs_difference(
            y_no_filter_norm,
            y_filter_norm,
            percentiles,
        ),
        "y_raw_abs_diff_percentiles": percentile_abs_difference(
            y_no_filter_raw,
            y_filter_raw,
            percentiles,
        ),
        "yerr_raw_abs_diff_percentiles": percentile_abs_difference(
            yerr_no_filter_raw,
            yerr_filter_raw,
            percentiles,
        ),
        "percentiles": tuple(percentiles),
    }
    if print_summary:
        print("abs(y_no_filter_norm - y_filter_norm):", summary["y_norm_abs_diff_percentiles"])
        print("abs(y_no_filter_raw - y_filter_raw):", summary["y_raw_abs_diff_percentiles"])
        print("abs(yerr_no_filter_raw - yerr_filter_raw):", summary["yerr_raw_abs_diff_percentiles"])

    return summary


def compare_raw_prediction_distributions(
        gp_no_filter,
        data_no_filter,
        gp_filter,
        data_filter,
        percentiles=(50, 90, 99),
        include_yerr=True,
        yerr_scale=1.0,
        noise_floor=0.0,
        extra_noise=None,
        print_summary=True,
):
    """
    Compare raw-space GP predictive means and standard deviations.
    """
    mu_no_filter_raw, std_no_filter_raw, var_no_filter_raw = predict_observation_distribution(
        gp_no_filter,
        data_no_filter,
        include_yerr=include_yerr,
        yerr_scale=yerr_scale,
        noise_floor=noise_floor,
        extra_noise=extra_noise,
        return_raw_flux=True,
    )
    mu_filter_raw, std_filter_raw, var_filter_raw = predict_observation_distribution(
        gp_filter,
        data_filter,
        include_yerr=include_yerr,
        yerr_scale=yerr_scale,
        noise_floor=noise_floor,
        extra_noise=extra_noise,
        return_raw_flux=True,
    )

    summary = {
        "mu_raw_abs_diff_percentiles": percentile_abs_difference(
            mu_no_filter_raw,
            mu_filter_raw,
            percentiles,
        ),
        "std_raw_abs_diff_percentiles": percentile_abs_difference(
            std_no_filter_raw,
            std_filter_raw,
            percentiles,
        ),
        "var_raw_abs_diff_percentiles": percentile_abs_difference(
            var_no_filter_raw,
            var_filter_raw,
            percentiles,
        ),
        "percentiles": tuple(percentiles),
    }
    if print_summary:
        print("abs(mu_no_filter_raw - mu_filter_raw):", summary["mu_raw_abs_diff_percentiles"])
        print("abs(std_no_filter_raw - std_filter_raw):", summary["std_raw_abs_diff_percentiles"])
        print("abs(var_no_filter_raw - var_filter_raw):", summary["var_raw_abs_diff_percentiles"])

    return summary


def largest_standardized_residual_cases(object_results, top_n=20):
    """
    Return the held-out points with the largest absolute standardized residuals.
    Useful for debugging bad predictions, outliers, extrapolation failures, or underestimated uncertainty.
    """
    if len(object_results) == 0:
        raise ValueError("object_results must contain at least one result.")

    rows = []
    for result_idx, result in enumerate(object_results):
        y_true = np.asarray(result["y_true"])
        y_pred = np.asarray(result["y_pred"])
        y_std = np.maximum(np.asarray(result["y_std"]), PREDICTIVE_STD_EPSILON)
        z = (y_true - y_pred) / y_std
        abs_z = np.abs(z)

        for point_idx in range(len(y_true)):
            rows.append({
                "result_idx": result_idx,
                "point_idx": point_idx,
                "object_id": result["object_id"][point_idx],
                "band": result["band"][point_idx],
                "time": result["time"][point_idx],
                "X_test": result["X_test"][point_idx],
                "y": y_true[point_idx],
                "mean": y_pred[point_idx],
                "std": y_std[point_idx],
                "z": z[point_idx],
                "abs_z": abs_z[point_idx],
                "yerr": result["yerr"][point_idx],
                "outside_train_range": result["outside_train_range"][point_idx],
                "distance_to_train_range": result["distance_to_train_range"][point_idx],
                "near_peak": result["near_peak"][point_idx],
                "n_train": result["n_train"],
                "n_heldout": result["n_heldout"],
                "train_time_min": result["train_time_min"],
                "train_time_max": result["train_time_max"],
            })

    # Sort by absolute standardized residual and return the top cases
    rows = sorted(rows, key=lambda row: row["abs_z"], reverse=True)
    return rows[:top_n]


def print_largest_standardized_residual_cases(object_results, top_n=20):
    """
    Print the largest standardized residual cases in a notebook-friendly format.
    """
    rows = largest_standardized_residual_cases(object_results, top_n=top_n)
    for row in rows:
        print(
            "object_id:", row["object_id"],
            "band:", row["band"],
            "time:", row["time"],
            "X_test:", row["X_test"],
            "y:", row["y"],
            "mean:", row["mean"],
            "std:", row["std"],
            "z:", row["z"],
            "yerr:", row["yerr"],
            "outside_train_range:", row["outside_train_range"],
            "near_peak:", row["near_peak"],
            "n_train:", row["n_train"],
        )
    return rows


def collect_heldout_predictions(object_results):
    """
    Concatenate held-out y_true, y_pred, and y_std arrays across objects.
    """
    if len(object_results) == 0:
        raise ValueError("object_results must contain at least one result.")

    y_true = np.concatenate([np.asarray(result["y_true"]) for result in object_results])
    y_pred = np.concatenate([np.asarray(result["y_pred"]) for result in object_results])
    y_std = np.concatenate([np.asarray(result["y_std"]) for result in object_results])

    return y_true, y_pred, y_std


def yerr_statistics(object_results):
    """
    Summarize held-out measurement errors and their relationship to residuals.
    """
    if len(object_results) == 0:
        raise ValueError("object_results must contain at least one result.")

    yerr = np.concatenate([np.asarray(result["yerr"]) for result in object_results])
    y_true, y_pred, y_std = collect_heldout_predictions(object_results)
    residual = y_true - y_pred
    z = residual / np.maximum(y_std, 1e-12)
    abs_z = np.abs(z)

    return {
        "min_yerr": float(np.min(yerr)),
        "median_yerr": float(np.median(yerr)),
        "mean_yerr": float(np.mean(yerr)),
        "p95_yerr": float(np.percentile(yerr, 95)),
        "max_yerr": float(np.max(yerr)),
        "median_yerr_top_1pct_abs_z": float(np.median(yerr[abs_z >= np.percentile(abs_z, 99)])),
        "median_yerr_top_5pct_abs_z": float(np.median(yerr[abs_z >= np.percentile(abs_z, 95)])),
    }
