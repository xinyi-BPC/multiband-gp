import numpy as np
from singleGP_model import predict_observation_distribution, _raw_observation_arrays

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
