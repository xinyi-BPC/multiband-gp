"""Internal diagnostics for single-band GP held-out predictions.

This module works on point-level prediction rows that already exist after GP
evaluation.  It does not split data, refit a GP, or create external light-curve
features.  The goal is to compare actual prediction error against the
uncertainty scale used for z/PIT/NLPD-style calibration checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd


TRUE_VALUE_COLUMNS = ("y_true", "flux_test", "y_test")
PREDICTED_MEAN_COLUMNS = ("mu_pred", "y_pred", "pred_mean", "mu")
PREDICTIVE_SIGMA_COLUMNS = ("sigma_pred", "pred_sigma", "std_pred", "sigma", "y_std")
YERR_COLUMNS = ("yerr_test", "yerr", "flux_err_test", "flux_err", "y_error")
OBJECT_ID_COLUMNS = ("object_id", "obj_id", "id")
OBJ_TYPE_COLUMNS = ("obj_type", "object_type", "class", "type")
BAND_COLUMNS = ("band", "filter")
EPS = 1e-12


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...], *, required: bool, role: str) -> str | None:
    """Return the first matching column name from a list of common alternatives."""
    for column in candidates:
        if column in df.columns:
            return column
    if required:
        available = ", ".join(map(str, df.columns))
        expected = ", ".join(candidates)
        raise ValueError(f"Could not find {role} column. Tried: {expected}. Available columns: {available}")
    return None


def _numeric_series(df: pd.DataFrame, column: str | None) -> pd.Series:
    """Return a numeric series, or all-NaN values when an optional column is absent."""
    if column is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return cast(pd.Series, pd.to_numeric(df[column], errors="coerce"))


def _valid_positive_series(values: pd.Series) -> pd.Series:
    """Keep finite positive values and mark invalid uncertainty values as NaN."""
    mask = np.isfinite(values.to_numpy(dtype=float)) & (values.to_numpy(dtype=float) > 0)
    return values.where(mask, np.nan)


def _finite_values(values: Any) -> np.ndarray:
    """Return finite one-dimensional float values."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _nan_mean(values: Any) -> float:
    values = _finite_values(values)
    if values.size == 0:
        return np.nan
    return float(np.mean(values))


def _nan_median(values: Any) -> float:
    values = _finite_values(values)
    if values.size == 0:
        return np.nan
    return float(np.median(values))


def _nan_percentile(values: Any, q: float) -> float:
    values = _finite_values(values)
    if values.size == 0:
        return np.nan
    return float(np.percentile(values, q))


def _nan_std(values: Any) -> float:
    values = _finite_values(values)
    if values.size == 0:
        return np.nan
    return float(np.std(values))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < EPS:
        return np.nan
    return float(numerator / denominator)


def _first_available_value(group: pd.DataFrame, column: str | None) -> Any:
    """Return the first non-null group value for optional metadata."""
    if column is None:
        return np.nan
    values = group[column].dropna()
    if values.empty:
        return np.nan
    return values.iloc[0]


def _as_1d_array(value: Any, n_values: int, *, default: Any = np.nan, dtype: Any = object) -> np.ndarray:
    """Return value as a one-dimensional array aligned to n_values."""
    if value is None:
        return np.full(n_values, default, dtype=dtype)
    values = np.asarray(value, dtype=dtype)
    if values.ndim == 0:
        return np.full(n_values, values.item(), dtype=dtype)
    values = values.reshape(-1)
    if len(values) == n_values:
        return values
    if len(values) == 1:
        return np.full(n_values, values[0], dtype=dtype)
    raise ValueError(f"Cannot align value of length {len(values)} to {n_values} prediction points.")


def _first_present_mapping_value(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Return the first present mapping value for one of the requested keys."""
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _lookup_class_labels(object_ids: np.ndarray, class_lookup: Any) -> np.ndarray:
    """Map object IDs to class labels when labels are stored outside prediction results."""
    if class_lookup is None:
        return np.full(len(object_ids), np.nan, dtype=object)

    if isinstance(class_lookup, pd.DataFrame):
        object_col = _find_column(class_lookup, OBJECT_ID_COLUMNS, required=True, role="lookup object id")
        class_col = _find_column(class_lookup, OBJ_TYPE_COLUMNS, required=True, role="lookup class label")
        lookup = dict(zip(class_lookup[object_col], class_lookup[class_col], strict=False))
    else:
        lookup = dict(class_lookup)

    return np.asarray([lookup.get(object_id, np.nan) for object_id in object_ids], dtype=object)


def prediction_dataframe_from_results(
    object_results: Any,
    *,
    class_lookup: Any = None,
) -> pd.DataFrame:
    """Build a point-level diagnostic input DataFrame from evaluation result dicts.

    ``evaluate_heldout_metrics`` already returns point-level arrays for
    ``y_true``, ``y_pred``, ``y_std``, ``yerr``, ``object_id``, and ``band``.
    It does not always include the class label.  If each result dict has
    ``obj_type`` or ``class``, this helper preserves it.  Otherwise pass
    ``class_lookup`` as either a dict keyed by object ID or a DataFrame with an
    object-id column and a class column.
    """
    rows: list[pd.DataFrame] = []
    for result in object_results:
        y_true = np.asarray(result["y_true"], dtype=float).reshape(-1)
        n_values = len(y_true)
        object_id = _as_1d_array(_first_present_mapping_value(result, OBJECT_ID_COLUMNS), n_values)
        class_value = _first_present_mapping_value(result, OBJ_TYPE_COLUMNS)
        obj_type = _as_1d_array(class_value, n_values)
        if pd.isna(pd.Series(obj_type)).all():
            obj_type = _lookup_class_labels(object_id, class_lookup)

        rows.append(
            pd.DataFrame(
                {
                    "object_id": object_id,
                    "obj_type": obj_type,
                    "band": _as_1d_array(_first_present_mapping_value(result, BAND_COLUMNS), n_values),
                    "y_true": y_true,
                    "y_pred": np.asarray(result["y_pred"], dtype=float).reshape(-1),
                    "y_std": np.asarray(result["y_std"], dtype=float).reshape(-1),
                    "yerr": _as_1d_array(result.get("yerr"), n_values, dtype=float),
                }
            )
        )

    if not rows:
        return pd.DataFrame(columns=["object_id", "obj_type", "band", "y_true", "y_pred", "y_std", "yerr"])
    return pd.concat(rows, ignore_index=True)


def _diagnostic_group_row(
    group: pd.DataFrame,
    *,
    object_col: str,
    obj_type_col: str | None,
    band_col: str | None,
) -> dict[str, Any]:
    """Aggregate point-level error and uncertainty diagnostics for one object-band."""
    error = group["_gp_diag_error"].to_numpy(dtype=float)
    squared_error = group["_gp_diag_squared_error"].to_numpy(dtype=float)
    sigma_used = group["_gp_diag_sigma_used"].to_numpy(dtype=float)
    z = group["_gp_diag_z"].to_numpy(dtype=float)

    rmse = np.sqrt(_nan_mean(squared_error))
    mae = _nan_mean(np.abs(error))
    sigma_rms = np.sqrt(_nan_mean(sigma_used ** 2))
    sigma_median = _nan_median(sigma_used)
    z_std = _nan_std(z)

    return {
        "object_id": _first_available_value(group, object_col),
        "obj_type": _first_available_value(group, obj_type_col),
        "band": _first_available_value(group, band_col),
        "n_test": int(len(group)),
        "rmse": float(rmse) if np.isfinite(rmse) else np.nan,
        "mae": mae,
        "sigma_rms": float(sigma_rms) if np.isfinite(sigma_rms) else np.nan,
        "sigma_mean": _nan_mean(sigma_used),
        "sigma_median": sigma_median,
        "sigma_q90": _nan_percentile(sigma_used, 90),
        "z_mean": _nan_mean(z),
        "z_std": z_std,
        "z_error": z_std - 1.0 if np.isfinite(z_std) else np.nan,
        "mean_abs_z": _nan_mean(np.abs(z)),
        "rmse_sigma_ratio": _safe_ratio(rmse, sigma_rms),
        "mae_sigma_ratio": _safe_ratio(mae, sigma_median),
    }


def compute_gp_internal_diagnostics(
    pred_df: pd.DataFrame,
    *,
    use_observation_noise: bool = False,
    yerr_col: str | None = None,
) -> pd.DataFrame:
    """Summarize GP mean-error and uncertainty-calibration diagnostics.

    Parameters
    ----------
    pred_df:
        Point-level held-out prediction DataFrame.  Each row should correspond
        to one evaluated point and include object identity, true value,
        predictive mean, and predictive standard deviation.  Common column-name
        alternatives are detected automatically.

    use_observation_noise:
        If ``False`` (default), use the predictive sigma column as-is.  This is
        the right choice when the column already matches the sigma used for the
        existing z/PIT/NLPD evaluation, such as ``y_std`` returned by
        ``evaluate_heldout_metrics``.  If ``True``, compute
        ``sigma_total = sqrt(sigma_pred**2 + yerr_test**2)``.  Use this only
        when evaluating noisy observed flux and the sigma column is GP latent
        predictive uncertainty without held-out observation noise included.

    yerr_col:
        Optional measurement-error column to use when
        ``use_observation_noise=True``.  If omitted, common names such as
        ``yerr_test`` and ``yerr`` are tried.  The measurement error must be in
        the same units/normalization as ``y_true``, ``mu_pred``, and
        ``sigma_pred``.

    Returns
    -------
    pd.DataFrame
        One row per object-band with RMSE/MAE, uncertainty scale summaries,
        standardized residual summaries, and error-to-uncertainty ratios.
    """
    if not isinstance(pred_df, pd.DataFrame):
        pred_df = pd.DataFrame(pred_df)
    if pred_df.empty:
        return pd.DataFrame(
            columns=[
                "object_id",
                "obj_type",
                "band",
                "n_test",
                "rmse",
                "mae",
                "sigma_rms",
                "sigma_mean",
                "sigma_median",
                "sigma_q90",
                "z_mean",
                "z_std",
                "z_error",
                "mean_abs_z",
                "rmse_sigma_ratio",
                "mae_sigma_ratio",
            ]
        )

    object_col = _find_column(pred_df, OBJECT_ID_COLUMNS, required=True, role="object id")
    obj_type_col = _find_column(pred_df, OBJ_TYPE_COLUMNS, required=False, role="object type")
    band_col = _find_column(pred_df, BAND_COLUMNS, required=False, role="band")
    true_col = _find_column(pred_df, TRUE_VALUE_COLUMNS, required=True, role="true value")
    pred_col = _find_column(pred_df, PREDICTED_MEAN_COLUMNS, required=True, role="predicted mean")
    sigma_col = _find_column(pred_df, PREDICTIVE_SIGMA_COLUMNS, required=True, role="predictive sigma")

    if use_observation_noise:
        yerr_col = yerr_col or _find_column(pred_df, YERR_COLUMNS, required=False, role="test observation error")
        if yerr_col is None:
            expected = ", ".join(YERR_COLUMNS)
            raise ValueError(
                "use_observation_noise=True requires a held-out measurement-error column. "
                f"Pass yerr_col explicitly or provide one of: {expected}."
            )

    work = pred_df.copy()
    y_true = _numeric_series(work, true_col)
    mu_pred = _numeric_series(work, pred_col)
    sigma_pred = _valid_positive_series(_numeric_series(work, sigma_col))

    if use_observation_noise:
        yerr = _numeric_series(work, yerr_col)
        yerr_mask = np.isfinite(yerr.to_numpy(dtype=float)) & (yerr.to_numpy(dtype=float) >= 0)
        yerr = yerr.where(yerr_mask, np.nan)
        sigma_used = (sigma_pred.pow(2) + yerr.pow(2)).pow(0.5)
    else:
        sigma_used = sigma_pred
    sigma_used = _valid_positive_series(sigma_used)

    error = y_true - mu_pred
    squared_error = error ** 2
    z = error / sigma_used
    z = z.where(np.isfinite(z.to_numpy(dtype=float)), np.nan)

    work["_gp_diag_error"] = error
    work["_gp_diag_squared_error"] = squared_error
    work["_gp_diag_sigma_used"] = sigma_used
    work["_gp_diag_z"] = z
    work["_gp_diag_abs_z"] = np.abs(z)

    group_cols = [object_col]
    if band_col is not None:
        group_cols.append(band_col)

    rows: list[dict[str, Any]] = []
    group_key: str | list[str] = group_cols[0] if len(group_cols) == 1 else group_cols
    for _, group_df in work.groupby(group_key, dropna=False, sort=False):
        rows.append(
            _diagnostic_group_row(
                cast(pd.DataFrame, group_df),
                object_col=object_col,
                obj_type_col=obj_type_col,
                band_col=band_col,
            )
        )
    return pd.DataFrame(rows)


def _plot_diagonal_reference(ax: Any, x: pd.Series, y: pd.Series) -> None:
    """Draw the RMSE = RMS sigma reference over the visible finite range."""
    values = pd.concat([x, y], ignore_index=True)
    values = pd.to_numeric(values, errors="coerce")
    values = values[np.isfinite(values) & (values > 0)]
    if values.empty:
        return
    lower = float(values.min())
    upper = float(values.max())
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="black", linewidth=1, label="RMSE = RMS sigma")


def _prepare_xy_for_plot(
    diag_df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    log_scale: bool,
) -> pd.DataFrame:
    """Return rows with finite x/y values, filtering non-positive values for log plots."""
    if x_col not in diag_df or y_col not in diag_df:
        missing = [column for column in (x_col, y_col) if column not in diag_df]
        raise ValueError(f"Missing required plotting column(s): {missing}")

    plot_df = diag_df.copy()
    plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    mask = np.isfinite(plot_df[x_col]) & np.isfinite(plot_df[y_col])
    if log_scale:
        mask &= (plot_df[x_col] > 0) & (plot_df[y_col] > 0)
    return plot_df.loc[mask].copy()


def plot_rmse_vs_sigma_by_class(
    diag_df: pd.DataFrame,
    *,
    class_col: str = "obj_type",
    x_col: str = "sigma_rms",
    y_col: str = "rmse",
    log_scale: bool = True,
    save_path: str | Path | None = None,
) -> Any:
    """Plot actual error scale against predicted uncertainty scale by class.

    Points near the dashed diagonal have an error scale consistent with the GP
    uncertainty.  Points above it tend to be overconfident; points below it tend
    to be underconfident.  Upper-right points near the diagonal are difficult
    object-bands where the GP uncertainty is aware of the difficulty.
    """
    import matplotlib.pyplot as plt

    plot_df = _prepare_xy_for_plot(diag_df, x_col=x_col, y_col=y_col, log_scale=log_scale)
    fig, ax = plt.subplots(figsize=(7, 5))

    if class_col in plot_df.columns:
        classes = plot_df[class_col].astype("object").where(plot_df[class_col].notna(), "missing")
        for class_value, class_df in plot_df.groupby(classes, sort=False):
            ax.scatter(class_df[x_col], class_df[y_col], alpha=0.75, label=str(class_value), edgecolors="none")
        ax.legend(title=class_col, fontsize="small", title_fontsize="small", loc="best")
    else:
        ax.scatter(plot_df[x_col], plot_df[y_col], alpha=0.75, edgecolors="none")

    _plot_diagonal_reference(ax, plot_df[x_col], plot_df[y_col])
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("Predicted uncertainty scale: RMS sigma")
    ax.set_ylabel("Actual error scale: RMSE")
    ax.set_title("Mean prediction error versus GP predictive uncertainty")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(Path(save_path), dpi=200, bbox_inches="tight")
    else:
        plt.show()
    return fig, ax


def plot_rmse_vs_sigma_colored_by_zerror(
    diag_df: pd.DataFrame,
    *,
    x_col: str = "sigma_rms",
    y_col: str = "rmse",
    z_col: str = "z_error",
    log_scale: bool = True,
    save_path: str | Path | None = None,
) -> Any:
    """Plot error scale versus uncertainty scale, colored by calibration error."""
    import matplotlib.pyplot as plt

    if z_col not in diag_df:
        raise ValueError(f"Missing required color column: {z_col}")
    plot_df = _prepare_xy_for_plot(diag_df, x_col=x_col, y_col=y_col, log_scale=log_scale)
    plot_df[z_col] = pd.to_numeric(plot_df[z_col], errors="coerce")

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        plot_df[x_col],
        plot_df[y_col],
        c=plot_df[z_col],
        cmap="coolwarm",
        alpha=0.8,
        edgecolors="none",
    )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("z_error = z_std - 1")

    _plot_diagonal_reference(ax, plot_df[x_col], plot_df[y_col])
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("Predicted uncertainty scale: RMS sigma")
    ax.set_ylabel("Actual error scale: RMSE")
    ax.set_title("Mean prediction error versus GP predictive uncertainty")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(Path(save_path), dpi=200, bbox_inches="tight")
    else:
        plt.show()
    return fig, ax


def run_gp_internal_diagnostic_analysis(
    pred_df: pd.DataFrame,
    output_dir: str | Path | None = None,
    use_observation_noise: bool = False,
    yerr_col: str | None = None,
) -> dict[str, Any]:
    """Compute diagnostics, print a compact summary, and generate diagnostic plots."""
    diag_df = compute_gp_internal_diagnostics(
        pred_df,
        use_observation_noise=use_observation_noise,
        yerr_col=yerr_col,
    )

    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
        diag_df.to_csv(output_path / "gp_internal_diagnostics.csv", index=False)

    summary = {
        "n_object_band_rows": int(len(diag_df)),
        "median_rmse": _nan_median(diag_df.get("rmse", pd.Series(dtype=float))),
        "median_sigma_rms": _nan_median(diag_df.get("sigma_rms", pd.Series(dtype=float))),
        "median_rmse_sigma_ratio": _nan_median(diag_df.get("rmse_sigma_ratio", pd.Series(dtype=float))),
        "median_z_error": _nan_median(diag_df.get("z_error", pd.Series(dtype=float))),
    }

    print("GP internal diagnostic summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    class_plot_path = output_path / "rmse_vs_sigma_by_class.png" if output_path is not None else None
    zerror_plot_path = output_path / "rmse_vs_sigma_zerror.png" if output_path is not None else None
    fig_class, ax_class = plot_rmse_vs_sigma_by_class(diag_df, save_path=class_plot_path)
    fig_zerror, ax_zerror = plot_rmse_vs_sigma_colored_by_zerror(diag_df, save_path=zerror_plot_path)

    return {
        "diagnostics": diag_df,
        "summary": summary,
        "figures": {
            "rmse_vs_sigma_by_class": fig_class,
            "rmse_vs_sigma_zerror": fig_zerror,
        },
        "axes": {
            "rmse_vs_sigma_by_class": ax_class,
            "rmse_vs_sigma_zerror": ax_zerror,
        },
    }
