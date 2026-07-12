"""Class-level inter-band distances from irregular multiband light curves."""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    # Reuse the project's central mapping without making GP dependencies mandatory.
    from MOGP_model import DEFAULT_BAND_TO_WAVELENGTH, _resolve_wavelength
except ImportError:
    DEFAULT_BAND_TO_WAVELENGTH = {
        "u": 367.1, "g": 482.7, "r": 622.3, "i": 754.6, "z": 869.1,
        0: 367.1, 1: 482.7, 2: 622.3, 3: 754.6, 4: 869.1,
    }

    def _resolve_wavelength(band, band_to_wavelength):
        for key in (band, str(band)):
            if key in band_to_wavelength:
                return float(band_to_wavelength[key])
        try:
            return float(band_to_wavelength[int(band)])
        except (KeyError, TypeError, ValueError):
            raise KeyError(f"No wavelength configured for band {band!r}.") from None


TIME_BINS = np.linspace(-500, 500, 50)
BIN_CENTERS = (TIME_BINS[:-1] + TIME_BINS[1:]) / 2

_COLUMN_ALIASES = {
    "object_id": ("object_id", "obj_id", "id"),
    "class": ("obj_type", "class", "target", "object_type", "type"),
    "band": ("band", "passband", "filter"),
    "time": (
        "time_relative_to_peak",
        "time_from_peak",
        "relative_time",
        "time_rel_peak",
        "time",
        "t",
    ),
    "flux": ("flux", "y"),
}


def _resolve_columns(df, columns=None):
    columns = {} if columns is None else dict(columns)
    resolved = {}
    for role, aliases in _COLUMN_ALIASES.items():
        candidates = (columns[role],) if role in columns else aliases
        resolved[role] = next((name for name in candidates if name in df.columns), None)
        if resolved[role] is None:
            raise ValueError(
                f"Could not find the {role!r} column; tried {list(candidates)}. "
                "Pass columns={'role': 'actual_name'} to specify it."
            )
    return resolved


def robust_peak_scale(flux, n_top=5):
    values = np.abs(np.asarray(flux, dtype=float))
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    n_top = min(n_top, len(values))
    top_values = np.partition(values, -n_top)[-n_top:]
    return float(np.median(top_values))


def compute_inter_band_distances(df, columns=None, min_shared_bins=3):
    """Build class-band templates and their pairwise RMSE distances.

    Each object is normalized once, across all of its bands, by the 95th
    percentile of its finite absolute flux values. The returned tuple contains
    the tidy distance table, Spearman summary, and class templates.
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    names = _resolve_columns(df, columns)
    work = df[[names[k] for k in ("object_id", "class", "band", "time", "flux")]].copy()
    work.columns = ["object_id", "class", "band", "time", "flux"]
    work["time"] = pd.to_numeric(work["time"], errors="coerce")
    work["flux"] = pd.to_numeric(work["flux"], errors="coerce")
    work = work.dropna(subset=["object_id", "class", "band", "time", "flux"])

    class_counts = work.groupby("object_id", sort=False)["class"].nunique()
    if (class_counts > 1).any():
        bad = class_counts[class_counts > 1].index.tolist()[:5]
        raise ValueError(f"Each object must have one class label; conflicting objects include {bad}.")

    scales = work.groupby("object_id", sort=False)["flux"].transform(
        robust_peak_scale
    )
    work = work[np.isfinite(scales) & (scales > 0)].copy()
    work["normalized_flux"] = work["flux"] / scales[np.isfinite(scales) & (scales > 0)]
    work["time_bin"] = pd.cut(
        work["time"], TIME_BINS, labels=False, include_lowest=True
    )
    work = work.dropna(subset=["time_bin"])
    work["time_bin"] = work["time_bin"].astype(int)

    object_bins = (
        work.groupby(["object_id", "class", "band", "time_bin"], sort=False, as_index=False)
        ["normalized_flux"].median()
    )
    templates = (
        object_bins.groupby(["class", "band", "time_bin"], sort=False, as_index=False)
        ["normalized_flux"].median()
        .rename(columns={"normalized_flux": "template_flux"})
    )
    object_counts = object_bins.groupby(["class", "band"])["object_id"].nunique()

    rows = []
    for class_label, class_templates in templates.groupby("class", sort=False):
        bands = sorted(class_templates["band"].unique(), key=str)
        pivot = class_templates.pivot(index="time_bin", columns="band", values="template_flux")
        for band_1, band_2 in combinations(bands, 2):
            paired = pivot.reindex(columns=[band_1, band_2]).dropna()
            n_shared = len(paired)
            d_total = (
                float(np.sqrt(np.mean(np.square(paired[band_1] - paired[band_2]))))
                if n_shared >= min_shared_bins else np.nan
            )
            rows.append({
                "class": class_label,
                "band_1": band_1,
                "band_2": band_2,
                "d_total": d_total,
                "n_shared_bins": n_shared,
                "n_objects_band_1": int(object_counts.get((class_label, band_1), 0)),
                "n_objects_band_2": int(object_counts.get((class_label, band_2), 0)),
            })
    distances = pd.DataFrame(rows, columns=[
        "class", "band_1", "band_2", "d_total", "n_shared_bins",
        "n_objects_band_1", "n_objects_band_2",
    ])
    return distances, _spearman_summary(distances), templates


def _pair_wavelength_separation(row, band_to_wavelength):
    try:
        return abs(
            _resolve_wavelength(row.band_1, band_to_wavelength)
            - _resolve_wavelength(row.band_2, band_to_wavelength)
        )
    except KeyError:
        return np.nan


def _bands_in_wavelength_order(bands, band_to_wavelength):
    def sort_key(band):
        try:
            return (0, _resolve_wavelength(band, band_to_wavelength), str(band))
        except KeyError:
            return (1, np.inf, str(band))

    return sorted(bands, key=sort_key)


def _spearman_summary(distances, band_to_wavelength=None):
    band_to_wavelength = band_to_wavelength or DEFAULT_BAND_TO_WAVELENGTH
    rows = []
    for class_label, group in distances.groupby("class", sort=False):
        separation = group.apply(
            _pair_wavelength_separation, axis=1, band_to_wavelength=band_to_wavelength
        ).to_numpy(dtype=float)
        d_total = group["d_total"].to_numpy(dtype=float)
        valid = np.isfinite(separation) & np.isfinite(d_total)
        n_pairs = int(valid.sum())
        if n_pairs >= 2:
            result = spearmanr(separation[valid], d_total[valid])
            rho, pvalue = float(result.statistic), float(result.pvalue)
        else:
            rho, pvalue = np.nan, np.nan
        rows.append({"class": class_label, "spearman_rho": rho,
                     "spearman_pvalue": pvalue, "n_band_pairs": n_pairs})
    return pd.DataFrame(rows, columns=[
        "class", "spearman_rho", "spearman_pvalue", "n_band_pairs"
    ])


def _save_class_plots(distances, output_dir, band_to_wavelength):
    class_groups = list(distances.groupby("class", sort=False))
    if not class_groups:
        return
    ncols = min(3, len(class_groups))
    nrows = int(np.ceil(len(class_groups) / ncols))
    finite_distances = distances.loc[np.isfinite(distances["d_total"]), "d_total"]
    vmax = float(finite_distances.max()) if len(finite_distances) else 1.0

    heatmap_fig, heatmap_axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False,
        layout="constrained",
    )
    images = []
    for ax, (class_label, group) in zip(heatmap_axes.flat, class_groups):
        bands = _bands_in_wavelength_order(
            set(group["band_1"]) | set(group["band_2"]), band_to_wavelength
        )
        matrix_values = np.full((len(bands), len(bands)), np.nan)
        np.fill_diagonal(matrix_values, 0.0)
        matrix = pd.DataFrame(matrix_values, index=bands, columns=bands)
        for row in group.itertuples(index=False):
            matrix.loc[row.band_1, row.band_2] = row.d_total
            matrix.loc[row.band_2, row.band_1] = row.d_total

        image = ax.imshow(matrix.to_numpy(dtype=float), cmap="viridis", vmin=0, vmax=vmax)
        images.append(image)
        ax.set(xticks=range(len(bands)), yticks=range(len(bands)),
               xticklabels=bands, yticklabels=bands,
               title=str(class_label), xlabel="Band", ylabel="Band")
    for ax in heatmap_axes.flat[len(class_groups):]:
        ax.set_visible(False)
    heatmap_fig.suptitle("Inter-band distance by class")
    heatmap_fig.colorbar(
        images[0], ax=list(heatmap_axes.flat[:len(class_groups)]), label="D_total", shrink=0.85
    )
    heatmap_fig.savefig(output_dir / "inter_band_distance_heatmaps.png", dpi=160)
    plt.close(heatmap_fig)

    scatter_fig, scatter_axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False,
        layout="constrained",
    )
    for ax, (class_label, group) in zip(scatter_axes.flat, class_groups):
        separation = group.apply(
            _pair_wavelength_separation, axis=1, band_to_wavelength=band_to_wavelength
        )
        valid = np.isfinite(separation) & np.isfinite(group["d_total"])
        ax.scatter(separation[valid], group.loc[valid, "d_total"])
        ax.set(title=str(class_label),
               xlabel="Absolute central-wavelength difference (nm)", ylabel="D_total")
        ax.grid(alpha=0.25)
    for ax in scatter_axes.flat[len(class_groups):]:
        ax.set_visible(False)
    scatter_fig.suptitle("Wavelength separation versus inter-band distance by class")
    scatter_fig.savefig(output_dir / "inter_band_distance_wavelength_scatter.png", dpi=160)
    plt.close(scatter_fig)


def run_inter_band_distance_analysis(
        df, output_dir=".", columns=None, band_to_wavelength=None, min_shared_bins=3):
    """Compute distances/correlations, save both CSVs and all class plots.

    Example for the project's usual long-form analysis DataFrame::

        distances, correlations = run_inter_band_distance_analysis(
            observations_df, output_dir="inter_band_distance_outputs"
        )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    band_to_wavelength = band_to_wavelength or DEFAULT_BAND_TO_WAVELENGTH
    distances, _, _ = compute_inter_band_distances(
        df, columns=columns, min_shared_bins=min_shared_bins
    )
    correlations = _spearman_summary(distances, band_to_wavelength)
    distances.to_csv(output_dir / "inter_band_distances.csv", index=False)
    correlations.to_csv(output_dir / "inter_band_distance_spearman.csv", index=False)
    _save_class_plots(distances, output_dir, band_to_wavelength)
    return distances, correlations


def compute_wavelength_compatibility(distances, band_to_wavelength=None):
    """Compute the exploratory wavelength-distance correlation per class."""
    if not isinstance(distances, pd.DataFrame):
        distances = pd.read_csv(distances)
    band_to_wavelength = band_to_wavelength or DEFAULT_BAND_TO_WAVELENGTH
    summary = _spearman_summary(distances, band_to_wavelength).rename(columns={
        "spearman_rho": "compatibility_spearman_rho",
        "spearman_pvalue": "compatibility_spearman_pvalue",
    })
    return summary


def _ablation_metrics(result, model_name):
    if model_name not in result.get("artifacts", {}):
        raise ValueError(
            f"Ablation result for object {result.get('object_id')!r} does not contain "
            f"{model_name!r}. Run with models=('mogp_real_wavelength', "
            "'mogp_independent_band_control')."
        )
    artifact = result["artifacts"][model_name]
    if "metrics" in artifact:
        return artifact["metrics"]

    # Support ablation results created before metrics were stored in artifacts.
    from evaluation_metrics import evaluate_mogp_heldout_metrics
    return evaluate_mogp_heldout_metrics(
        artifact["gp"], artifact["heldout_data"],
        train_data=artifact["train_data"], evaluate_raw_metrics=True,
    )


def build_mogp_vs_model_d_pointwise(ablation_results, identity_tolerance=1e-10):
    """Extract matched pointwise predictions from existing ablation results."""
    frames = []
    for result in ablation_results:
        mogp = _ablation_metrics(result, "mogp_real_wavelength")
        model_d = _ablation_metrics(result, "mogp_independent_band_control")
        y_true = np.asarray(mogp["y_true"], dtype=float)
        y_true_d = np.asarray(model_d["y_true"], dtype=float)
        mu_mogp = np.asarray(mogp["y_pred"], dtype=float)
        sigma_mogp = np.asarray(mogp["y_std"], dtype=float)
        mu_model_d = np.asarray(model_d["y_pred"], dtype=float)
        sigma_model_d = np.asarray(model_d["y_std"], dtype=float)

        if not (len(y_true) == len(y_true_d) == len(mu_mogp) == len(mu_model_d)):
            raise AssertionError("MOGP and Model D prediction lengths differ.")
        if not np.allclose(y_true, y_true_d, rtol=0, atol=identity_tolerance):
            raise AssertionError("MOGP and Model D were not evaluated on identical truths.")
        if not (np.all(np.isfinite(sigma_mogp)) and np.all(sigma_mogp > 0)
                and np.all(np.isfinite(sigma_model_d)) and np.all(sigma_model_d > 0)):
            raise ValueError(
                f"Non-finite or non-positive predictive standard deviation for "
                f"object {result.get('object_id')!r}, band {result.get('target_band')!r}."
            )

        from evaluation_metrics import PREDICTIVE_STD_EPSILON, negative_log_predictive_density
        variance_mogp = np.maximum(sigma_mogp ** 2, PREDICTIVE_STD_EPSILON)
        variance_model_d = np.maximum(sigma_model_d ** 2, PREDICTIVE_STD_EPSILON)
        nlpd_mogp = negative_log_predictive_density(y_true, mu_mogp, variance_mogp)
        nlpd_model_d = negative_log_predictive_density(y_true, mu_model_d, variance_model_d)
        delta_nlpd = nlpd_model_d - nlpd_mogp
        delta_sharp = 0.5 * np.log(variance_model_d / variance_mogp)
        delta_std_error = 0.5 * (
            (y_true - mu_model_d) ** 2 / variance_model_d
            - (y_true - mu_mogp) ** 2 / variance_mogp
        )
        if not np.allclose(
                delta_nlpd, delta_sharp + delta_std_error,
                rtol=identity_tolerance, atol=identity_tolerance):
            raise AssertionError("NLPD decomposition identity check failed.")

        heldout = result["artifacts"]["mogp_real_wavelength"]["heldout_data"]
        class_label = result.get("class", heldout.get("obj_type", np.nan))
        row_ids = np.asarray(
            heldout.get("row_id", np.arange(len(y_true))), dtype=object
        )
        frames.append(pd.DataFrame({
            "object_id": result["object_id"],
            "class": class_label,
            "band": np.asarray(mogp["band"], dtype=object),
            "target_band": result.get("target_band"),
            "point_index": np.arange(len(y_true)),
            "row_id": row_ids,
            "y_true": y_true,
            "mu_mogp": mu_mogp,
            "sigma_mogp": sigma_mogp,
            "mu_model_d": mu_model_d,
            "sigma_model_d": sigma_model_d,
            "nlpd_mogp": nlpd_mogp,
            "nlpd_model_d": nlpd_model_d,
            "delta_nlpd": delta_nlpd,
            "delta_sharp": delta_sharp,
            "delta_std_error": delta_std_error,
            "delta_squared_error": (
                (y_true - mu_model_d) ** 2 - (y_true - mu_mogp) ** 2
            ),
        }))
    if not frames:
        raise ValueError("No ablation results were provided.")
    return pd.concat(frames, ignore_index=True)


def aggregate_mogp_vs_model_d(pointwise):
    """Aggregate matched pointwise comparisons by object, then by class."""
    object_rows = []
    for (object_id, class_label), group in pointwise.groupby(
            ["object_id", "class"], sort=False, dropna=False):
        error_mogp = group["y_true"] - group["mu_mogp"]
        error_model_d = group["y_true"] - group["mu_model_d"]
        rmse_mogp = float(np.sqrt(np.mean(error_mogp ** 2)))
        rmse_model_d = float(np.sqrt(np.mean(error_model_d ** 2)))
        object_rows.append({
            "object_id": object_id, "class": class_label,
            "n_test_points": len(group),
            "nlpd_mogp_object": group["nlpd_mogp"].mean(),
            "nlpd_model_d_object": group["nlpd_model_d"].mean(),
            "delta_nlpd_object": group["delta_nlpd"].mean(),
            "delta_sharp_object": group["delta_sharp"].mean(),
            "delta_std_error_object": group["delta_std_error"].mean(),
            "delta_squared_error_object": group["delta_squared_error"].mean(),
            "rmse_mogp_object": rmse_mogp,
            "rmse_model_d_object": rmse_model_d,
            "delta_rmse_object": rmse_model_d - rmse_mogp,
        })
    objects = pd.DataFrame(object_rows)

    class_rows = []
    for class_label, group in objects.groupby("class", sort=False, dropna=False):
        delta = group["delta_nlpd_object"]
        class_rows.append({
            "class": class_label,
            "median_delta_nlpd": delta.median(),
            "mean_delta_nlpd": delta.mean(),
            "q25_delta_nlpd": delta.quantile(0.25),
            "q75_delta_nlpd": delta.quantile(0.75),
            "median_delta_sharp": group["delta_sharp_object"].median(),
            "median_delta_std_error": group["delta_std_error_object"].median(),
            "median_delta_squared_error": group["delta_squared_error_object"].median(),
            "median_delta_rmse": group["delta_rmse_object"].median(),
            "fraction_improved_nlpd": (delta > 0).mean(),
            "fraction_improved_rmse": (group["delta_rmse_object"] > 0).mean(),
            "n_objects": group["object_id"].nunique(),
            "total_test_points": int(group["n_test_points"].sum()),
        })
    return objects, pd.DataFrame(class_rows)


def compute_band_pair_transfer_analysis(
        ablation_results, distances, transfer_pairs, support_window=0.25):
    """Compute class D, effective support, and NLPD gain per aux->target pair.

    Effective support is first averaged over held-out target times within each
    object, then summarized by the class median so dense objects do not dominate.
    Each result must have exactly one active auxiliary band for attribution.
    """
    if not isinstance(distances, pd.DataFrame):
        distances = pd.read_csv(distances)
    transfer_pairs = [tuple(pair) for pair in transfer_pairs]
    if any(len(pair) != 2 for pair in transfer_pairs):
        raise ValueError("Each transfer pair must be (auxiliary_band, target_band).")

    object_rows = []
    observed_configurations = set()
    for result in ablation_results:
        target_band = result.get("target_band")
        active_aux = tuple(
            str(band) for band, ratio in result.get("aux_band_ratios", {}).items()
            if float(ratio) > 0
        )
        observed_configurations.add((str(target_band), active_aux))
        matching_pairs = [
            (aux, target) for aux, target in transfer_pairs
            if str(target) == str(target_band)
        ]
        if not matching_pairs:
            continue
        active_aux = [
            band for band, ratio in result.get("aux_band_ratios", {}).items()
            if float(ratio) > 0
        ]
        if len(active_aux) != 1:
            raise ValueError(
                f"Object {result.get('object_id')!r}, target {target_band!r} has "
                f"{len(active_aux)} active auxiliary bands ({active_aux}); pair-specific "
                "gain requires exactly one."
            )
        aux_band = active_aux[0]
        if not any(str(aux) == str(aux_band) for aux, _ in matching_pairs):
            continue

        pointwise = build_mogp_vs_model_d_pointwise([result])
        artifact = result["artifacts"]["mogp_real_wavelength"]
        train_data = artifact["train_data"]
        heldout_data = artifact["heldout_data"]
        train_band = np.asarray(train_data["band"], dtype=object)
        aux_times = np.asarray(train_data["t"], dtype=float)[
            np.asarray([str(value) == str(aux_band) for value in train_band])
        ]
        target_times = np.asarray(heldout_data["t"], dtype=float)
        support = (
            float(np.mean([
                np.sum(np.abs(aux_times - time) <= support_window)
                for time in target_times
            ]))
            if len(target_times) else np.nan
        )
        error_mogp = pointwise["y_true"] - pointwise["mu_mogp"]
        error_model_d = pointwise["y_true"] - pointwise["mu_model_d"]
        rmse_mogp = float(np.sqrt(np.mean(error_mogp ** 2)))
        rmse_model_d = float(np.sqrt(np.mean(error_model_d ** 2)))
        object_rows.append({
            "object_id": result["object_id"],
            "class": result.get("class", heldout_data.get("obj_type", np.nan)),
            "auxiliary_band": aux_band,
            "target_band": target_band,
            "effective_aux_support_object": support,
            "delta_nlpd_object": pointwise["delta_nlpd"].mean(),
            "delta_sharp_object": pointwise["delta_sharp"].mean(),
            "delta_std_error_object": pointwise["delta_std_error"].mean(),
            "delta_squared_error_object": pointwise["delta_squared_error"].mean(),
            "rmse_mogp_object": rmse_mogp,
            "rmse_model_d_object": rmse_model_d,
            "delta_rmse_object": rmse_model_d - rmse_mogp,
            "n_test_points": len(pointwise),
        })
    if not object_rows:
        requested = [f"{aux}->{target}" for aux, target in transfer_pairs]
        observed = [
            f"{','.join(aux) if aux else '(none)'}->{target}"
            for target, aux in sorted(observed_configurations)
        ]
        raise ValueError(
            f"No ablation results matched requested transfer pairs {requested}. "
            f"Observed active-auxiliary->target configurations: {observed}. "
            "Rerun the ablation with the requested target_band and exactly one "
            "positive aux_band_ratios entry."
        )
    object_pairs = pd.DataFrame(object_rows)

    rows = []
    for (class_label, aux_band, target_band), group in object_pairs.groupby(
            ["class", "auxiliary_band", "target_band"], sort=False, dropna=False):
        pair_mask = (
            ((distances["band_1"].astype(str) == str(aux_band))
             & (distances["band_2"].astype(str) == str(target_band)))
            | ((distances["band_1"].astype(str) == str(target_band))
               & (distances["band_2"].astype(str) == str(aux_band)))
        ) & (distances["class"].astype(str) == str(class_label))
        pair_distance = distances.loc[pair_mask, "d_total"]
        if len(pair_distance) != 1:
            raise ValueError(
                f"Expected one D_total row for class {class_label!r} and pair "
                f"{aux_band!r}->{target_band!r}; found {len(pair_distance)}."
            )
        delta = group["delta_nlpd_object"]
        rows.append({
            "class": class_label,
            "auxiliary_band": aux_band,
            "target_band": target_band,
            "d_total_pair": float(pair_distance.iloc[0]),
            "effective_aux_support": group["effective_aux_support_object"].median(),
            "median_delta_nlpd": delta.median(),
            "mean_delta_nlpd": delta.mean(),
            "q25_delta_nlpd": delta.quantile(0.25),
            "q75_delta_nlpd": delta.quantile(0.75),
            "median_delta_sharp": group["delta_sharp_object"].median(),
            "median_delta_std_error": group["delta_std_error_object"].median(),
            "median_delta_squared_error": group["delta_squared_error_object"].median(),
            "median_delta_rmse": group["delta_rmse_object"].median(),
            "fraction_improved_nlpd": (delta > 0).mean(),
            "fraction_improved_rmse": (group["delta_rmse_object"] > 0).mean(),
            "n_objects": group["object_id"].nunique(),
            "total_test_points": int(group["n_test_points"].sum()),
            "support_window": float(support_window),
        })
    return object_pairs, pd.DataFrame(rows)


def _plot_band_pair_transfer(pair_summary, output_dir):
    pairs = list(pair_summary[["auxiliary_band", "target_band"]]
                 .drop_duplicates().itertuples(index=False, name=None))
    fig, axes = plt.subplots(
        len(pairs), 2, figsize=(11, 4.5 * len(pairs)), squeeze=False,
        layout="constrained",
    )
    for row_index, (aux_band, target_band) in enumerate(pairs):
        group = pair_summary[
            (pair_summary["auxiliary_band"].astype(str) == str(aux_band))
            & (pair_summary["target_band"].astype(str) == str(target_band))
        ]
        for ax, x_column, xlabel in [
            (axes[row_index, 0], "d_total_pair", f"D_total ({aux_band}, {target_band})"),
            (axes[row_index, 1], "effective_aux_support",
             f"Effective support ({aux_band} → {target_band})"),
        ]:
            ax.scatter(group[x_column], group["median_delta_nlpd"])
            for item in group.itertuples(index=False):
                ax.annotate(str(item[0]), (getattr(item, x_column), item.median_delta_nlpd),
                            xytext=(4, 4), textcoords="offset points", fontsize=8)
            ax.axhline(0, color="grey", linewidth=1)
            ax.set(xlabel=xlabel, ylabel="Median NLPD gain: Model D − MOGP",
                   title=f"{aux_band} → {target_band}")
            ax.grid(alpha=0.2)
    fig.savefig(output_dir / "band_pair_distance_support_vs_nlpd_gain.png", dpi=160)
    plt.close(fig)

    gain_fig, gain_axes = plt.subplots(
        len(pairs), 1, figsize=(max(8, 0.7 * pair_summary["class"].nunique()),
                               4.5 * len(pairs)),
        squeeze=False, layout="constrained",
    )
    decomposition_fig, decomposition_axes = plt.subplots(
        len(pairs), 1, figsize=(max(8, 0.8 * pair_summary["class"].nunique()),
                               4.5 * len(pairs)),
        squeeze=False, layout="constrained",
    )
    for row_index, (aux_band, target_band) in enumerate(pairs):
        group = pair_summary[
            (pair_summary["auxiliary_band"].astype(str) == str(aux_band))
            & (pair_summary["target_band"].astype(str) == str(target_band))
        ].sort_values("median_delta_nlpd")
        x = np.arange(len(group))
        lower = group["median_delta_nlpd"] - group["q25_delta_nlpd"]
        upper = group["q75_delta_nlpd"] - group["median_delta_nlpd"]

        ax = gain_axes[row_index, 0]
        ax.errorbar(x, group["median_delta_nlpd"], yerr=[lower, upper],
                    fmt="o", capsize=3)
        ax.axhline(0, color="grey", linewidth=1)
        ax.set(xticks=x, xticklabels=group["class"],
               ylabel="NLPD gain: Model D − MOGP",
               title=f"Class-level NLPD gain: {aux_band} → {target_band}")
        ax.tick_params(axis="x", rotation=45)

        ax = decomposition_axes[row_index, 0]
        width = 0.38
        ax.bar(x - width / 2, group["median_delta_sharp"], width, label="Sharpness")
        ax.bar(x + width / 2, group["median_delta_std_error"], width,
               label="Standardized error")
        ax.plot(x, group["median_delta_nlpd"], "ko", label="Total NLPD gain")
        ax.axhline(0, color="grey", linewidth=1)
        ax.set(xticks=x, xticklabels=group["class"], ylabel="Median contribution",
               title=f"NLPD decomposition: {aux_band} → {target_band}")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
    gain_fig.savefig(output_dir / "band_pair_class_nlpd_gain.png", dpi=160)
    decomposition_fig.savefig(
        output_dir / "band_pair_class_nlpd_decomposition.png", dpi=160
    )
    plt.close(gain_fig)
    plt.close(decomposition_fig)


def run_band_pair_transfer_analysis(
        ablation_results, transfer_pairs,
        distances="inter_band_distance_outputs/inter_band_distances.csv",
        output_dir="inter_band_distance_outputs", support_window=0.25):
    """Save the complete pair-specific MOGP-gain analysis and figures.

    This workflow intentionally uses D_total for each requested band pair; it
    does not use the across-pair wavelength-compatibility Spearman statistic.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    object_pairs, pair_summary = compute_band_pair_transfer_analysis(
        ablation_results, distances, transfer_pairs, support_window=support_window
    )
    object_pairs.to_csv(output_dir / "object_level_band_pair_transfer.csv", index=False)
    pair_summary.to_csv(output_dir / "class_level_band_pair_transfer.csv", index=False)
    _plot_band_pair_transfer(pair_summary, output_dir)
    for row in pair_summary.itertuples(index=False):
        print(
            f"{row[0]} {row.auxiliary_band}->{row.target_band}: "
            f"D={row.d_total_pair:.10g}, support={row.effective_aux_support:.10g}, "
            f"median_delta_nlpd={row.median_delta_nlpd:.10g}"
        )
    return object_pairs, pair_summary


def _plot_compatibility_and_gain(summary, output_dir):
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    ax.scatter(summary["compatibility_spearman_rho"], summary["median_delta_nlpd"])
    for row in summary.itertuples(index=False):
        ax.annotate(str(row[0]), (row.compatibility_spearman_rho, row.median_delta_nlpd),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="grey", linewidth=1)
    ax.axvline(0, color="grey", linewidth=1)
    ax.set(xlabel="Wavelength-distance compatibility, Spearman ρ",
           ylabel="Median NLPD gain: Model D − MOGP")
    fig.savefig(output_dir / "compatibility_vs_nlpd_gain.png", dpi=160)
    plt.close(fig)

    ordered = summary.sort_values("median_delta_nlpd")
    x = np.arange(len(ordered))
    lower = ordered["median_delta_nlpd"] - ordered["q25_delta_nlpd"]
    upper = ordered["q75_delta_nlpd"] - ordered["median_delta_nlpd"]
    fig, ax = plt.subplots(figsize=(max(7, 0.65 * len(ordered)), 5), layout="constrained")
    ax.errorbar(x, ordered["median_delta_nlpd"], yerr=[lower, upper], fmt="o", capsize=3)
    ax.axhline(0, color="grey", linewidth=1)
    ax.set(xticks=x, xticklabels=ordered["class"], ylabel="NLPD gain: Model D − MOGP",
           title="Class-level NLPD gain")
    ax.tick_params(axis="x", rotation=45)
    fig.savefig(output_dir / "class_nlpd_gain.png", dpi=160)
    plt.close(fig)

    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7, 0.75 * len(ordered)), 5), layout="constrained")
    ax.bar(x - width / 2, ordered["median_delta_sharp"], width, label="Sharpness")
    ax.bar(x + width / 2, ordered["median_delta_std_error"], width,
           label="Standardized error")
    ax.plot(x, ordered["median_delta_nlpd"], "ko", label="Total NLPD gain")
    ax.axhline(0, color="grey", linewidth=1)
    ax.set(xticks=x, xticklabels=ordered["class"], ylabel="Median contribution",
           title="NLPD gain decomposition by class")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.savefig(output_dir / "class_nlpd_decomposition.png", dpi=160)
    plt.close(fig)


def run_compatibility_vs_mogp_gain_analysis(
        ablation_results, distances="inter_band_distance_outputs/inter_band_distances.csv",
        output_dir="inter_band_distance_outputs", band_to_wavelength=None):
    """Save matched MOGP/Model-D point, object, class tables and three figures.

    In the existing ablation loop, retain each full result (not only its rows)::

        all_ablation_results.append(result)
        pointwise, objects, summary = run_compatibility_vs_mogp_gain_analysis(
            all_ablation_results
        )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pointwise = build_mogp_vs_model_d_pointwise(ablation_results)
    objects, class_gain = aggregate_mogp_vs_model_d(pointwise)
    compatibility = compute_wavelength_compatibility(distances, band_to_wavelength)
    summary = compatibility.merge(class_gain, on="class", how="inner", validate="one_to_one")

    pointwise.to_csv(output_dir / "pointwise_mogp_vs_model_d.csv", index=False)
    objects.to_csv(output_dir / "object_level_mogp_vs_model_d.csv", index=False)
    summary.to_csv(output_dir / "class_level_compatibility_and_gain.csv", index=False)
    _plot_compatibility_and_gain(summary, output_dir)
    for row in summary.itertuples(index=False):
        print(
            f"{row[0]}: compatibility_rho={row.compatibility_spearman_rho:.10g}, "
            f"median_delta_nlpd={row.median_delta_nlpd:.10g}, "
            f"median_delta_sharp={row.median_delta_sharp:.10g}, "
            f"median_delta_std_error={row.median_delta_std_error:.10g}"
        )
    return pointwise, objects, summary


def print_class_gain_diagnostics(
        object_results, class_summary=None, class_label="M-dwarf", precision=12):
    """Print object distributions, missing counts, and exact class-level values."""
    object_columns = [
        "delta_nlpd_object", "delta_sharp_object",
        "delta_std_error_object", "delta_rmse_object",
    ]
    missing_columns = [
        "delta_nlpd_object", "delta_sharp_object", "delta_std_error_object",
    ]
    missing = [name for name in ["class", *object_columns] if name not in object_results]
    if missing:
        raise ValueError(f"object_results is missing required columns: {missing}")
    selected = object_results[object_results["class"] == class_label]
    if selected.empty:
        raise ValueError(f"No object-level rows found for class {class_label!r}.")

    with pd.option_context("display.precision", precision):
        print(selected[object_columns].describe())
        print("\nMissing values:")
        print(selected[missing_columns].isna().sum())
        if class_summary is not None:
            class_values = class_summary[class_summary["class"] == class_label]
            print("\nExact class-level values:")
            print(class_values.to_string(index=False, float_format=lambda value: f"{value:.{precision}g}"))
    return selected
