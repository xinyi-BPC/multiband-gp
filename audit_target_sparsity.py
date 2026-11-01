"""Audit saved sparsity CSVs without refitting or changing scientific outputs.

Usage: .venv/bin/python audit_target_sparsity.py target_sparsity_outputs
"""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation_metrics import paired_target_sparsity_results


def _indices(value):
    if not isinstance(value, str):
        return np.asarray(value, dtype=int)
    if "..." in value:
        raise ValueError("Truncated index arrays cannot be audited.")
    return np.fromstring(value.strip().strip("[]").replace(",", " "), sep=" ", dtype=int)


def audit_saved_sparsity(output_dir, audit_dir=None):
    output_dir = Path(output_dir)
    audit_dir = Path(audit_dir) if audit_dir else output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(output_dir / "target_sparsity_rows.csv")
    base = ["object_id", "target_band", "aux_train_fraction", "random_state"]
    count_summary = df.groupby(["target_train_fraction", "model"]).agg(
        cases=("object_id", "size"),
        n_target_train_min=("n_train_target_band", "min"),
        n_target_train_median=("n_train_target_band", "median"),
        n_target_train_max=("n_train_target_band", "max"),
        n_aux_train_min=("n_train_other_bands", "min"),
        n_aux_train_max=("n_train_other_bands", "max"),
        n_evaluation_min=("n_heldout_target_band", "min"),
        n_evaluation_max=("n_heldout_target_band", "max"),
        nrmse_median=("nrmse", "median"), crps_median=("crps", "median"),
    ).reset_index()
    cases = []
    for key, group in df.groupby(base):
        row = dict(zip(base, key))
        for column in ("heldout_indices", "target_pool_indices", "y_true",
                       "aux_selected_indices_by_band", "object_flux_scale"):
            row[column + "_fixed"] = bool(group[column].nunique(dropna=False) == 1)
        row["nested_target_subsets"] = True
        row["target_within_pool"] = True
        row["no_evaluation_leakage"] = True
        row["counts_match_saved_indices"] = True
        row["full_pool_at_0_8"] = True
        row["models_use_same_target_and_auxiliary_indices"] = True
        for _, model_group in group.groupby("model"):
            previous = set()
            for _, entry in model_group.sort_values("target_train_fraction").iterrows():
                target = set(_indices(entry.target_train_indices))
                pool = set(_indices(entry.target_pool_indices))
                evaluation = set(_indices(entry.heldout_indices))
                auxiliary = ast.literal_eval(entry.aux_selected_indices_by_band)
                n_valid = len(pool) + len(evaluation)
                expected = len(pool) if entry.target_train_fraction == 0.8 else min(
                    len(pool), int(np.floor(entry.target_train_fraction * n_valid)))
                row["nested_target_subsets"] &= previous <= target
                row["target_within_pool"] &= target <= pool
                row["no_evaluation_leakage"] &= not bool(target & evaluation)
                row["counts_match_saved_indices"] &= (
                    len(target) == entry.n_train_target_band == expected
                    and len(evaluation) == entry.n_heldout_target_band
                    and (0 if entry.model == "single_band_gp" else sum(
                        len(indices) for indices in auxiliary.values())) == entry.n_train_other_bands)
                if entry.target_train_fraction == 0.8:
                    row["full_pool_at_0_8"] &= target == pool
                previous = target
        for _, fraction_group in group.groupby("target_train_fraction"):
            row["models_use_same_target_and_auxiliary_indices"] &= all(
                fraction_group[column].nunique(dropna=False) == 1 for column in
                ("target_train_indices", "aux_selected_indices_by_band", "heldout_indices"))
        cases.append(row)
    case_table = pd.DataFrame(cases)
    checks = {column: bool(case_table[column].all())
              for column in case_table.columns if column not in base}
    checks["normalization_fixed_across_seeds"] = bool((df.groupby(
        ["object_id", "target_band", "aux_train_fraction"]
    ).object_flux_scale.nunique(dropna=False) == 1).all())
    pairs = paired_target_sparsity_results(df)
    summary = {
        "n_model_rows": len(df), "n_objects": int(df.object_id.nunique()),
        "n_seeds": int(df.random_state.nunique()), "n_paired_cases": len(pairs),
        "checks": checks,
        "nrmse_formula_max_abs_error": float(np.max(abs(df.nrmse - df.rmse / df.object_flux_scale))),
        "object_flux_scales": df.object_flux_scale.unique().tolist(),
        "fitted_hyperparameters_available": "fitted_time_length_scale" in df,
    }
    saved = pd.read_csv(output_dir / "target_sparsity_summary.csv")
    object_path = output_dir / "target_sparsity_object_rows.csv"
    aggregate_input = pd.read_csv(object_path) if object_path.exists() else df
    keys = ["model", "target_band", "target_train_fraction", "aux_train_fraction"]
    for metric in ("nrmse", "crps"):
        recomputed = aggregate_input.groupby(keys)[metric].median()
        summary[metric + "_saved_median_max_abs_error"] = float(np.max(abs(
            recomputed - saved.set_index(keys)[metric + "_median"])))
    saved_pairs = pd.read_csv(output_dir / "target_sparsity_paired_rows.csv")
    keys = base + ["target_train_fraction"]
    for metric in ("delta_nrmse", "delta_crps"):
        summary[metric + "_saved_max_abs_error"] = float(np.max(abs(
            pairs.set_index(keys)[metric] - saved_pairs.set_index(keys)[metric])))
    if summary["fitted_hyperparameters_available"]:
        parameters = [c for c in df.columns if c.startswith("fitted_") and c != "fitted_kernel"]
        df[base + ["model", "target_train_fraction"] + parameters + [
            "preprocessing_t_scale", "preprocessing_flux_scale"]].to_csv(
                audit_dir / "hyperparameters.csv", index=False)
    count_summary.to_csv(audit_dir / "training_counts_and_metrics.csv", index=False)
    case_table.to_csv(audit_dir / "case_checks.csv", index=False)
    # Make the distinction between paired medians and differences of medians explicit.
    means = df.groupby(["target_train_fraction", "model"]).crps.median().unstack()
    differences = pd.DataFrame({
        "difference_of_crps_medians": means["mogp_independent_band_control"] - means["mogp_real_wavelength"],
        "median_of_paired_crps_differences": pairs.groupby("target_train_fraction").delta_crps.median(),
    })
    differences.to_csv(audit_dir / "crps_aggregation_comparison.csv")
    (audit_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    parser.add_argument("--audit-dir")
    args = parser.parse_args()
    print(json.dumps(audit_saved_sparsity(args.output_dir, args.audit_dir), indent=2))
