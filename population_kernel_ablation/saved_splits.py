"""Load Part 1's saved observations directly, without any sampling RNG calls."""
from pathlib import Path
import json

import numpy as np
from data_processing import (_extract_valid_band_observations,
                             process_one_obj_one_band_train_heldout)
from MOGP_model import (DEFAULT_BAND_TO_WAVELENGTH, _raw_rows_from_split,
                        _concat_raw_row_groups, _build_mogp_data_from_raw_rows)
from part2_analysis import load_part1_rows, CASE_KEYS
from singleGP_model import object_level_empirical_flux_scale

ROOT = Path(__file__).resolve().parents[1]


def load_sources():
    examples = json.loads((ROOT / 'three_model_experiment/objects_300.json').read_text())
    rows = load_part1_rows(ROOT / 'three_model_experiment/full_300/target_sparsity_rows.csv',
                           fractions=(0.2, 0.8))
    assert len(examples) == 300
    return {str(e['object_id']): e for e in examples}, rows


def reconstruct_saved_case(example, saved_rows):
    """Build production GP dictionaries using exclusively serialized indices.

    This does not call _make_target_sparsity_plan or any sampling routine.
    Historical eligibility is inherited from saved cases, rather than reapplied.
    The denominator uses complete observations for evaluation only.
    """
    if set(saved_rows.model) != {'single_band_gp', 'mogp_real_wavelength',
                                 'mogp_independent_band_control'} or len(saved_rows) != 3:
        raise ValueError('Expected exactly one saved row per historical model')
    single = saved_rows[saved_rows.model == 'single_band_gp'].iloc[0]
    multi = saved_rows[saved_rows.model == 'mogp_real_wavelength'].iloc[0]
    for key in CASE_KEYS:
        assert saved_rows[key].nunique() == 1, key
    for _, row in saved_rows.iterrows():
        for key in ('target_train_indices', 'target_pool_indices', 'heldout_indices'):
            np.testing.assert_array_equal(row[key], single[key], err_msg=key)
        assert row.aux_selected_indices_by_band == single.aux_selected_indices_by_band
    assert str(example['object_id']) == str(single.object_id)
    assert single.target_band == 'i'
    selected = np.asarray(single.target_train_indices, dtype=int)
    heldout = np.asarray(single.heldout_indices, dtype=int)
    # force_peak_in_train has no effect on explicit indices; disabling its
    # unused complete-curve peak scan avoids consulting held-out flux at all.
    train_s, test_s = process_one_obj_one_band_train_heldout(
        example, target_band='i', subtract_background=False,
        force_peak_in_train=False, split_indices=(selected, heldout))
    if train_s is None or test_s is None:
        raise ValueError('Saved eligible case failed production preprocessing')
    np.testing.assert_array_equal(train_s['train_indices'], selected)
    np.testing.assert_array_equal(test_s['heldout_indices'], heldout)
    groups = []
    for band, indices in [('i', selected), *multi.aux_selected_indices_by_band.items()]:
        t, y, err = _extract_valid_band_observations(example, band)
        indices = np.asarray(indices, dtype=int)
        assert len(indices) == len(set(indices))
        if band != 'i':
            assert band in ('u', 'g', 'r', 'z')
            assert len(indices) == int(np.floor(0.5 * len(t)))
        groups.append(_raw_rows_from_split(dict(band=band, t_raw=t, y_raw=y, yerr_raw=err), indices))
    raw_train = _concat_raw_row_groups(groups)
    t, y, err = _extract_valid_band_observations(example, 'i')
    raw_test = _raw_rows_from_split(dict(band='i', t_raw=t, y_raw=y, yerr_raw=err), heldout)
    train_m = _build_mogp_data_from_raw_rows(example, raw_train, train_s, DEFAULT_BAND_TO_WAVELENGTH)
    test_m = _build_mogp_data_from_raw_rows(example, raw_test, train_s, DEFAULT_BAND_TO_WAVELENGTH)
    np.testing.assert_array_equal(train_m['row_id'], multi.training_row_ids)
    independent = saved_rows[saved_rows.model == 'mogp_independent_band_control'].iloc[0]
    np.testing.assert_array_equal(train_m['row_id'], independent.training_row_ids)
    np.testing.assert_array_equal([f'i:{i}' for i in selected], single.training_row_ids)
    assert not set(train_m['row_id']) & set(test_m['row_id'])
    for _, row in saved_rows.iterrows():
        np.testing.assert_allclose(train_s['flux_scale'], row.preprocessing_flux_scale, rtol=1e-13)
        np.testing.assert_allclose(train_s['t_scale'], row.preprocessing_t_scale, rtol=1e-13)
    scale = object_level_empirical_flux_scale(example=example)
    np.testing.assert_allclose(saved_rows.object_flux_scale, scale, rtol=1e-13)
    np.testing.assert_allclose(test_s['X'][:, 0], test_m['X'][:, 0], rtol=0, atol=0)
    np.testing.assert_array_equal(test_s['y_raw'], test_m['y_raw'])
    return dict(single_train=train_s, single_test=test_s, multi_train=train_m,
                multi_test=test_m, object_flux_scale=scale)


def validate_nested_saved_rows(rows):
    """No inference of missing splits: every fraction must exist in the CSV."""
    first = rows[rows.model == 'mogp_real_wavelength']
    count = 0
    for (_, seed), group in first.groupby(['object_id', 'random_state']):
        assert set(group.target_train_fraction) == {0.2, 0.8}
        low = group[group.target_train_fraction == 0.2].iloc[0]
        high = group[group.target_train_fraction == 0.8].iloc[0]
        assert set(low.target_train_indices).issubset(high.target_train_indices)
        np.testing.assert_array_equal(high.target_train_indices, high.target_pool_indices)
        np.testing.assert_array_equal(low.target_pool_indices, high.target_pool_indices)
        np.testing.assert_array_equal(low.heldout_indices, high.heldout_indices)
        assert low.aux_selected_indices_by_band == high.aux_selected_indices_by_band
        count += 1
    return count
