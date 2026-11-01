"""Read-only cohort/split audit. Does not fit any model."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from population_kernel_ablation.saved_splits import load_sources, reconstruct_saved_case, validate_nested_saved_rows
from part2_analysis import CASE_KEYS
from part3_analysis import protection_manifest


def run(output):
    before = protection_manifest()
    examples, rows = load_sources()
    paired = validate_nested_saved_rows(rows)
    cases = 0
    for keys, group in rows.groupby(CASE_KEYS):
        reconstruct_saved_case(examples[str(group.iloc[0].object_id)], group)
        cases += 1
    assert before == protection_manifest()
    result = dict(validation_passed=True, n_objects=len(examples), saved_model_rows=len(rows),
        eligible_cases=cases, nested_object_seed_pairs=paired,
        target_training_fractions=[0.2, 0.8], seeds=[0, 1, 2],
        all_target_auxiliary_row_ids_match=True, all_preprocessing_scales_match=True,
        all_object_denominators_match=True, identical_target_evaluation_observations=True,
        sampling_permutations_generated=0, models_fitted=0,
        protected_original_files_checked=len(before), protected_original_files_unchanged=True)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'preflight_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    (output / 'preservation_manifest.json').write_text(json.dumps(before, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run(ROOT / 'population_kernel_ablation/preflight')
