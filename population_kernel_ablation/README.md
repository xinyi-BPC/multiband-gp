# Population kernel ablation

This code runs a separate five-model experiment using the original 300-object cohort and **saved Part 1 indices directly**. It never regenerates sampling plans or writes historical outputs. The validated `band_amplitude_ablation/kernel.py` is imported unchanged.

The output directory is `band_amplitude_population_ablation/`. Original zero-restart A/B/C references are in `historical_scores.csv`; all primary results use new five-restart fits. Models are A single-band, B independent-band, C shared amplitude with historical variance bounds, D shared amplitude with matched variance bounds, and E band amplitudes.

Run from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/band-amplitude-mpl OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python population_kernel_ablation/preflight.py
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/band-amplitude-mpl OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python population_kernel_ablation/experiment.py --stage pilot
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/band-amplitude-mpl OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python population_kernel_ablation/failure_control.py
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/band-amplitude-mpl OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python population_kernel_ablation/experiment.py --stage full --workers 4
```

The pilot selection is frozen from historical results before fitting. Full fitting is gated on the pilot validation. The full run reuses completed pilot fits through per-case JSON checkpoints, and supports safe resumption with the same configuration. A changed model configuration is rejected; use a separate output directory for a different experiment. `preflight.py` refuses to overwrite its audit directory.

Fitting is parallelized with ordinary Python worker processes; each uses one BLAS thread. Five-model scores, optimizer warnings, bound hits, initialization projections, runtimes, exact input IDs, and any errors are saved per case. No fitted models or predictions are shared between different objects.

`analysis.py` averages all three seeds within each object before population medians. The primary subset requires all five successful models and all three seeds. Case-level bootstrap summaries resample entire object seed-clusters, and object-level summaries resample the seed-averaged objects. The numerical tie rule, bootstrap seed/count, catastrophic score thresholds, and initialization handling are frozen in `configuration.json`.

The robust initialization estimator is unchanged. To honor the prescribed finite optimization bounds across this population, an estimate outside the interval is projected to the nearest interior point for initialization only; `initializations.csv` records its raw estimate, actual initial value, method, and projection flag. No bounds are widened or chosen from evaluation outcomes.

Tests:

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/band-amplitude-mpl OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m unittest discover -s tests -p 'test_band_amplitude_ablation.py'
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/band-amplitude-mpl OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m unittest discover -s tests -p 'test_population_kernel_*.py'
```

These test the kernel, actual saved-split reconstruction without sampling calls, held-out-flux independence of training transforms, matched variance bounds, seed averaging, object-cluster sample counts, and numerical ties. All plotting uncertainty is latent; all score uncertainty includes observation errors.

## Current four-model figure view

All images now display only A single-band, B independent-band, D shared matched
bounds, and E band amplitudes. The original-bound C configuration is excluded
from every image. The transition figure uses the existing matched D/A → E/A
counts, and the parameter plots exclude C. Original numerical results and
checkpoints retain their full audit history.

The representative target-flux zoom includes separate u/g/r/z observation panels
with the exact cached auxiliary training rows highlighted. Titles omit the seed
and the verbose zoom description; a footnote still states the target zoom's
clipping behavior. All observed fluxes remain in raw units and use one shared
MJD reference.

Regenerate and replace the existing PNG/PDF files without any optimization:

```sh
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/band-amplitude-mpl \
OPENBLAS_NUM_THREADS=1 .venv/bin/python population_kernel_ablation/replot.py
```

The exact i+z control lacked a saved dense matched-bound curve. Its posterior is
reconstructed by Cholesky conditioning at the saved optimized parameters and
saved training rows. The four models' evaluation means and predictive standard
deviations must match their original per-point diagnostics before export. This
performs no GP fitting, optimization, or sampling. See
`band_amplitude_population_ablation/four_model_plot_validation.json` and
`FOUR_MODEL_FIGURES.md` for the validation and figure gallery.
