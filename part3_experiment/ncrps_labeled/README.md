# Complete auxiliary-observation diagnostics

Panel metrics: nCRPS and nRMSE. Subfigure labels: True. Existing selected configurations and rankings are retained; title deltas are converted to nCRPS units when requested. nCRPS = CRPS / S_all, with S_all = max(Q95(abs(all complete valid raw flux across all bands)), 1e-8), using the saved object_flux_scale. This same denominator is fixed across fractions and models and matches the existing nRMSE denominator. Saved ncrps values are verified, then displayed directly. `part3_displayed_metrics.csv` and `part3_displayed_selection.csv` contain the displayed scores and deltas; each labeled figure has a `.panels.csv` index.

Updated 31 selected Part 3 configurations using saved dense posteriors and evaluation predictions; **zero GP refits**. Original Part 1/2 and Part 3 result artifacts remain unchanged (229 protected files).

Modified `part3_analysis.plot_case` now invokes the shared `part3_auxiliary_plots.render_complete_case` composer; `part3_experiment/update_auxiliary_figures.py` regenerates the existing cases from cached predictions without fitting. The original three-model comparison remains at fractions 0.2/0.8. Each row is followed by a dedicated observation-only target-band panel with its own raw-flux limits, then separate raw-flux auxiliary panels; auxiliary training markers refer only to the independent/real models, not the single-band GP. Only active experiment bands are shown. Full observations, measurement errors, selected/unselected auxiliary points, and training/evaluation/unused target points are retained. Target and auxiliary panels share x limits encompassing every valid observation; auxiliary y limits include all observations and error bars, with no percentile clipping. All six target panels share limits including the latent ±2σ bands and test predictive ±1σ. Peak markers identify largest **raw flux** (not largest absolute flux).

The former plot origin was the saved target-training peak, which can differ between fractions. The corrected display origin is the minimum valid MJD of the COMPLETE target-band observations, shared across fractions, all models, all bands and prediction grids. Every displayed time is original MJD minus this single object-level reference; zero denotes the first complete target observation, not a peak. The original fitting reference is retained separately as `fit_reference_time` and is never changed. Before plotting, exact comparisons verify complete target/auxiliary raw times, fluxes and errors, identical original prediction grids, fixed evaluation indices, identical auxiliary training selections and nested target training sets. All panels use the same x limits across fractions. `part3_time_coordinate_audit.csv` records old fitting references, new display references, common limits and these checks. Cached posteriors are already in raw flux units through the original tested inverse transform (`normalized mean * target flux_scale + background`, `normalized std * target flux_scale`). No second rescaling is applied. Explicit saved-index preprocessing checks verify the reference, target time/flux scales and full target time range; the numerical prediction cache itself is not changed.

Flux range means P95−P5 over all valid raw observations in that band. Aux/target ratios are NaN for zero target range. Proximity uses a configurable ±5-day window and retains signed lags. Real GP global maximum AND minimum on the existing 600-point target grid are inspected; neither is automatically labeled incorrect. Observed auxiliary local maxima are computed on the complete chronological observations (endpoints/global maximum included), then matched to the actual selected training indices. These discrete local maxima are descriptive and can include noise. Reported temporal coincidence does not establish causality. A target region with zero full-curve observations in this window is described as observationally unconstrained, distinct from having no target TRAINING observations.

## Requested examples

### Object 6460481, target fraction 0.2

- Real GP maximum: t=60144.80259, flux=205.23; 1 selected auxiliary local maxima within ±5 days. Full auxiliary maximum lag=-0.517 days (selected=True). Target observations nearby: all=1, train=0, evaluation=0; observed within window. Nearest target observation is 2.537 days away, flux=13.6774.
- Real GP minimum: t=60565.87826, flux=-10.2658; 0 selected auxiliary local maxima within ±5 days. Full auxiliary maximum lag=-421.593 days (selected=True). Target observations nearby: all=1, train=1, evaluation=0; observed within window. Nearest target observation is 0.290 days away, flux=-10.314.

### Object 6460481, target fraction 0.8

- Real GP maximum: t=60151.40252, flux=51.8903; 0 selected auxiliary local maxima within ±5 days. Full auxiliary maximum lag=-7.117 days (selected=True). Target observations nearby: all=0, train=0, evaluation=0; observationally unconstrained within window. Nearest target observation is 9.137 days away, flux=13.6774.
- Real GP minimum: t=60172.52230, flux=-20.9389; 0 selected auxiliary local maxima within ±5 days. Full auxiliary maximum lag=-28.237 days (selected=True). Target observations nearby: all=0, train=0, evaluation=0; observationally unconstrained within window. Nearest target observation is 6.693 days away, flux=-2.34817.

### Object 34161124, target fraction 0.2

- Real GP maximum: t=60181.46169, flux=637.738; 1 selected auxiliary local maxima within ±5 days. Full auxiliary maximum lag=+0.659 days (selected=True). Target observations nearby: all=1, train=0, evaluation=1; observed within window. Nearest target observation is 1.270 days away, flux=-1.63119.
- Real GP minimum: t=60262.52514, flux=-34.2391; 0 selected auxiliary local maxima within ±5 days. Full auxiliary maximum lag=-80.404 days (selected=True). Target observations nearby: all=0, train=0, evaluation=0; observationally unconstrained within window. Nearest target observation is 58.408 days away, flux=-2.94861.

### Object 34161124, target fraction 0.8

- Real GP maximum: t=60182.74842, flux=627.349; 1 selected auxiliary local maxima within ±5 days. Full auxiliary maximum lag=-0.627 days (selected=True). Target observations nearby: all=1, train=0, evaluation=1; observed within window. Nearest target observation is 2.557 days away, flux=-1.63119.
- Real GP minimum: t=60211.05629, flux=-46.2662; 0 selected auxiliary local maxima within ±5 days. Full auxiliary maximum lag=-28.935 days (selected=True). Target observations nearby: all=0, train=0, evaluation=0; observationally unconstrained within window. Nearest target observation is 6.939 days away, flux=-2.94861.

## Peak coverage for the requested single-auxiliary cases

| Object | Fraction | Target train / eval / unused | Target maximum partition | Aux train / all | Aux maximum selected | Aux/target P95−P5 |
| --- | --- | --- | --- | --- | --- | --- |
| 6460481 | 0.2 | 4/5/12 | training | 4/8 | True | 6.55 |
| 6460481 | 0.8 | 16/5/0 | training | 4/8 | True | 6.55 |
| 34161124 | 0.2 | 3/4/11 | unused | 14/28 | True | 1.45 |
| 34161124 | 0.8 | 14/4/0 | training | 14/28 | True | 1.45 |

## Summary across selected configurations

- 124 real-GP extrema inspected (max/min, two fractions).
- 81 extrema have no complete target observations within the window; 109 have no target training observations within it.
- 35 extrema coincide with at least one selected observed auxiliary local maximum. The per-band CSV retains exact lags and fluxes; this count describes timing only.

See `part3_peak_coverage.csv`, `part3_prediction_peak_proximity.csv`, `part3_plotted_observations.csv` and per-configuration CSVs in `peak_diagnostics/`. `part3_updated_figure_index.csv` lists all figures. Existing residual plots remain available in the original Part 3 directory.

## Figure index

- Object 6460481, part2, aux=g, seed=2: [updated figure](./part3_figures/object_6460481_part2_aux_g_seed_2_total_nrmse_fraction_0.2.png)
- Object 34161124, part1, aux=all, seed=0: [updated figure](./part3_figures/object_34161124_part1_aux_all_seed_0_cross_ncrps_fraction_0.2.png)
- Object 34161124, part2, aux=z, seed=0: [updated figure](./part3_figures/object_34161124_part2_aux_z_seed_0_cross_ncrps_fraction_0.2.png)
- Object 34866353, part1, aux=all, seed=1: [updated figure](./part3_figures/object_34866353_part1_aux_all_seed_1_cross_ncrps_fraction_0.2.png)
- Object 126309844, part1, aux=all, seed=2: [updated figure](./part3_figures/object_126309844_part1_aux_all_seed_2_cross_ncrps_fraction_0.2.png)
- Object 4413277, part1, aux=all, seed=1: [updated figure](./part3_figures/object_4413277_part1_aux_all_seed_1_cross_ncrps_fraction_0.2.png)
- Object 9858227, part1, aux=all, seed=1: [updated figure](./part3_figures/object_9858227_part1_aux_all_seed_1_cross_ncrps_fraction_0.2.png)
- Object 3100669, part1, aux=all, seed=0: [updated figure](./part3_figures/object_3100669_part1_aux_all_seed_0_cross_nrmse_fraction_0.2.png)
- Object 85439795, part1, aux=all, seed=2: [updated figure](./part3_figures/object_85439795_part1_aux_all_seed_2_cross_nrmse_fraction_0.2.png)
- Object 130739978, part1, aux=all, seed=1: [updated figure](./part3_figures/object_130739978_part1_aux_all_seed_1_cross_nrmse_fraction_0.2.png)
- Object 34866353, part1, aux=all, seed=0: [updated figure](./part3_figures/object_34866353_part1_aux_all_seed_0_total_ncrps_fraction_0.2.png)
- Object 44307163, part1, aux=all, seed=1: [updated figure](./part3_figures/object_44307163_part1_aux_all_seed_1_total_ncrps_fraction_0.2.png)
- Object 44307163, part1, aux=all, seed=0: [updated figure](./part3_figures/object_44307163_part1_aux_all_seed_0_total_ncrps_fraction_0.2.png)
- Object 34866353, part1, aux=all, seed=2: [updated figure](./part3_figures/object_34866353_part1_aux_all_seed_2_cross_ncrps_fraction_0.8.png)
- Object 44307163, part1, aux=all, seed=2: [updated figure](./part3_figures/object_44307163_part1_aux_all_seed_2_cross_ncrps_fraction_0.8.png)
- Object 36842485, part1, aux=all, seed=2: [updated figure](./part3_figures/object_36842485_part1_aux_all_seed_2_cross_nrmse_fraction_0.8.png)
- Object 34866353, part2, aux=r, seed=1: [updated figure](./part3_figures/object_34866353_part2_aux_r_seed_1_cross_ncrps_fraction_0.2.png)
- Object 44307163, part2, aux=z, seed=0: [updated figure](./part3_figures/object_44307163_part2_aux_z_seed_0_cross_ncrps_fraction_0.2.png)
- Object 9858227, part2, aux=z, seed=1: [updated figure](./part3_figures/object_9858227_part2_aux_z_seed_1_cross_ncrps_fraction_0.2.png)
- Object 67410269, part2, aux=g, seed=0: [updated figure](./part3_figures/object_67410269_part2_aux_g_seed_0_cross_ncrps_fraction_0.2.png)
- Object 3100669, part2, aux=r, seed=0: [updated figure](./part3_figures/object_3100669_part2_aux_r_seed_0_cross_nrmse_fraction_0.2.png)
- Object 59907038, part2, aux=g, seed=2: [updated figure](./part3_figures/object_59907038_part2_aux_g_seed_2_cross_nrmse_fraction_0.2.png)
- Object 127779559, part2, aux=r, seed=0: [updated figure](./part3_figures/object_127779559_part2_aux_r_seed_0_cross_nrmse_fraction_0.2.png)
- Object 34866353, part2, aux=r, seed=0: [updated figure](./part3_figures/object_34866353_part2_aux_r_seed_0_total_ncrps_fraction_0.2.png)
- Object 44307163, part2, aux=z, seed=1: [updated figure](./part3_figures/object_44307163_part2_aux_z_seed_1_total_ncrps_fraction_0.2.png)
- Object 44307163, part2, aux=r, seed=1: [updated figure](./part3_figures/object_44307163_part2_aux_r_seed_1_total_ncrps_fraction_0.2.png)
- Object 100122518, part2, aux=r, seed=0: [updated figure](./part3_figures/object_100122518_part2_aux_r_seed_0_total_ncrps_fraction_0.2.png)
- Object 44307163, part2, aux=r, seed=2: [updated figure](./part3_figures/object_44307163_part2_aux_r_seed_2_cross_ncrps_fraction_0.8.png)
- Object 34866353, part2, aux=r, seed=2: [updated figure](./part3_figures/object_34866353_part2_aux_r_seed_2_cross_ncrps_fraction_0.8.png)
- Object 130739978, part2, aux=g, seed=1: [updated figure](./part3_figures/object_130739978_part2_aux_g_seed_1_cross_nrmse_fraction_0.8.png)
- Object 115557873, part2, aux=r, seed=2: [updated figure](./part3_figures/object_115557873_part2_aux_r_seed_2_cross_nrmse_fraction_0.8.png)

## Per-configuration timing and target coverage

The following descriptive table uses real-GP max/min on the saved grid. Aux peak bands list only selected complete-curve local maxima inside the time window. A dash means no such peak in the window; it is not evidence against a mechanism. Target counts use all valid target observations, including unused/evaluation data.

| Object | Experiment / active aux | Seed | Fraction | GP extremum | Nearby trained aux peak bands | Target all / train nearby |
| --- | --- | --- | --- | --- | --- | --- |
| 6460481 | part2 / g | 2 | 0.2 | maximum | g | 1 / 0 |
| 6460481 | part2 / g | 2 | 0.2 | minimum | — | 1 / 1 |
| 6460481 | part2 / g | 2 | 0.8 | maximum | — | 0 / 0 |
| 6460481 | part2 / g | 2 | 0.8 | minimum | — | 0 / 0 |
| 34161124 | part1 / all | 0 | 0.2 | maximum | z | 1 / 0 |
| 34161124 | part1 / all | 0 | 0.2 | minimum | — | 0 / 0 |
| 34161124 | part1 / all | 0 | 0.8 | maximum | z | 1 / 0 |
| 34161124 | part1 / all | 0 | 0.8 | minimum | — | 0 / 0 |
| 34161124 | part2 / z | 0 | 0.2 | maximum | z | 1 / 0 |
| 34161124 | part2 / z | 0 | 0.2 | minimum | — | 0 / 0 |
| 34161124 | part2 / z | 0 | 0.8 | maximum | z | 1 / 0 |
| 34161124 | part2 / z | 0 | 0.8 | minimum | — | 0 / 0 |
| 34866353 | part1 / all | 1 | 0.2 | maximum | r | 2 / 0 |
| 34866353 | part1 / all | 1 | 0.2 | minimum | — | 0 / 0 |
| 34866353 | part1 / all | 1 | 0.8 | maximum | — | 0 / 0 |
| 34866353 | part1 / all | 1 | 0.8 | minimum | — | 0 / 0 |
| 126309844 | part1 / all | 2 | 0.2 | maximum | r | 0 / 0 |
| 126309844 | part1 / all | 2 | 0.2 | minimum | — | 0 / 0 |
| 126309844 | part1 / all | 2 | 0.8 | maximum | — | 0 / 0 |
| 126309844 | part1 / all | 2 | 0.8 | minimum | — | 0 / 0 |
| 4413277 | part1 / all | 1 | 0.2 | maximum | — | 0 / 0 |
| 4413277 | part1 / all | 1 | 0.2 | minimum | — | 0 / 0 |
| 4413277 | part1 / all | 1 | 0.8 | maximum | — | 0 / 0 |
| 4413277 | part1 / all | 1 | 0.8 | minimum | — | 0 / 0 |
| 9858227 | part1 / all | 1 | 0.2 | maximum | z | 0 / 0 |
| 9858227 | part1 / all | 1 | 0.2 | minimum | — | 0 / 0 |
| 9858227 | part1 / all | 1 | 0.8 | maximum | z | 0 / 0 |
| 9858227 | part1 / all | 1 | 0.8 | minimum | — | 0 / 0 |
| 3100669 | part1 / all | 0 | 0.2 | maximum | r | 0 / 0 |
| 3100669 | part1 / all | 0 | 0.2 | minimum | — | 0 / 0 |
| 3100669 | part1 / all | 0 | 0.8 | maximum | r | 0 / 0 |
| 3100669 | part1 / all | 0 | 0.8 | minimum | — | 0 / 0 |
| 85439795 | part1 / all | 2 | 0.2 | maximum | u | 1 / 0 |
| 85439795 | part1 / all | 2 | 0.2 | minimum | r | 1 / 0 |
| 85439795 | part1 / all | 2 | 0.8 | maximum | — | 1 / 1 |
| 85439795 | part1 / all | 2 | 0.8 | minimum | r | 0 / 0 |
| 130739978 | part1 / all | 1 | 0.2 | maximum | g | 1 / 0 |
| 130739978 | part1 / all | 1 | 0.2 | minimum | — | 2 / 0 |
| 130739978 | part1 / all | 1 | 0.8 | maximum | g | 1 / 0 |
| 130739978 | part1 / all | 1 | 0.8 | minimum | — | 0 / 0 |
| 34866353 | part1 / all | 0 | 0.2 | maximum | — | 0 / 0 |
| 34866353 | part1 / all | 0 | 0.2 | minimum | — | 0 / 0 |
| 34866353 | part1 / all | 0 | 0.8 | maximum | — | 0 / 0 |
| 34866353 | part1 / all | 0 | 0.8 | minimum | — | 0 / 0 |
| 44307163 | part1 / all | 1 | 0.2 | maximum | z | 0 / 0 |
| 44307163 | part1 / all | 1 | 0.2 | minimum | — | 0 / 0 |
| 44307163 | part1 / all | 1 | 0.8 | maximum | z | 0 / 0 |
| 44307163 | part1 / all | 1 | 0.8 | minimum | — | 0 / 0 |
| 44307163 | part1 / all | 0 | 0.2 | maximum | — | 1 / 0 |
| 44307163 | part1 / all | 0 | 0.2 | minimum | — | 0 / 0 |
| 44307163 | part1 / all | 0 | 0.8 | maximum | — | 1 / 0 |
| 44307163 | part1 / all | 0 | 0.8 | minimum | — | 0 / 0 |
| 34866353 | part1 / all | 2 | 0.2 | maximum | r | 2 / 1 |
| 34866353 | part1 / all | 2 | 0.2 | minimum | r | 1 / 0 |
| 34866353 | part1 / all | 2 | 0.8 | maximum | — | 0 / 0 |
| 34866353 | part1 / all | 2 | 0.8 | minimum | — | 0 / 0 |
| 44307163 | part1 / all | 2 | 0.2 | maximum | — | 1 / 0 |
| 44307163 | part1 / all | 2 | 0.2 | minimum | — | 0 / 0 |
| 44307163 | part1 / all | 2 | 0.8 | maximum | — | 0 / 0 |
| 44307163 | part1 / all | 2 | 0.8 | minimum | — | 0 / 0 |
| 36842485 | part1 / all | 2 | 0.2 | maximum | r | 0 / 0 |
| 36842485 | part1 / all | 2 | 0.2 | minimum | — | 0 / 0 |
| 36842485 | part1 / all | 2 | 0.8 | maximum | r | 0 / 0 |
| 36842485 | part1 / all | 2 | 0.8 | minimum | — | 1 / 1 |
| 34866353 | part2 / r | 1 | 0.2 | maximum | r | 2 / 0 |
| 34866353 | part2 / r | 1 | 0.2 | minimum | — | 0 / 0 |
| 34866353 | part2 / r | 1 | 0.8 | maximum | — | 0 / 0 |
| 34866353 | part2 / r | 1 | 0.8 | minimum | — | 0 / 0 |
| 44307163 | part2 / z | 0 | 0.2 | maximum | — | 0 / 0 |
| 44307163 | part2 / z | 0 | 0.2 | minimum | — | 1 / 0 |
| 44307163 | part2 / z | 0 | 0.8 | maximum | — | 1 / 1 |
| 44307163 | part2 / z | 0 | 0.8 | minimum | — | 1 / 0 |
| 9858227 | part2 / z | 1 | 0.2 | maximum | z | 0 / 0 |
| 9858227 | part2 / z | 1 | 0.2 | minimum | — | 0 / 0 |
| 9858227 | part2 / z | 1 | 0.8 | maximum | z | 0 / 0 |
| 9858227 | part2 / z | 1 | 0.8 | minimum | — | 0 / 0 |
| 67410269 | part2 / g | 0 | 0.2 | maximum | — | 0 / 0 |
| 67410269 | part2 / g | 0 | 0.2 | minimum | — | 1 / 1 |
| 67410269 | part2 / g | 0 | 0.8 | maximum | — | 1 / 1 |
| 67410269 | part2 / g | 0 | 0.8 | minimum | — | 1 / 1 |
| 3100669 | part2 / r | 0 | 0.2 | maximum | r | 0 / 0 |
| 3100669 | part2 / r | 0 | 0.2 | minimum | — | 1 / 0 |
| 3100669 | part2 / r | 0 | 0.8 | maximum | r | 0 / 0 |
| 3100669 | part2 / r | 0 | 0.8 | minimum | — | 1 / 0 |
| 59907038 | part2 / g | 2 | 0.2 | maximum | — | 0 / 0 |
| 59907038 | part2 / g | 2 | 0.2 | minimum | — | 1 / 0 |
| 59907038 | part2 / g | 2 | 0.8 | maximum | — | 0 / 0 |
| 59907038 | part2 / g | 2 | 0.8 | minimum | — | 1 / 1 |
| 127779559 | part2 / r | 0 | 0.2 | maximum | r | 0 / 0 |
| 127779559 | part2 / r | 0 | 0.2 | minimum | — | 0 / 0 |
| 127779559 | part2 / r | 0 | 0.8 | maximum | — | 0 / 0 |
| 127779559 | part2 / r | 0 | 0.8 | minimum | — | 2 / 1 |
| 34866353 | part2 / r | 0 | 0.2 | maximum | — | 0 / 0 |
| 34866353 | part2 / r | 0 | 0.2 | minimum | — | 0 / 0 |
| 34866353 | part2 / r | 0 | 0.8 | maximum | — | 0 / 0 |
| 34866353 | part2 / r | 0 | 0.8 | minimum | — | 0 / 0 |
| 44307163 | part2 / z | 1 | 0.2 | maximum | — | 1 / 1 |
| 44307163 | part2 / z | 1 | 0.2 | minimum | — | 1 / 0 |
| 44307163 | part2 / z | 1 | 0.8 | maximum | z | 0 / 0 |
| 44307163 | part2 / z | 1 | 0.8 | minimum | — | 1 / 1 |
| 44307163 | part2 / r | 1 | 0.2 | maximum | r | 0 / 0 |
| 44307163 | part2 / r | 1 | 0.2 | minimum | — | 0 / 0 |
| 44307163 | part2 / r | 1 | 0.8 | maximum | — | 0 / 0 |
| 44307163 | part2 / r | 1 | 0.8 | minimum | — | 0 / 0 |
| 100122518 | part2 / r | 0 | 0.2 | maximum | — | 1 / 0 |
| 100122518 | part2 / r | 0 | 0.2 | minimum | — | 0 / 0 |
| 100122518 | part2 / r | 0 | 0.8 | maximum | — | 0 / 0 |
| 100122518 | part2 / r | 0 | 0.8 | minimum | — | 0 / 0 |
| 44307163 | part2 / r | 2 | 0.2 | maximum | — | 0 / 0 |
| 44307163 | part2 / r | 2 | 0.2 | minimum | — | 0 / 0 |
| 44307163 | part2 / r | 2 | 0.8 | maximum | — | 0 / 0 |
| 44307163 | part2 / r | 2 | 0.8 | minimum | — | 1 / 0 |
| 34866353 | part2 / r | 2 | 0.2 | maximum | r | 2 / 1 |
| 34866353 | part2 / r | 2 | 0.2 | minimum | r | 1 / 0 |
| 34866353 | part2 / r | 2 | 0.8 | maximum | r | 0 / 0 |
| 34866353 | part2 / r | 2 | 0.8 | minimum | — | 0 / 0 |
| 130739978 | part2 / g | 1 | 0.2 | maximum | g | 1 / 0 |
| 130739978 | part2 / g | 1 | 0.2 | minimum | — | 1 / 0 |
| 130739978 | part2 / g | 1 | 0.8 | maximum | g | 1 / 0 |
| 130739978 | part2 / g | 1 | 0.8 | minimum | — | 2 / 1 |
| 115557873 | part2 / r | 2 | 0.2 | maximum | — | 0 / 0 |
| 115557873 | part2 / r | 2 | 0.2 | minimum | — | 0 / 0 |
| 115557873 | part2 / r | 2 | 0.8 | maximum | — | 1 / 1 |
| 115557873 | part2 / r | 2 | 0.8 | minimum | — | 0 / 0 |
