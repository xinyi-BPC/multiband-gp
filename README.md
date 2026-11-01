# Multi-output Gaussian Processes for Light Curves

This repository contains single-band and multi-output Gaussian-process models,
preprocessing utilities, evaluation metrics, notebooks, and experiment outputs
for irregular multi-band astronomical light curves.

## Repository layout

- `singleGP_model.py`, `MOGP_model.py`: single-band and multi-output GP models.
- `data_processing.py`: light-curve filtering, splitting, alignment, and normalization.
- `evaluation_metrics.py`: held-out nRMSE, nCRPS, calibration, coverage, and PIT metrics.
- `feature_extraction.py`, `visualization.py`: feature and plotting utilities.
- `normalized_evaluation/`: normalized Part 1 and Part 2 results.
- `part3_experiment/ncrps_labeled/`: final Part 3 nCRPS/nRMSE tables and figures.
- `band_amplitude_population_ablation/`, `population_kernel_ablation/`: ablation studies.
- `figures/`: figures previously generated in the repository root.
- `reports/`: project PDF reports.

## Getting started

Create a virtual environment and install the scientific Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start with `getting_started.ipynb` for data inspection. 
