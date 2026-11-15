# Thesis Project: Asset Pricing with Machine Learning

A replication of Gu, Kelly, and Xiu (2020), *“Empirical Asset Pricing via Machine Learning.”*

This repository implements an expanding-window forecasting framework for U.S. equity returns using linear, tree-based, and neural-network models. It follows the paper’s methodology while providing a modular codebase for experimentation, validation, and reproducible research.

## Overview

The pipeline constructs a monthly stock-characteristic panel, engineers features, trains models in an expanding-window setting, generates out-of-sample return forecasts, and evaluates predictive performance and portfolio strategies.

Key components include:

- Expanding-window training with annual model refitting
- Gu et al. (2020)-style linear models based on `HuberRegressor`
- PCR, PLS, GBRT, random forests, MLPs, and LSTM models
- 94 stock characteristics, industry indicators, and macroeconomic interactions
- Cross-sectional feature scaling and missing-value handling
- Parquet-based caching for datasets and feature panels
- Module-level logging and reproducibility controls

## Repository Structure

```text
thesis-project/
├── config.py                        # Configuration dataclasses and experiment settings
├── data_inputs.py                   # Data loading and input validation
├── dataset_builder.py               # Dataset construction, merging, and imputation
├── features.py                      # Feature engineering and scaling
├── transformers.py                  # Custom grouped transformers and preprocessing utilities
├── models_huber.py                  # Huber-regression implementations
├── models_linear.py                 # Linear models, including PCR and PLS
├── models_trees.py                  # GBRT and random-forest models
├── models_mlp.py                    # Multi-layer perceptron models
├── models_alternative.py            # Additional or experimental model implementations
├── models_lstm.txt                  # LSTM implementation notes/reference
├── evaluation.py                    # Forecast evaluation and portfolio metrics
├── expanding_window.py              # Expanding-window forecasting workflow
├── run_experiments.py               # Main experiment entry point
├── io_utils.py                      # Input/output utilities and logging
├── visualization.py                 # Plotting and result visualization
├── replication.ipynb                # Replication workflow notebook
├── dataset_builder_book.ipynb       # Dataset-construction exploration notebook
├── test_*.py                        # Unit and integration tests
├── descriptives/                    # Descriptive statistics and related outputs
├── output/                          # Experiment results and generated artifacts
└── README.md                        # This file
```

## Installation

### Requirements

- Python 3.9 or later
- `pip` or `conda`

### Install dependencies

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn torch pyarrow
```

`pyarrow` is required for Parquet-based caching. `fastparquet` can be used as an alternative:

```bash
pip install fastparquet
```

## Usage

### Run the full experiment

```bash
python run_experiments.py
```

By default, this runs the expanding-window experiment over the configured forecast period and produces predictions, evaluation metrics, and model artifacts.

### Configure the experiment

Edit `config.py` to control:

- `DataRegimeConfig`: full-data or coding-mode data regime
- `ExpandingWindowConfig`: training, validation, and forecast periods
- `ModelSelectionConfig`: models included in the experiment
- `CacheConfig`: caching behavior
- `HyperGridConfig`: hyperparameter search grids

### Select models

```python
# Run all configured models
run_all_models = True
models_to_run = None

# Run only the OLS-3 benchmark
run_all_models = False
models_to_run = ["OLS_3"]

# Run a custom subset
run_all_models = False
models_to_run = ["OLS_3", "GBRT", "MLP"]
```

### Data regimes

| Mode | Description |
|---|---|
| `full` | Uses the complete characteristic set, industry indicators, and macroeconomic interactions. |
| `coding` | Uses a reduced sample and feature set for faster development and debugging. |

```python
regime_cfg = DataRegimeConfig(mode="coding")
```

## Data Requirements

Place the following input files in the `data/` directory:

1. **`datashare.csv`** — 94 stock characteristics
   - `permno`: CRSP permanent security identifier
   - `date`: monthly observation date
   - Columns 3–96: the 94 characteristics
   - `sic2`: two-digit SIC industry code

2. **`crsp_monthly.csv`** — CRSP monthly returns
   - Required columns: `permno`, `date`, `ret`, and `dlret`

3. **`Data2024_monthly_goyal.csv`** — Goyal-Welch macroeconomic predictors

The pipeline supports mixed monthly date formats, including `YYYYMM` and standard date strings.

## Expanding-Window Methodology

For each forecast year, the training window expands by one year, while the validation window rolls forward. The model is refit annually and used to produce forecasts for the following 12 months.

```text
Training:   1957 ─────────────────────────────────→ expands annually
Validation:       1975 ────────────────────────→ rolls forward, 12 years
Forecast:              1987 1988 1989 ... 2021
```

For a forecast year \(Y\):

1. Train on data from 1957 through \(1974 + (Y - 1987)\).
2. Validate on the preceding 12 years ending in \(1986 + (Y - 1987)\).
3. Generate monthly forecasts for all stocks in year \(Y\).
4. Evaluate forecasts and refit the model for the next year.

The implementation supports the original 1987–2016 forecast period as well as the extended 1987–2021 period.

## Models

### Linear models

Following Gu et al. (2020), the linear benchmark models use robust Huber regression rather than ordinary least squares.

- `OLS_3`: Huber regression with size, value, and momentum factors
- `OLS_full`: Huber regression with the full feature set
- `Huber`: Huber regression with tuned regularization and robustness parameters
- `PCR`: Principal-component regression
- `PLS`: Partial least-squares regression

### Tree-based models

- `GBRT`: Gradient-boosted regression trees
- `RandomForest`: Random forest

### Neural networks

- `MLP`: Multi-layer perceptron
- `LSTM`: Long short-term memory network; available when all models are selected

## Feature Engineering

The preprocessing workflow includes:

1. Monthly winsorization at the 1st and 99th percentiles
2. Cross-sectional quantile scaling of the 94 characteristics to the interval \([-1, 1]\)
3. Industry indicators for the leading SIC2 industries
4. Interactions between characteristics and macroeconomic variables

In full mode, the feature set includes 94 characteristics, macroeconomic variables, industry indicators, and characteristic-by-macro interactions. Macro variables are not included as standalone features, consistent with the original paper’s setup.

## Outputs

Experiment results are written to `output/unified/`:

```text
output/unified/
├── all_predictions.parquet
├── metrics_by_year.csv
├── metrics_by_year_original.csv
├── predictions_year_<year>.parquet
└── models_year_<year>.pkl
```

For each model and forecast year, the evaluation includes:

- Stock-level predictive \(R^2\)
- Portfolio-level predictive \(R^2\)
- Market-timing Sharpe ratio

## Caching

Cached datasets, feature panels, and feature-name mappings are stored under `cache/`. Caching reduces repeated preprocessing and helps ensure consistent results across runs.

## Logging

Logs are saved as:

```text
logs/eap_ml_<mode>_<timestamp>.log
```

Logs include timestamps, log levels, module names, dataset dimensions, model settings, and major pipeline milestones.

## Reproducibility

- A global random seed is set at the start of the experiment.
- Deterministic PyTorch settings are enabled by default.
- Caching supports consistent preprocessing across runs.
- The full experiment runs continuously across the configured forecast period.

## Citation

Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. *The Review of Financial Studies*, 33(5), 2223–2273.

## License

This project is intended for academic research purposes.
