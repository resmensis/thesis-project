# Empirical Asset Pricing via Machine Learning: A Replication

A modular Python replication of Gu, Kelly, and Xiu (2020), *"Empirical Asset Pricing via Machine Learning"* (Review of Financial Studies, 33(5), 2223–2273). The pipeline builds a monthly U.S. stock-characteristic panel, trains return-prediction models in an expanding-window setup with annual refitting, and evaluates out-of-sample performance.

## Features

- Monthly CRSP / characteristic / macro panel with 94 firm characteristics and 8 Goyal–Welch macro predictors
- Expanding-window (or rolling-window) forecasting with annual refitting, 1987–2021
- Models: OLS-3, OLS (full), PLS, PCR, Huber-based variants, Linear SVR, Random Forest, GBRT, and feed-forward networks NN1–NN5
- Validation-based hyperparameter tuning using the Gu et al. predictive R² as the scoring function
- Parquet and pickle caching of datasets, feature panels, and predictions
- Configuration through dataclasses in `config.py`
- Module-level logging and global seeding for reproducibility
- Evaluation tables and figures, including cumulative-return plots with NBER recession shading

## Repository Structure

```text
thesis-project/
├── config.py              # All configuration dataclasses (data, windows, grids, models, evaluation)
├── run_experiments.py     # Main entry point: dataset → features → predictions → evaluation
├── data_inputs.py         # Loading and validating raw input files
├── dataset_builder.py     # Merging characteristics, CRSP returns, and macro data
├── features.py            # Feature panel construction and scaling
├── transformers.py        # Custom preprocessing transformers
├── prediction_window.py   # Expanding/rolling-window training and forecasting loop
├── models_linear.py       # OLS, PLS, PCR, and related linear models
├── models_huber.py        # Huber-loss models
├── models_trees.py        # Random Forest and GBRT
├── models_mlp.py          # NN1–NN5 (scikit-learn MLPRegressor, grid-searched)
├── evaluation.py          # Predictive R², portfolio sorts, and result tables
├── visualization.py       # Figures
├── test_visualization.py  # Tests for the visualization module
├── io_utils.py            # Seeding, logging, and parquet/pickle helpers
└── README.md
```

Generated at runtime (not tracked): `data/`, `cache/`, `output/`, `descriptives/`, `logs/`.

## Installation

Requirements: Python 3.9+ (3.10+ recommended).

```bash
git clone [https://github.com/resmensis/thesis-project.git](https://github.com/resmensis/thesis-project.git)
cd thesis-project
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install pandas numpy scipy scikit-learn matplotlib seaborn pyarrow
```

`pyarrow` (or `fastparquet`) is required for parquet caching.

## Data

Place the following files in `data/` (paths are set in `DataFilesConfig`):

| File | Content |
|---|---|
| `datashare.csv` | The 94 firm characteristics (Green, Hand, and Zhang), with `permno`, `date`, and `sic2` |
| `crsp_monthly_new.csv` | CRSP monthly data: `permno`, `date`/`yyyymm`, `ret`, `dlret`, and optionally `prc`, `shrout` |
| `Data2024_monthly_goyal.csv` | Goyal–Welch macro predictors |

Macro predictors used: `d/p`, `e/p`, `b/m`, `ntis`, `tbl`, `tms`, `dfy`, `svar`. A reduced set (`d/p`, `ntis`, `tms`, `svar`) is available via `use_all_macros=False`.

The input data is licensed (CRSP via WRDS) and is not included in this repository.

## Usage

```bash
python run_experiments.py
```

The script runs these stages and skips any whose cached output exists:

1. Build the complete dataset (`cache/complete_dataset.parquet`)
2. Build the feature panel (`cache/feature_panel_<mode>_<suffix>.parquet`)
3. Run the window-based forecasts
4. Merge Fama–French characteristic portfolios and evaluate

Settings are defined at the top of `main()` in `run_experiments.py`. Edit them there.

### Key configuration

| Config class | Purpose |
|---|---|
| `DataRegimeConfig` | `mode="full"` or `"coding"` (reduced sample of 100 stocks and 10 characteristics for debugging), industry dummies, macro interactions, OLS-3-only mode |
| `WindowConfig` | `window_type` (`"expanding"` or `"rolling"`), train/validation/test years, refit frequency |
| `ModelSelectionConfig` | `run_all_models` and `models_to_run` |
| `HyperGridConfig` | Base and extended hyperparameter grids; `use_extended_grids=True` enables the larger ones |
| `CacheConfig` | Enable caching and force rebuilds of the dataset, features, or model fits |
| `RunControlConfig` | `dataset_creation_only`, `feature_panel_only`, `load_predictions` |
| `ReproducibilityConfig` | Seed (default 42), deterministic torch, quantile-scaler settings |
| `EvaluationConfig` | Model order and labels, exclusions, cumulative-return models, NBER recession dates |

### Selecting models

```python
# All models
ModelSelectionConfig(run_all_models=True, models_to_run=[])

# A subset
ModelSelectionConfig(run_all_models=False, models_to_run=["OLS_3", "GBRT", "MLP"])
```

Available names: `OLS_full`, `OLS_3`, `PLS`, `PCR_Linear`, `PCR_Huber`, `RandomForest`, `GBRT`, `MLP`, `LinearSVR`, `OLS_3_old`.

### Using cached predictions

Set `RunControlConfig(load_predictions=True)`. If `all_predictions.parquet` and all yearly prediction files exist, training is skipped and only evaluation is rerun.

## Forecasting Design

| Period | Years |
|---|---|
| Initial training | 1957–1974 |
| Validation (12 years) | 1975–1986 |
| Out-of-sample test | 1987–2016 (original) and 1987–2021 (extended) |

Models are refit every year. In expanding mode the training window grows by one year per refit. In both modes the validation window moves forward by one year. Hyperparameters are chosen on the validation set by the Gu et al. predictive R², then the model forecasts the next 12 months.

Neural networks use the architectures in `HyperGridConfig.mlp_architectures`:

| Name | Hidden layers |
|---|---|
| NN1 | 32 |
| NN2 | 32, 16 |
| NN3 | 32, 16, 8 |
| NN4 | 32, 16, 8, 4 |
| NN5 | 32, 16, 8, 4, 2 |

## Output

Results go to `output/<mode>_mode-<window>_window/<feature_suffix>_<models>_<grid>/` (for coding mode, `output/coding_mode-<window>_window/`). This folder contains:

```text
all_predictions.parquet
all_predictions_fama_french.parquet
predictions/            # one file per forecast year
visualization/          # figures
feature_importance/     # feature-importance outputs
```

Evaluation tables and figures are written to the same directory. Logs go to `logs/` (logger name `eap_ml`), at INFO level in full mode and DEBUG in coding mode.

## Reproducibility

- A global seed (`random_state=42`) is set once at the start of each run.
- Deterministic torch behaviour is on by default.
- Cache file names encode the configuration (mode, industry dummies, macro interactions, grid type, model set), so different setups don't overwrite each other.

## Status and Known Limitations

- LSTM hyperparameter grids exist in `config.py`, but no LSTM model is in `ModelName` or the repository yet.
- The run configuration lives in `main()` rather than in a CLI or config file.
- Tests are currently limited to `test_visualization.py`.

## Citation

Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. *The Review of Financial Studies*, 33(5), 2223–2273.

## License

For academic research purposes. Add a `LICENSE` file before wider distribution.