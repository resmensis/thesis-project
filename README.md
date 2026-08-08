# Thesis Project: Asset Pricing with Machine Learning

A replication of Gu, Kelly, and Xiu (2020) - "Empirical Asset Pricing via Machine Learning"

## Overview

This project implements an expanding window forecasting framework for asset pricing using various machine learning algorithms. The code replicates the methodology from the seminal paper while providing flexibility for experimentation.

## Key Features

- **Expanding Window Forecasting**: Annual model refitting with expanding training windows
- **Gu et al. (2020) Models**: All models use HuberRegressor as specified in the paper
- **Complete Model Ensemble**: OLS, PCR, PLS, GBRT, Random Forest, MLP, **LSTM**
- **Feature Engineering**: 94 stock characteristics, industry dummies, macro interactions
- **Cross-Sectional Scaling**: Quantile-based normalization of characteristics
- **Caching**: Parquet-based caching for datasets and feature panels
- **Comprehensive Logging**: Detailed logging with per-module identification

## Installation

### Prerequisites

- Python 3.9+
- pip or conda

### Install Dependencies

```bash
# Core dependencies
pip install pandas numpy scikit-learn matplotlib torch

# Parquet support (REQUIRED)
pip install pyarrow

# Optional: fastparquet alternative
pip install fastparquet
```

## Project Structure

```
thesis-project/
├── config.py              # Configuration dataclasses (including ModelSelectionConfig)
├── data_inputs.py         # Data loading functions
├── dataset_builder.py     # Dataset construction and imputation
├── features.py            # Feature engineering and scaling
├── sample_splits.py       # Time-based data splitting
├── models_linear.py       # Linear models (HuberRegressor, PCR, PLS)
├── models_trees.py        # Tree models (GBRT, Random Forest)
├── models_mlp.py          # Neural networks (MLP)
├── models_lstm.py         # LSTM models
├── evaluation.py          # Evaluation metrics
├── expanding_window.py    # Expanding window logic
├── run_experiments.py     # Main entry point
├── io_utils.py            # I/O utilities and logging
└── README.md              # This file
```

## Usage

### Quick Start

```bash
# Run the full experiment (all models including LSTM, 1987-2021)
python run_experiments.py
```

### Configuration

Edit `config.py` to customize:

- **DataRegimeConfig**: `mode="full"` or `mode="coding"`
- **ExpandingWindowConfig**: Time periods and refitting frequency
- **ModelSelectionConfig**: Which models to run
- **CacheConfig**: Enable/disable caching
- **HyperGridConfig**: Hyperparameter grids for model tuning

### Model Selection

Control which models to run in `config.py`:

```python
# Run all models (including LSTM) - default
run_all_models = True
models_to_run = None

# Run only OLS-3 benchmark (fastest)
run_all_models = False
models_to_run = ["OLS_3"]

# Run custom subset
run_all_models = False
models_to_run = ["OLS_3", "GBRT", "MLP"]
```

### Running Modes

#### Full Mode (Default)
- Uses all 94 characteristics
- Creates all macro interactions (94 × 8 = 752 interaction terms)
- Full industry dummies (up to 74)

#### Coding Mode
- Subsampled data (500 stocks/month by default)
- Reduced feature set for faster iteration
- Limited macro interactions (12 by default)

Change mode in `config.py`:
```python
regime_cfg = DataRegimeConfig(mode="coding")
```

## Data Requirements

### Input Files

Place these files in the `data/` directory:

1. **datashare.csv**: 94 stock characteristics
   - Column 1: `permno` (stock identifier)
   - Column 2: `date`
   - Columns 3-96: 94 characteristics
   - Column 97: `sic2` (2-digit SIC code)

2. **crsp_monthly.csv**: CRSP monthly returns
   - Required columns: `permno`, `date`, `ret`, `dlret`

3. **Data2024_monthly_goyal.csv**: Macro predictors
   - Standard Goyal and Welch predictors

### Data Format

All dates should be in monthly format (YYYYMM or date strings). The code handles mixed date formats automatically.

## Expanding Window Methodology

### Timeline

```
Training:  1957───────────────────────────────────────────→ (expands yearly)
Validation:          1975────────────────────────────→ (rolls forward, 12 years)
Test:                    1987 1988 1989 ... 2016 ... 2021 (forecast yearly)
```

### Annual Refitting Process

For each forecast year Y (1987 to 2021):

1. **Training Period**: 1957 to (1974 + Y - 1987)
   - Example: For Y=1987, train on 1957-1974 (18 years)
   - Example: For Y=1988, train on 1957-1975 (19 years)
   - Expands by 1 year each iteration

2. **Validation Period**: 12 years ending at (1986 + Y - 1987)
   - Example: For Y=1987, validate on 1975-1986
   - Example: For Y=1988, validate on 1976-1987
   - Rolls forward, always 12 years

3. **Forecast Period**: 12 months of year Y
   - Generate predictions for all stocks
   - Evaluate performance

4. **Refit**: Increment year and repeat

### Timeframes

- **Original**: 1987-2016 (30 years)
- **Extended**: 1987-2021 (35 years)

The code runs continuously from 1987-2021 and saves intermediate evaluations at 2016.

## Output

### Directory Structure

```
output/
└── unified/
    ├── all_predictions.parquet      # All predictions 1987-2021
    ├── metrics_by_year.csv          # Full metrics (1987-2021)
    ├── metrics_by_year_original.csv # Intermediate metrics (1987-2016)
    ├── predictions_year_1987.parquet
    ├── predictions_year_1988.parquet
    ├── ...
    ├── models_year_1987.pkl
    ├── models_year_1988.pkl
    └── ...
```

### Metrics

For each model and year:

- `stock_r2`: Stock-level predictive R2
- `portfolio_r2`: Portfolio-level predictive R2
- `timing_sharpe`: Market timing strategy Sharpe ratio

### Cache

```
cache/
├── complete_dataset.parquet              # Merged dataset
├── complete_dataset_missingness_before.jpg  # Missingness visualization
├── complete_dataset_missingness_after.jpg   # After imputation
├── complete_dataset_missingness_comparison.jpg  # Before vs after
├── feature_panel_full.parquet          # Feature panel (full mode)
├── feature_panel_coding.parquet        # Feature panel (coding mode)
├── feature_cols_full.pkl               # Feature column names
└── feature_cols_coding.pkl
```

## Models (Gu et al. 2020 Specifications)

### Linear Models (All use HuberRegressor)

**Important:** Gu et al. (2020) use Huber robust regression for all "OLS" models, not standard OLS.

- **OLS_3**: HuberRegressor with 3 factors (size, value, momentum) - benchmark model
- **OLS_full**: HuberRegressor with all features
- **Huber**: HuberRegressor with tuned hyperparameters (epsilon, alpha)
- **PCR**: Principal Component Regression (PCA + HuberRegressor)
- **PLS**: Partial Least Squares regression

### Tree Models

- **GBRT**: Gradient Boosted Regression Trees
- **RandomForest**: Random Forest

### Neural Networks

- **MLP**: Multi-Layer Perceptron (feedforward neural network)
- **LSTM**: Long Short-Term Memory (recurrent neural network, runs only when `run_all_models=True`)

## Feature Engineering

### Preprocessing Steps

1. **Winsorization**: 1st and 99th percentiles by month
2. **Cross-Sectional Scaling**: QuantileTransformer to [-1, 1] (only 94 characteristics)
3. **Industry Dummies**: Top 74 SIC2 industries
4. **Macro Interactions**: Characteristic × macro variable

### Feature Sets

- **Full Mode**: 94 characteristics + 8 macro + 74 industry + 752 interactions = ~928 features
- **Coding Mode**: Reduced subset (configurable)

### Important Notes

- **Macro variables are NOT included in feature_cols** (removed to follow original paper)
- **Only the 94 characteristics are scaled** to [-1, 1]
- **Macro interactions are created AFTER scaling**

## Logging

Logs are saved to `logs/eap_ml_{mode}_{timestamp}.log` with:

- Timestamp
- Log level (INFO/DEBUG)
- Module name (e.g., `eap_ml.dataset_builder`, `eap_ml.models_linear`)
- Message

### Log Levels

- **INFO**: Major milestones, dataset shapes, model performance
- **DEBUG**: Detailed operations, parameter values, intermediate steps

## Reproducibility

- **Random Seed**: Set globally to 42 at the start
- **Torch Deterministic**: Enabled by default
- **Caching**: Ensures identical results across runs
- **Unified Timeframe**: Runs 1987-2021 continuously (not separate runs)

## Citation

Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. *The Review of Financial Studies*, 33(5), 2223-2273.

## License

This project is for academic research purposes.
