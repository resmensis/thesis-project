# empirical_asset_pricing_ml

Updated replication package scaffold for Gu, Kelly, and Xiu (2020).

## Included updates
- Modular architecture with cached intermediate products.
- Uses `datashare.csv`, `crsp_monthly.csv`, and `Data2024_monthly_goyal.csv`.
- Harmonizes heterogeneous monthly dates to month-end.
- Supports original sample and extended sample through 2021-12-31.
- Adds coding mode that reduces characteristics, interactions, industry dummies, and observations, but preserves the timeframe and keeps OLS-3 variables plus at least one monthly, quarterly, and annual characteristic.
- Sets `random_state = 42` across the project.
- Includes LSTM support via PyTorch.
- Provides base and extended hyperparameter grids.

## Setup
Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib jupyter torch pyarrow
```

## Main entry point
Run:

```bash
python run_experiments.py
```

## Important adaptation notes
- Adjust characteristic column names in `config.py`, especially the OLS-3 columns.
- Confirm which macro columns exist in `Data2024_monthly_goyal.csv`.
- Install required packages: pandas, numpy, scikit-learn, matplotlib, pyarrow, torch.
