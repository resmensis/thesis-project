from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

from config import ExpandingWindowConfig, DataRegimeConfig, HyperGridConfig, ReproducibilityConfig, CacheConfig
from sample_splits import subset_timeframe
from models_linear import fit_pooled_huber_3, fit_pooled_huber_full, tune_pcr, tune_pls
from models_trees import tune_gbrt, tune_random_forest
from models_mlp import tune_mlp_models
from models_lstm import tune_lstm_models
from io_utils import ensure_dir, save_parquet, maybe_load_parquet, save_pickle, maybe_load_pickle
from evaluation import predictive_r2, compute_sharpe_ratio, aggregate_portfolio_predictions, build_market_timing_strategy

logger = logging.getLogger("eap_ml.expanding_window")


def get_yearly_splits(
    df: pd.DataFrame,
    config: ExpandingWindowConfig,
    test_end_year: int,
) -> list[Dict[str, Any]]:
    """
    Generate yearly train/val/test splits for expanding window forecasting.
    
    For each forecast year Y from test_start_year to test_end_year:
    - Train: config.train_start_year to (config.initial_train_end_year + (Y - config.test_start_year))
    - Val: (config.val_start_year + (Y - config.test_start_year) - 11) to (config.val_end_year + (Y - config.test_start_year))
    - Test: January Y to December Y
    
    Returns list of dicts with train_df, val_df, test_df, year, train_end, val_end.
    """
    splits = []
    
    for year in range(config.test_start_year, test_end_year + 1):
        # Calculate offsets
        year_offset = year - config.test_start_year
        
        # Training period expands by 1 year each iteration
        train_end_year = config.initial_train_end_year + year_offset
        train_end_date = f"{train_end_year}-12-31"
        
        # Validation period rolls forward, always 12 years
        val_start_year = config.val_end_year - 11 + year_offset
        val_end_year = config.val_end_year + year_offset
        val_start_date = f"{val_start_year}-01-01"
        val_end_date = f"{val_end_year}-12-31"
        
        # Test period: 12 months of forecast year
        test_start_date = f"{year}-01-01"
        test_end_date = f"{year}-12-31"
        
        # Extract subsets
        train_df = df[df["date"] <= train_end_date].copy()
        val_df = df[(df["date"] >= val_start_date) & (df["date"] <= val_end_date)].copy()
        test_df = df[(df["date"] >= test_start_date) & (df["date"] <= test_end_date)].copy()
        
        # Check for sufficient data
        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            logger.warning(f"Year {year}: Insufficient data, skipping. Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
            continue
        
        splits.append({
            "year": year,
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "train_end": train_end_date,
            "val_end": val_end_date,
        })
    
    logger.info(f"Generated {len(splits)} yearly splits for expanding window")
    return splits


def train_models_for_year(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    regime_config: DataRegimeConfig,
    grid_config: HyperGridConfig,
    repro_config: ReproducibilityConfig,
    year: int,
    models_to_run: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Train models for a given year using expanded training data.
    
    Gu et al. (2020) model specifications:
    - OLS_3: HuberRegressor with 3 factors (size, value, momentum)
    - OLS_full: HuberRegressor with all features
    - PCR: PCA + HuberRegressor
    - PLS: PLSRegression
    - GBRT: GradientBoostingRegressor
    - RF: RandomForestRegressor
    - MLP: MLPRegressor
    - LSTM: LSTM (only when run_all=True)
    
    Args:
        models_to_run: List of model names to train. If None, run all models.
                      Use ["OLS_3"] for OLS-3 only.
    """
    logger.info(f"Year {year}: Training models on {len(train_df)} samples")
    
    X_train = train_df[feature_cols].fillna(0.0)
    y_train = train_df["excess_ret_total"]
    X_val = val_df[feature_cols].fillna(0.0)
    y_val = val_df["excess_ret_total"]
    
    # Copy for scaling if needed
    X_train_s = X_train.copy()
    X_val_s = X_val.copy()
    
    models = {}
    
    # Determine which models to run
    if models_to_run is None:
        # Run all models (default)
        run_all = True
    elif isinstance(models_to_run, list) and len(models_to_run) == 1 and models_to_run[0] == "OLS_3":
        run_all = False
    else:
        run_all = False
    
    # Benchmark: OLS-3 with 3 factors (Gu et al. 2020 use Huber, not LinearRegression)
    benchmark_features = [c for c in regime_config.chara_cols_benchmark if c in X_train_s.columns]
    if len(benchmark_features) == 3:
        if run_all or "OLS_3" in (models_to_run or []):
            logger.debug(f"Training OLS_3 (Huber) with features: {benchmark_features}")
            models["OLS_3"] = fit_pooled_huber_3(X_train_s[benchmark_features], y_train)
    
    if run_all or "OLS_full" in (models_to_run or []):
        # OLS-full: HuberRegressor with all features (Gu et al. 2020)
        logger.debug("Training OLS_full (Huber) with all features")
        models["OLS_full"] = fit_pooled_huber_full(X_train_s, y_train)

    if run_all or "PCR" in (models_to_run or []):
        # PCR: PCA + HuberRegressor
        logger.debug("Training PCR")
        models["PCR"] = tune_pcr(X_train_s, y_train, X_val_s, y_val)["model"]

    if run_all or "PLS" in (models_to_run or []):
        # PLS: PLSRegression
        logger.debug("Training PLS")
        models["PLS"] = tune_pls(X_train_s, y_train, X_val_s, y_val)["model"]

    if run_all or "RandomForest" in (models_to_run or []):
        # Random Forest
        rf_grid = grid_config.rf_extended if grid_config.use_extended_grids else grid_config.rf_base
        logger.debug("Training Random Forest")
        models["RandomForest"] = tune_random_forest(X_train, y_train, X_val, y_val, random_state=repro_config.random_state, **rf_grid)["model"]

    if run_all or "GBRT" in (models_to_run or []):
        # GBRT
        gbrt_grid = grid_config.gbrt_extended if grid_config.use_extended_grids else grid_config.gbrt_base
        logger.debug("Training GBRT")
        models["GBRT"] = tune_gbrt(X_train, y_train, X_val, y_val, random_state=repro_config.random_state, **gbrt_grid)["model"]

    if run_all or "MLP" in (models_to_run or []):
        mlp_grid = grid_config.mlp_extended if grid_config.use_extended_grids else grid_config.mlp_base
        logger.debug("Training MLP")
        models["MLP"] = tune_mlp_models(X_train_s, y_train, X_val_s, y_val, random_state=repro_config.random_state, **mlp_grid)["model"]

    if run_all or "OLS_3" in (models_to_run or []):
        lstm_grid = grid_config.lstm_extended if grid_config.use_extended_grids else grid_config.lstm_base
        # LSTM: Only when running all models (computationally expensive)
        logger.debug("Training LSTM")
        try:
            models["LSTM"] = tune_lstm_models(
                train_df=train_df, 
                val_df=val_df, 
                feature_cols=feature_cols, 
                param_grid=lstm_grid, 
                random_state=repro_config.random_state
            )["model"]
        except Exception as e:
            logger.warning(f"LSTM training failed: {e}. Skipping LSTM for year {year}.")  
        
        
    logger.info(f"Year {year}: Trained {len(models)} models: {list(models.keys())}")
    return models


def load_or_initialize_prediction_store(
    prediction_path: Path,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load an existing wide prediction store or initialize a new one.

    The store has one row per (permno, date).
    """
    existing = maybe_load_parquet(prediction_path, enabled=True)

    if existing is not None:
        required = ["permno", "date", "excess_ret_total", "mv"]
        missing = [c for c in required if c not in existing.columns]
        if missing:
            raise KeyError(
                f"Existing prediction store is missing columns: {missing}"
            )

        duplicates = existing.duplicated(["permno", "date"]).sum()
        if duplicates > 0:
            raise ValueError(
                f"Existing prediction store contains {duplicates} "
                "duplicate (permno, date) rows."
            )

        return existing

    return test_df[["permno", "date", "excess_ret_total", "mv"]].copy()


def update_prediction_store(
    store: pd.DataFrame,
    yearly_predictions: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """
    Update one model's prediction column in the wide prediction store.

    Only rows whose (permno, date) keys appear in yearly_predictions are
    overwritten. Predictions for all other years remain unchanged.
    """
    pred_col = f"pred_{model_name}"
    key_cols = ["permno", "date"]

    new_preds = yearly_predictions[key_cols + [pred_col]].copy()

    duplicates = new_preds.duplicated(key_cols).sum()
    if duplicates > 0:
        raise ValueError(
            f"New predictions for {model_name} contain {duplicates} "
            f"duplicate {key_cols} rows."
        )

    if pred_col not in store.columns:
        store[pred_col] = np.nan

    new_col = f"{pred_col}__new"

    store = store.merge(
        new_preds.rename(columns={pred_col: new_col}),
        on=key_cols,
        how="left",
        validate="one_to_one",
    )

    # Overwrite only where a new prediction exists.
    # For all other stock-months, retain the existing prediction.
    store[pred_col] = store[new_col].combine_first(store[pred_col])
    store = store.drop(columns=[new_col])

    return store


def generate_predictions(
    models: Dict[str, Any],
    test_df: pd.DataFrame,
    feature_cols: list[str],
    regime_config: DataRegimeConfig,
    year: int,
) -> pd.DataFrame:
    """
    Generate wide-format predictions.

    Returns one row per stock-month and one prediction column per model:
    pred_<MODEL_NAME>. The column `mv` is retained for later
    value-weighted portfolio construction.
    """
    logger.info(f"Year {year}: Generating predictions for {len(test_df)} samples")

    required_cols = ["permno", "date", "excess_ret_total", "mv"]
    missing = [c for c in required_cols if c not in test_df.columns]
    if missing:
        raise KeyError(f"test_df is missing required columns: {missing}")

    predictions = test_df[required_cols].copy()
    predictions["year"] = year

    X_test_full = test_df[feature_cols].fillna(0.0)

    benchmark_features = regime_config.chara_cols_benchmark
    X_test_benchmark = None
    if len(benchmark_features) == 3:
        X_test_benchmark = test_df[benchmark_features].fillna(0.0)

    for model_name, model in models.items():
        pred_col = f"pred_{model_name}"

        if model_name == "OLS_3":
            if X_test_benchmark is None:
                raise ValueError(
                    "OLS_3 requires exactly three benchmark features, but "
                    f"regime_config.chara_cols_benchmark has "
                    f"{len(benchmark_features)} columns."
                )
            pred = model.predict(X_test_benchmark).ravel()
        else:
            pred = model.predict(X_test_full).ravel()

        if len(pred) != len(predictions):
            raise ValueError(
                f"Model {model_name} produced {len(pred)} predictions "
                f"for {len(predictions)} test observations."
            )

        predictions[pred_col] = pred

    logger.info(
        f"Year {year}: Generated predictions for models: {list(models.keys())}"
    )
    return predictions

def run_expanding_window(
    feature_panel: pd.DataFrame,
    feature_cols: list[str],
    regime_config: DataRegimeConfig,
    grid_config: HyperGridConfig,
    repro_config: ReproducibilityConfig,
    cache_config: CacheConfig,
    expand_config: ExpandingWindowConfig,
    window_name: str,
    output_dir: str,
    models_to_run: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Run expanding-window forecasting with annual refitting.

    The function only generates out-of-sample predictions.
    Portfolio construction and overall performance evaluation
    are performed separately after all predictions exist.
    """
    logger.info(f"Starting expanding window forecasting: {window_name}")

    test_end_year = expand_config.test_end_year_extended

    yearly_splits = get_yearly_splits(
        df=feature_panel,
        config=expand_config,
        test_end_year=test_end_year,
    )

    if len(yearly_splits) == 0:
        logger.error("No valid yearly splits generated. Check data availability.")
        return pd.DataFrame()

    outdir = Path(output_dir) / window_name
    ensure_dir(outdir)

    all_predictions_path = outdir / "all_predictions.parquet"

    # Initialize or load the cumulative wide prediction store.
    first_test_df = yearly_splits[0]["test_df"]
    prediction_store = load_or_initialize_prediction_store(
        prediction_path=all_predictions_path,
        test_df=first_test_df,
    )

    for split in yearly_splits:
        year = split["year"]
        train_df = split["train_df"]
        val_df = split["val_df"]
        test_df = split["test_df"]
        train_end = split["train_end"]
        val_end = split["val_end"]

        logger.info(
            f"Year {year}: training {train_df['date'].min()} to {train_end}, "
            f"validating {val_df['date'].min()} to {val_end}, "
            f"forecasting {year}"
        )

        models = train_models_for_year(
            train_df=train_df,
            val_df=val_df,
            feature_cols=feature_cols,
            regime_config=regime_config,
            grid_config=grid_config,
            repro_config=repro_config,
            year=year,
            models_to_run=models_to_run,
        )

        if not models:
            logger.warning(f"Year {year}: no models were trained; skipping.")
            continue

        yearly_predictions = generate_predictions(
            models=models,
            test_df=test_df,
            feature_cols=feature_cols,
            regime_config=regime_config,
            year=year,
        )

        # Add this year's stock-months to the cumulative store if needed.
        prediction_store = pd.concat(
            [
                prediction_store,
                yearly_predictions[
                    ["permno", "date", "excess_ret_total", "mv", "year"]
                ],
            ],
            ignore_index=True,
        ).drop_duplicates(
            subset=["permno", "date"],
            keep="last",
        )

        # Update only the models trained in this run.
        for model_name in models:
            prediction_store = update_prediction_store(
                store=prediction_store,
                yearly_predictions=yearly_predictions,
                model_name=model_name,
            )

        # Save the current year's wide predictions.
        yearly_path = outdir / f"predictions/predictions_year_{year}.parquet"
        save_parquet(
            yearly_predictions,
            yearly_path,
            enabled=cache_config.enabled,
        )

        # Save the fitted models for reproducibility.
        models_path = outdir / f"models/models_year_{year}.pkl"
        save_pickle(
            models,
            models_path,
            enabled=cache_config.enabled,
        )

        # Persist the cumulative store after each year.
        save_parquet(
            prediction_store,
            all_predictions_path,
            enabled=cache_config.enabled,
        )

    completed_models = [
        column.removeprefix("pred_")
        for column in prediction_store.columns
        if column.startswith("pred_")
    ]

    logger.info(
        f"Expanding window completed. Prediction store shape: "
        f"{prediction_store.shape}; completed models: {completed_models}"
    )

    return prediction_store