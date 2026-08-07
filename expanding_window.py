from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

from config import ExpandingWindowConfig, DataRegimeConfig, HyperGridConfig, ReproducibilityConfig, CacheConfig
from sample_splits import subset_timeframe
from models_linear import fit_pooled_ols, tune_huber_regression, tune_pcr, tune_pls
from models_trees import tune_gbrt, tune_random_forest
from models_mlp import tune_mlp_models
from io_utils import ensure_dir, save_parquet, maybe_load_parquet, save_pickle, maybe_load_pickle

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
    
    Args:
        models_to_run: List of model names to train. If None, run all models.
                      Use ["OLS_3"] for OLS-3 only.
    """
    logger.info(f"Year {year}: Training models on {len(train_df)} samples")
    
    X_train = train_df[feature_cols].fillna(0.0)
    y_train = train_df["excess_ret_lead"]
    X_val = val_df[feature_cols].fillna(0.0)
    y_val = val_df["excess_ret_lead"]
    
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
    
    # Benchmark OLS with 3 factors
    benchmark_features = [c for c in [regime_config.ols3_size_col, regime_config.ols3_bm_col, regime_config.ols3_mom_col] if c in X_train_s.columns]
    if len(benchmark_features) == 3:
        if run_all or "OLS_3" in (models_to_run or []):
            models["OLS_3"] = fit_pooled_ols(X_train_s[benchmark_features], y_train)
    
    if run_all:
        # Full OLS
        models["OLS_full"] = fit_pooled_ols(X_train_s, y_train)
        
        # Huber
        models["Huber"] = tune_huber_regression(X_train_s, y_train, X_val_s, y_val)["model"]
        
        # PCR
        models["PCR"] = tune_pcr(X_train_s, y_train, X_val_s, y_val)["model"]
        
        # PLS
        models["PLS"] = tune_pls(X_train_s, y_train, X_val_s, y_val)["model"]
        
        # Tree models
        rf_grid = grid_config.rf_extended if grid_config.use_extended_grids else grid_config.rf_base
        gbrt_grid = grid_config.gbrt_extended if grid_config.use_extended_grids else grid_config.gbrt_base
        mlp_grid = grid_config.mlp_extended if grid_config.use_extended_grids else grid_config.mlp_base
        
        models["GBRT"] = tune_gbrt(X_train, y_train, X_val, y_val, random_state=repro_config.random_state, **gbrt_grid)["model"]
        models["RandomForest"] = tune_random_forest(X_train, y_train, X_val, y_val, random_state=repro_config.random_state, **rf_grid)["model"]
        models["MLP"] = tune_mlp_models(X_train_s, y_train, X_val_s, y_val, random_state=repro_config.random_state, **mlp_grid)["model"]
    
    logger.info(f"Year {year}: Trained {len(models)} models")
    return models


def generate_predictions(
    models: Dict[str, Any],
    test_df: pd.DataFrame,
    feature_cols: list[str],
    year: int,
) -> pd.DataFrame:
    """
    Generate predictions for all models on test data.
    """
    logger.info(f"Year {year}: Generating predictions for {len(test_df)} samples")
    
    X_test = test_df[feature_cols].fillna(0.0)
    y_test = test_df["excess_ret_lead"]
    
    predictions = []
    
    for model_name, model in models.items():
        pred = model.predict(X_test).ravel()
        
        # Create DataFrame with predictions
        pred_df = test_df[["permno", "date", "excess_ret_lead"]].copy()
        pred_df["model"] = model_name
        pred_df["prediction"] = pred
        pred_df["year"] = year
        pred_df["actual"] = y_test.values
        
        predictions.append(pred_df)
    
    result = pd.concat(predictions, ignore_index=True)
    logger.info(f"Year {year}: Generated {len(result)} predictions")
    return result


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
    evaluate_at_original: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run expanding window forecasting with annual refitting.
    
    Args:
        models_to_run: List of model names to run. Use ["OLS_3"] for OLS-3 only.
        evaluate_at_original: If True, save intermediate evaluation at test_end_year_original.
    
    Returns:
    - all_predictions: DataFrame with all predictions
    - metrics_by_year: DataFrame with metrics aggregated by year
    """
    logger.info(f"Starting expanding window forecasting: {window_name}")
    
    # Determine test end year - ALWAYS run to extended end for continuity
    test_end_year = expand_config.test_end_year_extended
    
    # Generate yearly splits
    yearly_splits = get_yearly_splits(
        feature_panel,
        expand_config,
        test_end_year,
    )
    
    if len(yearly_splits) == 0:
        logger.error("No valid yearly splits generated. Check data availability.")
        return pd.DataFrame(), pd.DataFrame()
    
    # Prepare output directory
    outdir = Path(output_dir) / window_name
    ensure_dir(outdir)
    
    all_predictions = []
    metrics_by_year = []
    
    for split in yearly_splits:
        year = split["year"]
        train_df = split["train_df"]
        val_df = split["val_df"]
        test_df = split["test_df"]
        train_end = split["train_end"]
        val_end = split["val_end"]
        
        logger.info(f"Year {year}: Training on {train_df['date'].min()} to {train_end}, "
                   f"validating on {val_df['date'].min()} to {val_end}, forecasting {year}")
        
        # Train models
        models = train_models_for_year(
            train_df, val_df, feature_cols,
            regime_config, grid_config, repro_config, year,
            models_to_run=models_to_run,
        )
        
        # Generate predictions
        predictions = generate_predictions(models, test_df, feature_cols, year)
        all_predictions.append(predictions)
        
        # Save predictions for this year
        pred_path = outdir / f"predictions_year_{year}.parquet"
        save_parquet(predictions, pred_path, enabled=cache_config.enabled)
        logger.debug(f"Saved predictions to {pred_path}")
        
        # Compute metrics for this year
        from evaluation import predictive_r2, compute_sharpe_ratio, aggregate_portfolio_predictions
        
        year_metrics = []
        for model_name in models.keys():
            model_preds = predictions[predictions["model"] == model_name]
            
            # Stock-level R2
            stock_r2 = predictive_r2(model_preds["actual"], model_preds["prediction"])
            
            # Portfolio-level R2
            port_df = aggregate_portfolio_predictions(
                model_preds[["permno", "date", "me"] if "me" in model_preds.columns else model_preds[["permno", "date"]]],
                model_preds["actual"],
                model_preds["prediction"]
            )
            port_r2 = predictive_r2(port_df["port_true"], port_df["port_pred"])
            
            # Market timing Sharpe
            from evaluation import build_market_timing_strategy
            timing = build_market_timing_strategy(
                model_preds[["permno", "date", "me"] if "me" in model_preds.columns else model_preds[["permno", "date"]]],
                model_preds["actual"],
                model_preds["prediction"]
            )
            timing_sharpe = compute_sharpe_ratio(timing["strategy_ret"])
            
            year_metrics.append({
                "year": year,
                "model": model_name,
                "stock_r2": stock_r2,
                "portfolio_r2": port_r2,
                "timing_sharpe": timing_sharpe,
            })
        
        metrics_by_year.extend(year_metrics)
        
        # Save models for this year (optional)
        models_path = outdir / f"models_year_{year}.pkl"
        save_pickle(models, models_path, enabled=cache_config.enabled)
        logger.debug(f"Saved models to {models_path}")
        
        # Save intermediate evaluation at original end year
        if evaluate_at_original and year == expand_config.test_end_year_original:
            logger.info(f"Reached original end year {year}, saving intermediate evaluation")
            metrics_original_path = outdir / "metrics_by_year_original.csv"
            pd.DataFrame(year_metrics).to_csv(metrics_original_path, index=False)
    
    # Concatenate all predictions
    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_by_year_df = pd.DataFrame(metrics_by_year)
    
    # Save aggregated results
    all_pred_path = outdir / "all_predictions.parquet"
    save_parquet(all_predictions_df, all_pred_path, enabled=cache_config.enabled)
    
    metrics_path = outdir / "metrics_by_year.csv"
    metrics_by_year_df.to_csv(metrics_path, index=False)
    
    logger.info(f"Expanding window completed: {len(all_predictions_df)} predictions, {len(metrics_by_year_df)} metric rows")
    return all_predictions_df, metrics_by_year_df