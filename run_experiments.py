from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

from config import (
    CacheConfig,
    TimeframeConfig,
    DataFilesConfig,
    RunControlConfig,
    SplitConfig,
    DataRegimeConfig,
    HyperGridConfig,
    ReproducibilityConfig,
    CharacteristicsFrequency,
    LoggingConfig,
    ExpandingWindowConfig,
    ModelSelectionConfig,
)

from io_utils import ensure_dir, save_parquet, set_global_seed, setup_project_logger, maybe_load_parquet, save_pickle, maybe_load_pickle
from dataset_builder import build_complete_dataset, reduce_observations_for_coding, select_constant_stocks_for_coding_reduced
from sample_splits import subset_timeframe, chronological_split
from features import build_feature_panel
from models_linear import fit_pooled_huber_3, fit_pooled_huber_full, tune_huber_regression, tune_pcr, tune_pls
from models_trees import tune_gbrt, tune_random_forest
from models_mlp import tune_mlp_models
from expanding_window import run_expanding_window


def main():
    repro_cfg = ReproducibilityConfig(random_state=42)
    cache_cfg = CacheConfig()
    tf_cfg = TimeframeConfig()
    data_cfg = DataFilesConfig()
    run_ctrl_cfg = RunControlConfig()
    split_cfg = SplitConfig()
    regime_cfg = DataRegimeConfig(mode="full")
    grid_cfg = HyperGridConfig(use_extended_grids=False)
    freq_cfg = CharacteristicsFrequency()
    log_cfg = LoggingConfig()
    expand_cfg = ExpandingWindowConfig()
    model_cfg = ModelSelectionConfig()

    # Set global seed ONCE at the start - ensures reproducibility across all runs
    set_global_seed(repro_cfg.random_state, repro_cfg.torch_deterministic)

    logger, log_path = setup_project_logger(
        logger_name=log_cfg.logger_name,
        log_dir=log_cfg.log_dir,
        regime_mode=regime_cfg.mode,
        overwrite_log=log_cfg.overwrite_log,
        file_level_full=log_cfg.file_level_full,
        file_level_coding=log_cfg.file_level_coding,
        console_level_full=log_cfg.console_level_full,
        console_level_coding=log_cfg.console_level_coding,
    )

    logger.info("Starting experiment run.")
    logger.info(f"Random seed: {repro_cfg.random_state}")
    logger.info(f"Cache dir: {cache_cfg.cache_dir}")
    logger.info(f"Data regime: {regime_cfg.mode}")
    
    # Log model selection configuration
    if model_cfg.run_all_models:
        logger.info("Model selection: Running ALL models (including LSTM)")
        models_to_run = None  # None signals to run all models
    elif model_cfg.models_to_run is not None:
        logger.info(f"Model selection: Running custom subset: {model_cfg.models_to_run}")
        models_to_run = model_cfg.models_to_run
    else:
        logger.info("Model selection: Running default (all models)")
        models_to_run = None

    ensure_dir("output")
    ensure_dir(cache_cfg.cache_dir)

    # ------------------------------------------------------------------
    # 1. Build or load complete dataset (cache logic in run_experiments.py)
    # ------------------------------------------------------------------
    complete_cache_path = f"{cache_cfg.cache_dir}/complete_dataset.parquet"
    
    if cache_cfg.enabled and not cache_cfg.force_rebuild_dataset:
        complete = maybe_load_parquet(complete_cache_path, enabled=True)
        if complete is not None:
            logger.info(f"Loaded cached complete dataset from {complete_cache_path}")
        else:
            logger.info("Building complete dataset (cache miss).")
            complete = build_complete_dataset(
                datashare_path=data_cfg.datashare_path,
                crsp_path=data_cfg.crsp_monthly_path,
                macro_path=data_cfg.macro_path,
                possible_crsp_cols=data_cfg.possible_crsp_cols,
                possible_marco_cols=data_cfg.possible_marco_cols,
                cols_vars_monthly=freq_cfg.cols_vars_monthly,
                cols_vars_quarterly=freq_cfg.cols_vars_quarterly,
                cols_vars_annual=freq_cfg.cols_vars_annual,
                sic2_column=data_cfg.sic2_column,
            )
            save_parquet(complete, complete_cache_path, enabled=cache_cfg.save_complete_dataset)
    else:
        logger.info("Building complete dataset (cache disabled or force_rebuild).")
        complete = build_complete_dataset(
            datashare_path=data_cfg.datashare_path,
            crsp_path=data_cfg.crsp_monthly_path,
            macro_path=data_cfg.macro_path,
            possible_crsp_cols=data_cfg.possible_crsp_cols,
            possible_marco_cols=data_cfg.possible_marco_cols,
            cols_vars_monthly=freq_cfg.cols_vars_monthly,
            cols_vars_quarterly=freq_cfg.cols_vars_quarterly,
            cols_vars_annual=freq_cfg.cols_vars_annual,
            sic2_column=data_cfg.sic2_column,
        )
        save_parquet(complete, complete_cache_path, enabled=cache_cfg.save_complete_dataset)

    logger.info(f"Complete dataset shape: {complete.shape}")

    # Check if we should stop after dataset creation
    if run_ctrl_cfg.dataset_creation_only:
        print("Dataset creation only mode is ON.")
        print(f"Complete dataset created and cached at: {complete_cache_path}")
        return

    # Apply regime-specific reductions
    if regime_cfg.mode == "coding":
        logger.info("Applying coding-mode sampling reduction (500 stocks per month, random selection).")
        complete = reduce_observations_for_coding(complete, max_stocks_per_month=regime_cfg.coding_max_stocks_per_month, random_state=repro_cfg.random_state)
    
    elif regime_cfg.mode == "coding_reduced":
        logger.info("Applying coding_reduced mode: selecting 500 constant stocks across time.")
        # Select 500 permno based on random seed - constant across entire time period
        selected_permno = select_constant_stocks_for_coding_reduced(
            complete, 
            n_stocks=regime_cfg.coding_reduced_stocks, 
            random_state=repro_cfg.random_state
        )
        complete = complete[complete["permno"].isin(selected_permno)].copy()
        logger.info(f"Reduced to {len(selected_permno)} stocks, {len(complete)} observations")
        
        # Save reduced dataset to separate file
        reduced_cache_path = f"{cache_cfg.cache_dir}/complete_dataset_coding_reduced.parquet"
        save_parquet(complete, reduced_cache_path, enabled=cache_cfg.save_complete_dataset)
        logger.info(f"Saved reduced dataset to {reduced_cache_path}")

    # ------------------------------------------------------------------
    # 2. Build or load feature panel (cache logic in run_experiments.py)
    # ------------------------------------------------------------------
    feature_panel_path = f"{cache_cfg.cache_dir}/feature_panel_{regime_cfg.mode}.parquet"
    feature_cols_path = f"{cache_cfg.cache_dir}/feature_cols_{regime_cfg.mode}.pkl"
    
    if cache_cfg.enabled and not cache_cfg.force_refit_models:
        cached_panel = maybe_load_parquet(feature_panel_path, enabled=True)
        cached_cols = maybe_load_pickle(feature_cols_path, enabled=True)
        
        if cached_panel is not None and cached_cols is not None:
            logger.info(f"Loaded cached feature panel from {feature_panel_path}")
            feature_panel = cached_panel
            feature_cols = cached_cols
        else:
            logger.info("Building feature panel (cache miss).")
            feature_panel, feature_cols = build_feature_panel(
                complete, 
                regime_config=regime_cfg, 
                include_macro_interactions=True,
                reduced_macro_vars=regime_cfg.coding_reduced_macro_vars if regime_cfg.mode == "coding_reduced" else None,
            )
            save_parquet(feature_panel, feature_panel_path, enabled=cache_cfg.save_feature_panel)
            save_pickle(feature_cols, feature_cols_path, enabled=cache_cfg.save_feature_panel)
    else:
        logger.info("Building feature panel (cache disabled or force_refit).")
        feature_panel, feature_cols = build_feature_panel(
            complete, 
            regime_config=regime_cfg, 
            include_macro_interactions=True,
            reduced_macro_vars=regime_cfg.coding_reduced_macro_vars if regime_cfg.mode == "coding_reduced" else None,
        )
        save_parquet(feature_panel, feature_panel_path, enabled=cache_cfg.save_feature_panel)
        save_pickle(feature_cols, feature_cols_path, enabled=cache_cfg.save_feature_panel)
    
    logger.info(f"Feature panel shape: {feature_panel.shape}")
    logger.info(f"Number of feature columns: {len(feature_cols)}")

    # Check if we should stop after feature panel creation
    if run_ctrl_cfg.feature_panel_only:
        print("Feature panel only mode is ON.")
        print(f"Feature panel created and cached at: {feature_panel_path}")
        print(f"Feature columns cached at: {feature_cols_path}")
        return

    # ------------------------------------------------------------------
    # 3. Run expanding window forecasting
    # ------------------------------------------------------------------
    logger.info("Starting expanding window forecasting.")
    
    # IMPORTANT: Run both timeframes in ONE continuous execution
    # The random seed is set globally above, ensuring reproducibility
    # We run to extended end (2021) and evaluate at both 2016 and 2021
    
    # Single unified run - evaluates at both original (2016) and extended (2021)
    logger.info("Running unified expanding window (1987-2021) with evaluation at 2016 and 2021")
    
    predictions, metrics = run_expanding_window(
        feature_panel=feature_panel,
        feature_cols=feature_cols,
        regime_config=regime_cfg,
        grid_config=grid_cfg,
        repro_config=repro_cfg,
        cache_config=cache_cfg,
        expand_config=expand_cfg,
        window_name="unified",  # Single unified window
        output_dir="output",
        models_to_run=models_to_run,  # Determined by ModelSelectionConfig
        evaluate_at_original=True,  # Save intermediate evaluation at 2016
    )
    
    logger.info("Expanding window completed.")
    logger.info(f"Total predictions: {len(predictions)}")
    logger.info(f"Metrics by year: {len(metrics)} rows")
    
    # Results are saved to:
    # - output/unified/all_predictions.parquet
    # - output/unified/metrics_by_year.csv (full 1987-2021)
    # - output/unified/metrics_by_year_original.csv (1987-2016 only)


if __name__ == "__main__":
    main()