from __future__ import annotations
from pathlib import Path
import pandas as pd

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
)

from io_utils import ensure_dir, save_parquet, set_global_seed, setup_project_logger, maybe_load_parquet, save_pickle, maybe_load_pickle
from dataset_builder import build_complete_dataset, reduce_observations_for_coding
from sample_splits import subset_timeframe, chronological_split
from features import build_feature_panel
from models_linear import fit_pooled_ols, tune_huber_regression, tune_pcr, tune_pls
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

    ensure_dir("output")
    ensure_dir(cache_cfg.cache_dir)

    # ------------------------------------------------------------------
    # 1. Build or load complete dataset
    # ------------------------------------------------------------------
    logger.info("Building complete dataset.")
    complete = build_complete_dataset(
        datashare_path=data_cfg.datashare_path,
        crsp_path=data_cfg.crsp_monthly_path,
        macro_path=data_cfg.macro_path,
        out_path=f"{cache_cfg.cache_dir}/complete_dataset.parquet",
        cache_enabled=cache_cfg.enabled,
        force_rebuild=cache_cfg.force_rebuild_dataset,
        possible_crsp_cols=data_cfg.possible_crsp_cols,
        possible_marco_cols=data_cfg.possible_marco_cols,
        cols_vars_monthly=freq_cfg.cols_vars_monthly,
        cols_vars_quarterly=freq_cfg.cols_vars_quarterly,
        cols_vars_annual=freq_cfg.cols_vars_annual,
        sic2_column=data_cfg.sic2_column,
    )

    logger.info(f"Complete dataset shape: {complete.shape}")

    if run_ctrl_cfg.dataset_creation_only:
        print("Dataset creation only mode is ON.")
        print(f"Complete dataset created and cached at: {cache_cfg.cache_dir}/complete_dataset.parquet")
        return

    if regime_cfg.mode == "coding":
        logger.info("Applying coding-mode sampling reduction.")
        complete = reduce_observations_for_coding(complete, max_stocks_per_month=regime_cfg.coding_max_stocks_per_month, random_state=repro_cfg.random_state)

    # ------------------------------------------------------------------
    # 2. Build or load feature panel
    # ------------------------------------------------------------------
    feature_panel_path = f"{cache_cfg.cache_dir}/feature_panel_{regime_cfg.mode}.parquet"
    feature_cols_path = f"{cache_cfg.cache_dir}/feature_cols_{regime_cfg.mode}.pkl"
    
    if cache_cfg.enabled and not cache_cfg.force_refit_models:
        cached_panel = maybe_load_parquet(feature_panel_path, enabled=True)
        cached_cols = maybe_load_pickle(feature_cols_path, enabled=True)
        
        if cached_panel is not None and cached_cols is not None:
            logger.info(f"Loading cached feature panel from {feature_panel_path}")
            feature_panel = cached_panel
            feature_cols = cached_cols
        else:
            logger.info("Building feature panel (no cache found).")
            feature_panel, feature_cols = build_feature_panel(complete, regime_config=regime_cfg, include_macro_interactions=True)
            save_parquet(feature_panel, feature_panel_path, enabled=cache_cfg.save_feature_panel)
            save_pickle(feature_cols, feature_cols_path, enabled=cache_cfg.save_feature_panel)
    else:
        logger.info("Building feature panel (cache disabled or force_refit).")
        feature_panel, feature_cols = build_feature_panel(complete, regime_config=regime_cfg, include_macro_interactions=True)
        save_parquet(feature_panel, feature_panel_path, enabled=cache_cfg.save_feature_panel)
        save_pickle(feature_cols, feature_cols_path, enabled=cache_cfg.save_feature_panel)
    
    logger.info(f"Feature panel shape: {feature_panel.shape}")
    logger.info(f"Number of feature columns: {len(feature_cols)}")

    # ------------------------------------------------------------------
    # 3. Run expanding window forecasting
    # ------------------------------------------------------------------
    logger.info("Starting expanding window forecasting.")
    
    windows = {}
    if tf_cfg.run_original_window:
        windows["original"] = tf_cfg.original_end_date
    if tf_cfg.run_extended_window:
        windows["extended_2021"] = tf_cfg.extended_end_date

    all_predictions = {}
    all_metrics = {}
    
    for window_name, end_date in windows.items():
        logger.info(f"Running expanding window: {window_name} (until {end_date})")
        
        predictions, metrics = run_expanding_window(
            feature_panel=feature_panel,
            feature_cols=feature_cols,
            regime_config=regime_cfg,
            grid_config=grid_cfg,
            repro_config=repro_cfg,
            cache_config=cache_cfg,
            expand_config=expand_cfg,
            window_name=window_name,
            output_dir="output",
        )
        
        all_predictions[window_name] = predictions
        all_metrics[window_name] = metrics
    
    logger.info("All expanding window runs completed.")
    logger.info(f"Total predictions: {sum(len(df) for df in all_predictions.values())}")


if __name__ == "__main__":
    main()