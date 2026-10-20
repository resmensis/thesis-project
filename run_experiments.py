from __future__ import annotations
from pathlib import Path
import pandas as pd

from config import (
    CacheConfig,
    DataFilesConfig,
    RunControlConfig,
    DataRegimeConfig,
    HyperGridConfig,
    ReproducibilityConfig,
    CharacteristicsFrequency,
    LoggingConfig,
    ExpandingWindowConfig,
    ModelSelectionConfig,
)

from io_utils import ensure_dir, save_parquet, set_global_seed, setup_project_logger, maybe_load_parquet, save_pickle, maybe_load_pickle
from dataset_builder import build_complete_dataset
from features import build_feature_panel
from expanding_window import run_expanding_window
from evaluation import evaluate_predictions


def main():
    repro_cfg = ReproducibilityConfig(random_state=42)
    cache_cfg = CacheConfig(force_rebuild_feature_set=True, force_rebuild_dataset=True)
    data_cfg = DataFilesConfig()
    run_ctrl_cfg = RunControlConfig()
    regime_cfg = DataRegimeConfig(mode="coding", include_macro_interactions=False, ols3_only_mode=True)
    grid_cfg = HyperGridConfig(use_extended_grids=False)
    freq_cfg = CharacteristicsFrequency()
    log_cfg = LoggingConfig()
    expand_cfg = ExpandingWindowConfig()
    model_cfg = ModelSelectionConfig(run_all_models = False,
                                     models_to_run=["LSTM"]
                                    ) # models_to_run = ["OLS_3", "OLS_full", "RandomForest"] 

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
    random_state=repro_cfg.random_state
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
    # 1. Build or load complete dataset
    # ------------------------------------------------------------------
    logger.info("Building complete dataset.")

    complete_dataset_path=f"{cache_cfg.cache_dir}/complete_dataset.parquet"
    descriptives_path=cache_cfg.descriptives_dir

    feature_panel_path = f"{cache_cfg.cache_dir}/feature_panel_{regime_cfg.mode}.parquet"
    feature_cols_path = f"{cache_cfg.cache_dir}/feature_cols_{regime_cfg.mode}.pkl"

    predictions_path = f"output/{regime_cfg.mode}/all_predictions.parquet"


    if cache_cfg.enabled and not cache_cfg.force_rebuild_dataset:
        cached_panel = maybe_load_parquet(feature_panel_path, enabled=True)
        cached_cols = maybe_load_pickle(feature_cols_path, enabled=True)

        cached_dataset = maybe_load_parquet(complete_dataset_path, enabled=True)
        
        if cached_panel is not None and cached_cols is not None and not cache_cfg.force_rebuild_feature_set:
                logger.info(f"Feature panel and cols found in cashe. Skipping building/loading dataset.")
        elif cached_dataset is not None:
            logger.info(f"Loading cached dataset from {complete_dataset_path}")
            complete = cached_dataset
            logger.info(f"Complete dataset shape: {complete.shape}")
        else:
            logger.info("Building complete dataset (no cache found).")
            complete = build_complete_dataset(
                datashare_path=data_cfg.datashare_path,
                crsp_path=data_cfg.crsp_monthly_path,
                macro_path=data_cfg.macro_path,
                out_path=complete_dataset_path,
                descriptives_path=descriptives_path,
                cache_enabled=cache_cfg.enabled,
                possible_crsp_cols=data_cfg.possible_crsp_cols,
                possible_marco_cols=data_cfg.possible_marco_cols,
                cols_chara=data_cfg.chara_cols,
                cols_vars_monthly=freq_cfg.cols_vars_monthly,
                cols_vars_quarterly=freq_cfg.cols_vars_quarterly,
                cols_vars_annual=freq_cfg.cols_vars_annual,
                random_state=random_state
            )
            logger.info(f"Complete dataset shape: {complete.shape}")
    else:
        logger.info("Building complete dataset (cache disabled or force_refit).")
        complete = build_complete_dataset(
            datashare_path=data_cfg.datashare_path,
            crsp_path=data_cfg.crsp_monthly_path,
            macro_path=data_cfg.macro_path,
            out_path=complete_dataset_path,
            descriptives_path=descriptives_path,
            cache_enabled=cache_cfg.enabled,
            possible_crsp_cols=data_cfg.possible_crsp_cols,
            possible_marco_cols=data_cfg.possible_marco_cols,
            cols_chara=data_cfg.chara_cols,
            cols_vars_monthly=freq_cfg.cols_vars_monthly,
            cols_vars_quarterly=freq_cfg.cols_vars_quarterly,
            cols_vars_annual=freq_cfg.cols_vars_annual,
            random_state=random_state
        )
        logger.info(f"Complete dataset shape: {complete.shape}")


    # Check if we should stop after dataset creation
    if run_ctrl_cfg.dataset_creation_only:
        print("Dataset creation only mode is ON.")
        print(f"Complete dataset created and cached at: {complete_dataset_path}")
        return

    
    # ------------------------------------------------------------------
    # 2. Build or load feature panel
    # ------------------------------------------------------------------  
    if cache_cfg.enabled and not cache_cfg.force_rebuild_feature_set:
        cached_panel = maybe_load_parquet(feature_panel_path, enabled=True)
        cached_cols = maybe_load_pickle(feature_cols_path, enabled=True)
        
        if cached_panel is not None and cached_cols is not None:
            logger.info(f"Loading cached feature panel from {feature_panel_path}")
            feature_panel = cached_panel
            feature_cols = cached_cols
        else:
            logger.info("Building feature panel (no cache found).")
            feature_panel, feature_cols = build_feature_panel(
                complete,
                out_path_panel=feature_panel_path,
                out_path_cols=feature_cols_path,
                descriptives_path=descriptives_path,
                random_state=random_state,
                regime_config=regime_cfg,
                data_config=data_cfg,
                save_feature_panel=cache_cfg.save_feature_panel,
            )
    else:
        logger.info("Building feature panel (cache disabled or force_refit).")
        feature_panel, feature_cols = build_feature_panel(
            complete,
            out_path_panel=feature_panel_path,
            out_path_cols=feature_cols_path,
            descriptives_path=descriptives_path,
            random_state=random_state,
            regime_config=regime_cfg,
            data_config=data_cfg,
            save_feature_panel=cache_cfg.save_feature_panel,
        )
    
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
    
    predictions = run_expanding_window(
        feature_panel=feature_panel,
        feature_cols=feature_cols,
        regime_config=regime_cfg,
        grid_config=grid_cfg,
        repro_config=repro_cfg,
        cache_config=cache_cfg,
        expand_config=expand_cfg,
        window_name=regime_cfg.mode,
        output_dir="output",
        models_to_run=models_to_run,  # Determined by ModelSelectionConfig
    )
    
    logger.info("Expanding window completed.")
    logger.info(f"Total predictions: {len(predictions)}")
    
    # Results are saved to:
    # - output/unified/all_predictions.parquet
    # - output/unified/metrics_by_year.csv (full 1987-2021)
    # - output/unified/metrics_by_year_original.csv (1987-2016 only)


    evaluation = evaluate_predictions(
        predictions=predictions,
        regime_config=regime_cfg,
        output_dir="output",
        n_quantiles=10,
        dm_loss="squared_error",
        horizon=1,
        save_outputs=True,
    )




if __name__ == "__main__":
    main()