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
    WindowConfig,
    ModelSelectionConfig,
    EvaluationConfig,
    ModelName
)


from io_utils import ensure_dir, set_global_seed, setup_project_logger, maybe_load_parquet, maybe_load_pickle, has_expected_prediction_files
from dataset_builder import build_complete_dataset
from features import build_feature_panel
from prediction_window import run_pred_window
from evaluation import evaluate_predictions, create_evaulation_dataset


def main():
    repro_cfg = ReproducibilityConfig(random_state=42)
    cache_cfg = CacheConfig(force_rebuild_feature_set=False, force_rebuild_dataset=False)
    data_cfg = DataFilesConfig()
    run_ctrl_cfg = RunControlConfig(feature_panel_only=False, load_predictions=True) 
    regime_cfg = DataRegimeConfig(mode="full", include_industry_dummies=False, include_macro_interactions=False, use_all_macros=True, ols3_only_mode=False)
    grid_cfg = HyperGridConfig()
    freq_cfg = CharacteristicsFrequency()
    log_cfg = LoggingConfig()
    window_cfg = WindowConfig(window_type="expanding")
    model_cfg = ModelSelectionConfig(run_all_models = True,
                                     models_to_run= []
                                    )
    eval_cfg = EvaluationConfig()
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
    logger.info(f"Window type: {window_cfg.window_type}")
    
    # Log model selection configuration
    if model_cfg.run_all_models:
        logger.info("Model selection: Running ALL models")
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
    # 0. Paths
    # ------------------------------------------------------------------
    complete_dataset_path=f"{cache_cfg.cache_dir}/complete_dataset.parquet"
    fama_french_charas_path=f"{cache_cfg.cache_dir}/fama_french_charas.parquet"
    descriptives_path=cache_cfg.descriptives_dir

    industry_suffix = "wDummies" if regime_cfg.include_industry_dummies else "noDummies"

    if not regime_cfg.include_macro_interactions:
        macro_suffix = "noInter"
    elif regime_cfg.use_all_macros:
        macro_suffix = "allInter"
    else:
        macro_suffix = "redInter"

    feature_suffix = (
        "ols3_only_mode" 
        if regime_cfg.ols3_only_mode 
        else f"{industry_suffix}_{macro_suffix}"
    )

    if regime_cfg.mode == "full":
        feature_panel_path = f"{cache_cfg.cache_dir}/feature_panel_{regime_cfg.mode}_{feature_suffix}.parquet"
        feature_cols_path = f"{cache_cfg.cache_dir}/feature_cols_{regime_cfg.mode}_{feature_suffix}.pkl"
    else:
        feature_panel_path = f"{cache_cfg.cache_dir}/feature_panel_{regime_cfg.mode}.parquet"
        feature_cols_path = f"{cache_cfg.cache_dir}/feature_cols_{regime_cfg.mode}.pkl"

    grid_suffix = "baseGrid" if not grid_cfg.use_extended_grids else f"extGrid"
    model_suffix = "allModels" if model_cfg.run_all_models else "selModels"

    pred_window_path = (
        f"{regime_cfg.mode}_mode-{window_cfg.window_type}_window" 
        if regime_cfg.mode=="coding" 
        else f"{regime_cfg.mode}_mode-{window_cfg.window_type}_window/{feature_suffix}_{model_suffix}_{grid_suffix}"
    )
    pred_path = f"output/{pred_window_path}"
    all_predictions_path = f"{pred_path}/all_predictions.parquet"
    all_predictions_with_fama_path = f"{pred_path}/all_predictions_fama_french.parquet"
    yearly_predictions_path = f"{pred_path}/predictions"

    visualization_path = f"{pred_path}/visualization"

    # ------------------------------------------------------------------
    # 1. Build or load complete dataset
    # ------------------------------------------------------------------
    if run_ctrl_cfg.load_predictions and Path(all_predictions_path).exists() and has_expected_prediction_files(yearly_predictions_path):
        pass
    else:
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
                    fama_french_charas_path=fama_french_charas_path,
                    descriptives_path=descriptives_path,
                    cache_enabled=cache_cfg.enabled,
                    possible_crsp_cols=data_cfg.possible_crsp_cols,
                    possible_marco_cols=data_cfg.possible_marco_cols,
                    cols_chara=data_cfg.chara_cols,
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
                fama_french_charas_path=fama_french_charas_path,
                descriptives_path=descriptives_path,
                cache_enabled=cache_cfg.enabled,
                possible_crsp_cols=data_cfg.possible_crsp_cols,
                possible_marco_cols=data_cfg.possible_marco_cols,
                cols_chara=data_cfg.chara_cols,
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
    if run_ctrl_cfg.load_predictions and Path(all_predictions_path).exists() and has_expected_prediction_files(yearly_predictions_path):
        pass
    else:
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
    # 3. Run prediction window forecasting
    # ------------------------------------------------------------------
    logger.info(f"Starting windowized forecasting.")
    logger.info(f"Running unified {window_cfg.window_type} window (1987-2021) with evaluation at 2016 and 2021")

    if run_ctrl_cfg.load_predictions and Path(all_predictions_path).exists() and has_expected_prediction_files(yearly_predictions_path):
        pass
    else:
        predictions = run_pred_window(
            feature_panel=feature_panel,
            feature_cols=feature_cols,
            regime_config=regime_cfg,
            grid_config=grid_cfg,
            repro_config=repro_cfg,
            cache_config=cache_cfg,
            window_config=window_cfg,
            output_dir=pred_path,
            models_to_run=models_to_run,  
        )
        
        logger.info(f"{window_cfg.window_type} window completed.")
        logger.info(f"Total predictions: {len(predictions)}")
    

    # ------------------------------------------------------------------
    # 4. Run prediction evakuation
    # ------------------------------------------------------------------
    if not Path(all_predictions_with_fama_path).exists():
        if run_ctrl_cfg.load_predictions and Path(all_predictions_path).exists() and has_expected_prediction_files(yearly_predictions_path):
            predictions = maybe_load_parquet(all_predictions_path, enabled=True)
            fama_french_charas_df = maybe_load_parquet(fama_french_charas_path, enabled=True)

            all_predictions_with_fama = create_evaulation_dataset(
                predictions=predictions,
                fama_french_charas_df=fama_french_charas_df,
                output_dir=pred_path
            )
        else: 
            raise KeyError("missing data")
    else:
        all_predictions_with_fama = maybe_load_parquet(all_predictions_with_fama_path, enabled=True)


        

    evaluation = evaluate_predictions(
        predictions=all_predictions_with_fama,
        regime_config=regime_cfg,
        eval_config=eval_cfg,
        output_dir=pred_path,
        n_quantiles=10,
        save_outputs=True,
        feature_importance_dir=f"{pred_path}/feature_importance" ,
    )




if __name__ == "__main__":
    main()