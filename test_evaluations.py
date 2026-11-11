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
from visualization import  r2_metrics_to_gu_table


repro_cfg = ReproducibilityConfig(random_state=42)
cache_cfg = CacheConfig(force_rebuild_feature_set=False, force_rebuild_dataset=False)
data_cfg = DataFilesConfig()
run_ctrl_cfg = RunControlConfig(feature_panel_only=False) 
regime_cfg = DataRegimeConfig(mode="full", include_industry_dummies=False, include_macro_interactions=False, use_all_macros=True, ols3_only_mode=False)
grid_cfg = HyperGridConfig()
freq_cfg = CharacteristicsFrequency()
log_cfg = LoggingConfig()
expand_cfg = ExpandingWindowConfig()
model_cfg = ModelSelectionConfig(run_all_models = False,
                                    models_to_run= ["OLS_3", "OLS_3_old", "OLS_full", "PCR_Huber", "PCR_Linear"]
                                )


complete_dataset_path=f"{cache_cfg.cache_dir}/complete_dataset.parquet"
descriptives_path=cache_cfg.descriptives_dir

feature_panel_path = f"{cache_cfg.cache_dir}/feature_panel_{regime_cfg.mode}.parquet"
feature_cols_path = f"{cache_cfg.cache_dir}/feature_cols_{regime_cfg.mode}.pkl"

industry_suffix = "wDummies" if regime_cfg.include_industry_dummies else "noDummies"

if not regime_cfg.include_macro_interactions:
    macro_suffix = "noInter"
elif regime_cfg.use_all_macros:
    macro_suffix = "allInter"
else:
    macro_suffix = "redInter"

feature_suffix = "ol3_only_mode" if regime_cfg.ols3_only_mode else f"{industry_suffix}_{macro_suffix}"

if regime_cfg.mode == "full":
    feature_panel_path = f"{cache_cfg.cache_dir}/feature_panel_{regime_cfg.mode}_{feature_suffix}.parquet"
    feature_cols_path = f"{cache_cfg.cache_dir}/feature_cols_{regime_cfg.mode}_{feature_suffix}.pkl"
else:
    feature_panel_path = f"{cache_cfg.cache_dir}/feature_panel_{regime_cfg.mode}.parquet"
    feature_cols_path = f"{cache_cfg.cache_dir}/feature_cols_{regime_cfg.mode}.pkl"


predictions_path = f"output/{regime_cfg.mode}/all_predictions.parquet" if regime_cfg.mode=="coding" else f"output/{regime_cfg.mode}/{feature_suffix}/all_predictions.parquet"



predictions = maybe_load_parquet(predictions_path, enabled=True)


evaluation = evaluate_predictions(
    predictions=predictions,
    regime_config=regime_cfg,
    output_dir="output",
    n_quantiles=10,
    dm_loss="squared_error",
    horizon=1,
    save_outputs=True,
    feature_importance_dir=f"output/{regime_cfg.mode}/feature_importance",
)

