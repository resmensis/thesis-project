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
    EvaluationConfig
)

from io_utils import ensure_dir, save_parquet, set_global_seed, setup_project_logger, maybe_load_parquet, save_pickle, maybe_load_pickle
from dataset_builder import build_complete_dataset
from features import build_feature_panel
from prediction_window import run_pred_window
from evaluation import evaluate_predictions
from visualization import  r2_metrics, dm_tests, r2_comparison, portfolio_metrics, cumulative_returns, portfolio_r2


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


cached_panel = maybe_load_parquet(feature_panel_path, enabled=True)

