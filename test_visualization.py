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
    EvaluationConfig
)

from io_utils import ensure_dir, save_parquet, set_global_seed, setup_project_logger, maybe_load_parquet, save_pickle, maybe_load_pickle
from dataset_builder import build_complete_dataset
from features import build_feature_panel
from expanding_window import run_expanding_window
from evaluation import evaluate_predictions
from visualization import  r2_metrics, dm_tests, r2_comparison_to_gu_table


repro_cfg = ReproducibilityConfig(random_state=42)
cache_cfg = CacheConfig(force_rebuild_feature_set=False, force_rebuild_dataset=False)
data_cfg = DataFilesConfig()
run_ctrl_cfg = RunControlConfig(feature_panel_only=False) 
regime_cfg = DataRegimeConfig(mode="coding", include_industry_dummies=False, include_macro_interactions=False, use_all_macros=True, ols3_only_mode=False)
grid_cfg = HyperGridConfig()
freq_cfg = CharacteristicsFrequency()
log_cfg = LoggingConfig()
expand_cfg = ExpandingWindowConfig()
model_cfg = ModelSelectionConfig(run_all_models = False,
                                    models_to_run= ["OLS_3", "OLS_3_old", "OLS_full", "PCR_Huber", "PCR_Linear"]
                                )
eval_cfg = EvaluationConfig()


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

grid_suffix = "baseGrid" if not grid_cfg.use_extended_grids else f"extGrid"
model_suffix = "allModels" if model_cfg.run_all_models else "selModels"

pred_window_path = f"{regime_cfg.mode}/{model_suffix}_{grid_suffix}" if regime_cfg.mode=="coding" else f"{regime_cfg.mode}/{feature_suffix}_{model_suffix}_{grid_suffix}"
pred_path = f"output/{pred_window_path}"
all_predictions_path = f"{pred_path}/all_predictions.parquet" 

visualization_path = f"{pred_path}/visualization"



path_r2_metrics = f"{pred_path}/r2_metrics.csv"
path_r2_metrics_original = f"{pred_path}/r2_metrics_original.csv"
path_dm_tests = f"{pred_path}/dm_tests.csv"



r2_metrics_table = r2_metrics(
    r2_metrics=pd.read_csv(path_r2_metrics),
    eval_config=eval_cfg,
    output_path=f"{visualization_path}/r2_extended.tex",
    sample_years="1987 -- 2021",
    label="tab:stock_prediction_performance_extended"
    )

dm_tests_table = dm_tests(
    dm_tests=pd.read_csv(path_dm_tests),
    eval_config=eval_cfg,
    output_path=f"{visualization_path}/dm_tests.tex",
    sample_years="1987 -- 2021",
    )

r2_comparison_table = r2_comparison_to_gu_table(
    r2_metrics_original=pd.read_csv(path_r2_metrics_original),
    r2_metrics=pd.read_csv(path_r2_metrics),
    eval_config=eval_cfg,
    output_path=f"{visualization_path}/r2_comparison.tex",
    original_years="1987 -- 2016",
    extended_years="1987 -- 2021",
    )
