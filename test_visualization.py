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

grid_suffix = "baseGrid" if not grid_cfg.use_extended_grids else f"extGrid"
model_suffix = "allModels" if model_cfg.run_all_models else "selModels"



for i in ["expanding","rolling"]:
    pred_window_path = (
        f"{regime_cfg.mode}" #_mode-{window_cfg.window_type}_window
        if regime_cfg.mode=="coding" 
        else f"{regime_cfg.mode}_mode-{i}_window/{feature_suffix}_{model_suffix}_{grid_suffix}" #window_cfg.window_type
    )
    pred_path = f"output/{pred_window_path}"
    all_predictions_path = f"{pred_path}/all_predictions.parquet"
    all_predictions_with_fama_path = f"{pred_path}/all_predictions_fama_french.parquet"
    yearly_predictions_path = f"{pred_path}/predictions"

    evaluation_path = f"{pred_path}/evaluation"

    visualization_path = f"{pred_path}/visualization"
    # ------------------------------------------------------------------
    # 
    # ------------------------------------------------------------------


    path_r2_metrics = f"{pred_path}/r2_metrics.csv"
    path_r2_metrics_original = f"{pred_path}/r2_metrics_original.csv"
    path_dm_tests = f"{pred_path}/dm_tests.csv"
    path_portfolio_metrics = f"{pred_path}/portfolio_metrics.csv"
    path_portfolio_returns= f"{pred_path}/portfolio_returns.parquet"
    path_portfolio_presp_metrics = f"{pred_path}/fama_french_portfolio_metrics.csv"
    path_sp500_returns = f"{pred_path}/sp500_portfolio_returns.parquet"
    path_sp500_metrics = f"{pred_path}/sp500_portfolio_metrics.csv"



    """

    """

    r2_metrics_table = r2_metrics(
        r2_metrics=pd.read_csv(path_r2_metrics),
        eval_config=eval_cfg,
        output_path=visualization_path,
        file_name=f"r2_extended_{i}",
        sample_years="1987 -- 2021",
        label="tab:stock_prediction_performance_extended"
    )

    r2_metrics_original_table = r2_metrics(
        r2_metrics=pd.read_csv(path_r2_metrics_original),
        eval_config=eval_cfg,
        output_path=visualization_path,
        file_name=f"r2_original_{i}",
        sample_years="1987 -- 2016",
        label="tab:stock_prediction_performance_original"
    )

    dm_tests_table = dm_tests(
        dm_tests=pd.read_csv(path_dm_tests),
        eval_config=eval_cfg,
        output_path=visualization_path,
        file_name=f"dm_tests_{i}",
        sample_years="1987 -- 2021",
    )

    r2_comparison_table = r2_comparison(
        r2_metrics_original=pd.read_csv(path_r2_metrics_original),
        r2_metrics=pd.read_csv(path_r2_metrics),
        eval_config=eval_cfg,
        output_path=visualization_path,
        file_name=f"r2_comparison_{i}",
        original_years="1987 -- 2016",
        extended_years="1987 -- 2021",
    )

    portfolio_table = portfolio_metrics(
        portfolio_metrics=pd.read_csv(path_portfolio_metrics),
        eval_config=eval_cfg,
        output_path=visualization_path,
        file_name=f"ml_portfolio_performance_{i}",
        sample_years="1987 -- 2021",
    )

    cumulative_table = cumulative_returns(
        portfolio_returns=pd.read_parquet(path_portfolio_returns),
        eval_config=eval_cfg,
        output_path=visualization_path,
        file_name=f"cumulative_returns_{i}",
        sample_years="1987--2021",
        sp500_returns=pd.read_parquet(path_sp500_returns),
    )

    portfolio_r2_table = portfolio_r2(
        fama_french_portfolio_metrics=pd.read_csv(path_portfolio_presp_metrics),
        sp500_portfolio_metrics=pd.read_csv(path_sp500_metrics),
        eval_config=eval_cfg,
        output_path=visualization_path,
        file_name=f"prespecified_portfolio_r2_{i}",
        sample_years="1987 -- 2021",
    )

