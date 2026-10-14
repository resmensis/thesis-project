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
cache_cfg = CacheConfig()
data_cfg = DataFilesConfig()
run_ctrl_cfg = RunControlConfig()
regime_cfg = DataRegimeConfig(mode="coding")
grid_cfg = HyperGridConfig(use_extended_grids=False)
freq_cfg = CharacteristicsFrequency()
log_cfg = LoggingConfig()
expand_cfg = ExpandingWindowConfig()
model_cfg = ModelSelectionConfig(run_all_models = False,
        models_to_run = ["OLS_3", "OLS_full", "RandomForest"])


complete_dataset_path=f"{cache_cfg.cache_dir}/complete_dataset.parquet"
descriptives_path=cache_cfg.descriptives_dir

feature_panel_path = f"{cache_cfg.cache_dir}/feature_panel_{regime_cfg.mode}.parquet"
feature_cols_path = f"{cache_cfg.cache_dir}/feature_cols_{regime_cfg.mode}.pkl"

predictions_path = f"output/{regime_cfg.mode}/all_predictions.parquet"


predictions = maybe_load_parquet(predictions_path, enabled=True)

evaluation = evaluate_predictions(
    predictions=predictions,
    regime_config=regime_cfg,
    output_dir="output",
    n_quantiles=10,
    dm_loss="squared_error",
    horizon=1,
    save_outputs=True,
)

