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
)

from io_utils import ensure_dir, save_parquet, set_global_seed, setup_project_logger
from dataset_builder import build_complete_dataset, reduce_observations_for_coding
from sample_splits import subset_timeframe, chronological_split
from features import build_feature_panel
from models_linear import fit_pooled_ols, tune_huber_regression, tune_pcr, tune_pls
from models_trees import tune_gbrt, tune_random_forest
from models_mlp import tune_mlp_models


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
        cols_vars_annual=freq_cfg.cols_vars_annual
    )

    logger.info(f"Complete dataset shape: {complete.shape}")

    if run_ctrl_cfg.dataset_creation_only:
            print("Dataset creation only mode is ON.")
            print(f"Complete dataset created and cached at: {cache_cfg.cache_dir}/complete_dataset.parquet")
            return

    if regime_cfg.mode == "coding":
        logger.info("Applying coding-mode sampling reduction.")
        complete = reduce_observations_for_coding(complete, max_stocks_per_month=regime_cfg.coding_max_stocks_per_month, random_state=repro_cfg.random_state)

    feature_panel, feature_cols = build_feature_panel(complete, regime_config=regime_cfg, include_macro_interactions=True)
    if cache_cfg.save_feature_panel:
        save_parquet(feature_panel, f"{cache_cfg.cache_dir}/feature_panel_{regime_cfg.mode}.parquet", enabled=True)
    logger.info(f"Feature panel shape: {feature_panel.shape}")
    logger.info(f"Number of feature columns: {len(feature_cols)}")

    windows = {}
    if tf_cfg.run_original_window:
        windows["original"] = tf_cfg.original_end_date
    if tf_cfg.run_extended_window:
        windows["extended_2021"] = tf_cfg.extended_end_date

    for name, end_date in windows.items():
        df_win = subset_timeframe(feature_panel, end_date)
        splits = chronological_split(df_win, split_cfg.train_end, split_cfg.val_end, end_date)

        logger.info(f"Running window: {name} up to {end_date}")        

        X_train = splits["train"][feature_cols].fillna(0.0)
        y_train = splits["train"]["excess_ret_lead"]
        X_val = splits["val"][feature_cols].fillna(0.0)
        y_val = splits["val"]["excess_ret_lead"]
        X_test = splits["test"][feature_cols].fillna(0.0)
        y_test = splits["test"]["excess_ret_lead"]

        logger.info(
            f"Split sizes | train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
        )        

        meta_test = splits["test"][["permno", "date"]].copy()
        if "me" in splits["test"].columns:
            meta_test["me"] = splits["test"]["me"]
        else:
            meta_test["me"] = 1.0

        X_train_s = X_train.copy()
        X_val_s = X_val.copy()
        X_test_s = X_test.copy()

        benchmark_features = [c for c in [regime_cfg.ols3_size_col, regime_cfg.ols3_bm_col, regime_cfg.ols3_mom_col] if c in X_train_s.columns]

        models = {}
        if len(benchmark_features) == 3:
            models["OLS_3"] = fit_pooled_ols(X_train_s[benchmark_features], y_train)
        models["OLS_full"] = fit_pooled_ols(X_train_s, y_train)
        models["Huber"] = tune_huber_regression(X_train_s, y_train, X_val_s, y_val)["model"]
        models["PCR"] = tune_pcr(X_train_s, y_train, X_val_s, y_val)["model"]
        models["PLS"] = tune_pls(X_train_s, y_train, X_val_s, y_val)["model"]
        rf_grid = grid_cfg.rf_extended if grid_cfg.use_extended_grids else grid_cfg.rf_base
        gbrt_grid = grid_cfg.gbrt_extended if grid_cfg.use_extended_grids else grid_cfg.gbrt_base
        mlp_grid = grid_cfg.mlp_extended if grid_cfg.use_extended_grids else grid_cfg.mlp_base
        models["GBRT"] = tune_gbrt(X_train, y_train, X_val, y_val, random_state=repro_cfg.random_state, **gbrt_grid)["model"]
        models["RandomForest"] = tune_random_forest(X_train, y_train, X_val, y_val, random_state=repro_cfg.random_state, **rf_grid)["model"]
        models["MLP"] = tune_mlp_models(X_train_s, y_train, X_val_s, y_val, random_state=repro_cfg.random_state, **mlp_grid)["model"]

        outdir = Path("output") / name / regime_cfg.mode
        ensure_dir(outdir)
        pd.DataFrame({"feature": feature_cols}).to_csv(outdir / "feature_cols.csv", index=False)
        pd.DataFrame({"model": list(models.keys())}).to_csv(outdir / "models_trained.csv", index=False)
        meta_test.to_csv(outdir / "meta_test.csv", index=False)
        pd.DataFrame({"y_test": y_test}).to_csv(outdir / "y_test.csv", index=False)


if __name__ == "__main__":
    main()
