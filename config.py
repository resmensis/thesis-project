from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ReproducibilityConfig:
    random_state: int = 42
    torch_deterministic: bool = True
    scalar_n_quantiles: int = 1000


@dataclass
class RunControlConfig:
    dataset_creation_only: bool = False  # If True, stop after building complete_dataset
    feature_panel_only: bool = False  # If True, stop after building feature_panel
    load_predictions: bool = False


@dataclass
class CacheConfig:
    enabled: bool = True
    force_rebuild_dataset: bool = False  # If True, rebuild complete_dataset even if cached
    force_rebuild_feature_set: bool = False  # If True, rebuild build_feature_panel even if cached
    force_refit_models: bool = False  # If True, rerun predictions even if cached
    save_complete_dataset: bool = True
    save_feature_panel: bool = True
    save_predictions: bool = True
    save_metrics: bool = True
    cache_dir: str = "cache"
    descriptives_dir: str = "descriptives"


@dataclass
class DataFilesConfig:
    datashare_path: str = "C:/Coding/Project/data/datashare.csv"
    crsp_monthly_path: str = "C:/Coding/Project/data/crsp_monthly_new.csv"
    macro_path: str = "C:/Coding/Project/data/Data2024_monthly_goyal.csv"
    sic2_column: str = "sic2"  # Column name for industry codes in datashare.csv
    possible_crsp_cols: List[str] = field(default_factory=lambda: [
        "permno",
        "date", "yyyymm",
        "ret", "dlret", "prc", "shrout"
    ])
    possible_marco_cols: List[str] = field(default_factory=lambda: [
        "date", "yyyymm", 
        "d/p", "e/p", "b/m", "ntis", "tbl", "tms", "dfy", "svar",
        "rfree", "rf"
    ])
    chara_cols: List[str] = field(default_factory=lambda: [
        'baspread', 'beta', 'betasq', 'chmom', 'dolvol', 
        'idiovol', 'ill', 'indmom', 'maxret', 'mom12m',
        'mom1m', 'mom36m', 'mom6m', 'mvel1', 'pricedelay', 
        'retvol', 'std_dolvol', 'std_turn', 'turn', 'zerotrade',
        'aeavol', 'cash', 'chtx', 'cinvest', 'ear', 
        'ms', 'nincr', 'roaq', 'roavol', 'roeq', 
        'rsup', 'stdacc', 'stdcf',
        'absacc', 'acc', 'age', 'agr', 'bm', 'bm_ia', 
        'cashdebt', 'cashpr', 'cfp', 'cfp_ia', 'chatoia', 'chcsho', 'chempia', 'chinv', 
        'chpmia', 'convind', 'currat', 'depr', 'divi', 'divo', 
        'dy', 'egr', 'ep', 'gma', 'grcapx', 'grltnoa', 'herf', 'hire', 
        'invest', 'lev', 'lgr', 'mve_ia', 'operprof', 'orgcap', 'pchcapx_ia', 'pchcurrat', 'pchdepr',
        'pchgm_pchsale', 'pchquick', 'pchsale_pchinvt', 'pchsale_pchrect', 
        'pchsale_pchxsga', 'pchsaleinv', 'pctacc', 'ps', 'quick', 'rd', 'rd_mve', 'rd_sale', 'realestate', 
        'roic', 'salecash', 'saleinv', 'salerec', 'secured', 'securedind', 'sgr', 'sin', 'sp', 
        'tang', 'tb'
    ])
    marco_cols: List[str] = field(default_factory=lambda: [
        "d/p", "e/p", "b/m", "ntis", "tbl", "tms", "dfy", "svar"
    ])


@dataclass
class DataRegimeConfig:
    mode: str = "full"  # 'full' or 'coding'
    include_macro_interactions: bool = True
    ols3_only_mode: bool = False
    ols3_size_col: str = "mvel1"
    ols3_bm_col: str = "bm"
    ols3_mom_col: str = "mom12m"
    chara_cols_benchmark: List[str] = field(default_factory=lambda: [
        'mvel1', 'mom12m', 'bm', 
    ])
    coding_max_stocks: int = 100
    chara_cols_coding: List[str] = field(default_factory=lambda: [
        'mvel1', 'mom12m', 'bm', 
        'mom1m', 'mom6m', 'dy', 'cash', 'stdacc', 'chmom', 'ep'
    ])
    top_bottom: int = 1000
    top_bottom_coding: int = 25
    test_end_year_original: int = 2016


@dataclass
class HyperGridConfig:
    use_extended_grids: bool = False

    huber_base: Dict = field(default_factory=lambda: {
        "epsilon": [2.576, 3.291],
        "alpha": [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2],
        "max_iter": [500],
    })

    huber_extended: Dict = field(default_factory=lambda: {
        "epsilon": [1.35, 1.5, 2.0, 2.576, 3.291],
        "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2],
        "max_iter": [500, 1000],
    })

    pcr_base: Dict = field(default_factory=lambda: {
        "pca__n_components": [3, 5, 10],
    }) # , 20, 30, 50, 70, 90, 100

    pcr_extended: Dict = field(default_factory=lambda: {
        "pca__n_components": [3, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 90, 100],
    })

    pls_base: Dict = field(default_factory=lambda: {
        "n_components": [1, 2, 3, 4, 6, 8, 10],
    })

    pls_extended: Dict = field(default_factory=lambda: {
        "n_components": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    })

    rf_base: Dict = field(default_factory=lambda: {
        "n_estimators": [300],
        "max_depth": [1, 2, 3, 4, 5, 6],
        "max_features": [3, 5, 10, 20, 30, 50],
    })

    rf_extended: Dict = field(default_factory=lambda: {
        "n_estimators": [200, 300, 400],
        "max_depth": [1, 2, 3, 4, 5, 6],
        "max_features": ["sqrt", "log2", 3, 5, 10, 20, 30, 50],
    })

    gbrt_base: Dict = field(default_factory=lambda: {
        "n_estimators": [25, 50, 100, 200, 500, 1000],
        "learning_rate": [0.01, 0.1],
        "max_depth": [1, 2],
    })

    gbrt_extended: Dict = field(default_factory=lambda: {
        "n_estimators": [25, 50, 100, 200, 500, 1000],
        "learning_rate": [0.005, 0.01, 0.05, 0.1],
        "max_depth": [1, 2, 3, 4],
    })

    mlp_architectures: Dict = field(default_factory=lambda: {
        "NN1": (32,),
        "NN2": (32, 16),
        "NN3": (32, 16, 8),
        "NN4": (32, 16, 8, 4),
        "NN5": (32, 16, 8, 4, 2),
    })

    mlp_base: Dict = field(default_factory=lambda: {
        "activation": ["relu"],
        "alpha": [1e-5, 1e-4, 1e-3],
        "learning_rate_init": [0.001, 0.01],
        "max_iter": [300],
    })

    mlp_extended: Dict = field(default_factory=lambda: {
        "activation": ["relu"],
        "alpha": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        "learning_rate_init": [1e-2, 1e-3, 5e-4, 1e-4],
        "max_iter": [400],
    })

    lstm_base: Dict = field(default_factory=lambda: {
        "seq_len": [6, 12],
        "hidden_dim": [32, 64],
        "num_layers": [1, 2],
        "dropout": [0.0, 0.1],
        "lr": [1e-3, 5e-4],
        "batch_size": 1024,
        "epochs": 20,
    })
    lstm_extended: Dict = field(default_factory=lambda: {
        "seq_len": [3, 6, 12, 24],
        "hidden_dim": [32, 64, 128],
        "num_layers": [1, 2, 3],
        "dropout": [0.0, 0.1, 0.2, 0.3],
        "lr": [1e-2, 1e-3, 5e-4, 1e-4],
        "batch_size": 1024,
        "epochs": 30,
    })


@dataclass
class CharacteristicsFrequency:
    cols_vars_monthly: List[str] = field(default_factory=lambda: [
        'baspread', 'beta', 'betasq', 
        'chmom', 
        'dolvol', 
        'idiovol', 'ill', 'indmom', 
        'maxret', 'mom12m', 'mom1m', 'mom36m', 'mom6m', 
        'mvel1', 
        'pricedelay', 
        'retvol', 
        'std_dolvol', 'std_turn', 
        'turn', 'zerotrade'
    ])

    cols_vars_quarterly: List[str] = field(default_factory=lambda: [
        'aeavol', 
        'cash', 
        'chtx', 'cinvest', 
        'ear', 
        'ms', 
        'nincr', 
        'roaq', 'roavol', 'roeq', 
        'rsup', 
        'stdacc', 'stdcf'
    ])

    cols_vars_annual: List[str] = field(default_factory=lambda: [
        'absacc', 'acc', 
        'age', 'agr', 
        'bm', 'bm_ia', 
        'cashdebt', 'cashpr', 'cfp', 'cfp_ia', 'chatoia', 'chcsho', 'chempia', 'chinv', 
        'chpmia', 
        'convind', 'currat', 'depr', 'divi', 'divo', 
        'dy', 
        'egr', 'ep', 'gma', 'grcapx', 'grltnoa', 'herf', 'hire', 
        'invest', 'lev', 'lgr', 
        'mve_ia', 
        'operprof', 'orgcap', 'pchcapx_ia', 'pchcurrat', 'pchdepr', 'pchgm_pchsale', 'pchquick', 'pchsale_pchinvt', 'pchsale_pchrect', 
        'pchsale_pchxsga', 'pchsaleinv', 'pctacc', 
        'ps', 'quick', 'rd', 'rd_mve', 'rd_sale', 'realestate', 
        'roic',
        'salecash', 'saleinv', 'salerec', 'secured', 'securedind', 'sgr', 'sin', 'sp', 
        'tang', 'tb'
    ])


@dataclass
class LoggingConfig:
    enabled: bool = True
    log_dir: str = "logs"
    file_level_full: str = "INFO"
    file_level_coding: str = "DEBUG"
    console_level_full: str = "INFO"
    console_level_coding: str = "DEBUG"
    logger_name: str = "eap_ml"
    overwrite_log: bool = False


@dataclass
class ExpandingWindowConfig:
    """Configuration for expanding window forecasting with annual refitting."""
    train_start_year: int = 1957
    initial_train_end_year: int = 1974
    val_start_year: int = 1975
    val_end_year: int = 1986
    test_start_year: int = 1987
    test_end_year_original: int = 2016
    test_end_year_extended: int = 2021
    refit_frequency_years: int = 1


@dataclass
class ModelSelectionConfig:
    """
    Configuration for which models to run in the expanding window.
    
    Gu et al. (2020) model specifications:
    - OLS_3: HuberRegressor with 3 factors (size, value, momentum) - benchmark model
    - OLS_full: HuberRegressor with all features with tuned hyperparameters (epsilon, alpha)
    - PCR: Principal Component Regression (PCA + HuberRegressor)
    - PLS: Partial Least Squares regression
    - GBRT: Gradient Boosted Regression Trees
    - RandomForest: Random Forest
    - MLP: Multi-Layer Perceptron (feedforward neural network)
    - LSTM: Long Short-Term Memory (recurrent neural network, only when run_all_models=True)
    
    Default Setting:
    - run_all_models = True (default): Runs ALL models including LSTM
    - models_to_run = None (default): Ignored when run_all_models=True
    
    Usage Examples:
    
    1. Run all models (default - includes LSTM):
       ```python
       run_all_models = True
       models_to_run = None  # or any value, will be ignored
       ```
       Models run: OLS_3, OLS_full, PCR, PLS, GBRT, RandomForest, MLP, LSTM
    
    2. Run only OLS-3 benchmark (fastest, for debugging):
       ```python
       run_all_models = False
       models_to_run = ["OLS_3"]
       ```
       Models run: OLS_3 only
    
    3. Run custom subset (without LSTM):
       ```python
       run_all_models = False
       models_to_run = ["OLS_3", "GBRT", "MLP"]
       ```
       Models run: OLS_3, GBRT, MLP
    
    4. Run all models except LSTM (faster than full ensemble):
       ```python
       run_all_models = False
       models_to_run = ["OLS_3", "OLS_full", "PCR", "PLS", "GBRT", "RandomForest", "MLP"]
       ```
    
    Notes:
    - LSTM is computationally expensive and only runs when run_all_models=True
    - OLS_3 is the Gu et al. (2020) benchmark model and should always be included
    - All "OLS" models use HuberRegressor (robust regression) as per Gu et al. (2020)
    """
    run_all_models: bool = False  # If True, run all models (including LSTM) - DEFAULT
    models_to_run: Optional[List[str]] = field(default=None)  # Custom model selection (ignored if run_all_models=True)
    # models_to_run: List[str] = field(default_factory=lambda: ["OLS_3"])