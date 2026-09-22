from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from io_utils import save_parquet, save_pickle
from dataset_builder import summary_stats_extended
from transformers import GroupedQuantileTransformer

logger = logging.getLogger("eap_ml.features")


def create_industry_dummies(df: pd.DataFrame, sic2_column: str):
    logger.debug(f"Creating industry dummies.")
    out = df.copy()
    dummies = pd.get_dummies(out[sic2_column], prefix="ind", dtype=float)
    logger.debug(f"Created {len(dummies.columns)} industry dummies")
    return pd.concat([out, dummies], axis=1)


def create_macro_interactions(df: pd.DataFrame, char_cols, cols_macro):
    logger.debug(f"Creating macro interactions: {len(char_cols)} chars x {len(cols_macro)} macros")
    out = df.copy()
    for z in char_cols:
        for m in cols_macro:
            out[f"{z}__x__{m}"] = out[z] * out[m]
        out = out.copy()
    return out


def sample_permno_coding(
    df: pd.DataFrame,
    n_permnos: int = 500,
    permno_col: str = "permno",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Randomly select unique PERMNOs and retain all observations for them.
    """
    
    if permno_col not in df.columns:
        raise KeyError(f"Column '{permno_col}' is not in the DataFrame.")

    if n_permnos <= 0:
        raise ValueError(f"n_permnos must be positive; received {n_permnos}.")

    available_permnos = df[permno_col].dropna().unique()
    n_available = len(available_permnos)

    if n_permnos > n_available:
        raise ValueError(
            f"Requested {n_permnos} PERMNOs, but only {n_available} "
            f"non-missing unique PERMNOs are available."
        )

    rng = np.random.default_rng(random_state)

    selected_permnos = rng.choice(
        available_permnos,
        size=n_permnos,
        replace=False,
    )

    return df.loc[df[permno_col].isin(selected_permnos)].copy()


def build_feature_panel(
    df: pd.DataFrame,
    out_path_panel: str,
    out_path_cols: str,
    descriptives_path: str,
    cols_chara: list[str],
    cols_macro: list[str],
    sic2_column: str,
    n_quantiles: int,
    random_state: int,
    regime_config,
    save_feature_panel: bool,
):
    logger.info(f"Building feature panel, mode={regime_config.mode}, interactions={regime_config.include_macro_interactions}")


    # ------------------------------------------------------------------
    # 1. Create industry dummies 
    # ------------------------------------------------------------------
    out = create_industry_dummies(df, sic2_column)

    industry_cols = [c for c in out.columns if c.startswith("ind_")]
    logger.debug(f"Identified {len(cols_chara)} char, {len(cols_macro)} macro, {len(industry_cols)} industry dummies")


    # ------------------------------------------------------------------
    # 2. Interactions: Chara x Macro
    # ------------------------------------------------------------------
    if regime_config.mode == "full":
        if regime_config.include_macro_interactions:
            out = create_macro_interactions(out, cols_chara, cols_macro)

        interaction_cols = [c for c in out.columns if "__x__" in c]
        feature_cols = cols_chara + interaction_cols + industry_cols

        summary_stats_extended(
        out,
        date_col="date",
        output_path=descriptives_path,
        output_name="summary_feature_full",
        )

    
    # ------------------------------------------------------------------
    # 3. Sample reduction for coding version
    # ------------------------------------------------------------------
    if regime_config.mode == "coding":
        out = sample_permno_coding(
            out,
            n_permnos=regime_config.coding_max_stocks,
            permno_col="permno",
            random_state=random_state
        )
        
        char_cols_coding = regime_config.chara_cols_coding
        feature_cols = char_cols_coding

        summary_stats_extended(
        out,
        date_col="date",
        output_path=descriptives_path,
        output_name="summary_feature_coding",
        )


    logger.info(f"Feature panel built: {out.shape}, {len(feature_cols)} feature columns")
    save_parquet(out, out_path_panel, enabled=save_feature_panel)
    save_pickle(feature_cols, out_path_cols, enabled=save_feature_panel)
    return out, feature_cols