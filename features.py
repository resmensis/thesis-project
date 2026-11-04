from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from io_utils import save_parquet, save_pickle
from dataset_builder import summary_stats_extended

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


def plot_feature_correlation_heatmap(
    df: pd.DataFrame,
    feature_cols: list[str],
    output_path: Optional[str] = None,
    output_name: Optional[str] = None,
    max_features: int = 40,
) -> None:
    """Save a correlation heatmap for the numeric feature columns."""
    numeric_cols = [
        c for c in feature_cols
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(numeric_cols) < 2:
        logger.warning(
            "Skipping correlation heatmap: fewer than two numeric feature columns."
        )
        return

    if len(numeric_cols) > max_features:
        logger.warning(
            f"Correlation heatmap limited to first {max_features} numeric features "
            f"of {len(numeric_cols)} total."
        )
        numeric_cols = numeric_cols[:max_features]

    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(max(10, len(numeric_cols) * 0.45),
                                    max(8, len(numeric_cols) * 0.40)))
    sns.heatmap(
        corr,
        annot=False,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.3,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap")
    ax.tick_params(axis="x", rotation=90)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()

    # Save to CSV if output_name provided
    if output_name is not None:
        # Determine output path
        if output_path is None:
            output_dir = Path.cwd()
        else:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{output_name}.png"
        fig.savefig(output_file, dpi=300, bbox_inches="tight")



    plt.close(fig)

    logger.info(f"Saved correlation heatmap to {output_file}")


def build_feature_panel(
    df: pd.DataFrame,
    out_path_panel: str,
    out_path_cols: str,
    descriptives_path: str,
    random_state: int,
    regime_config,
    data_config,
    save_feature_panel: bool,
):
    logger.info(f"Building feature panel, mode={regime_config.mode}, interactions={regime_config.include_macro_interactions}")

    cols_chara = data_config.chara_cols
    cols_macro = data_config.marco_cols 
    sic2_column = data_config.sic2_column


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
            out = out.drop(columns=cols_macro)

        interaction_cols = [c for c in out.columns if "__x__" in c]
        feature_cols = cols_chara + interaction_cols + industry_cols

        if regime_config.ols3_only_mode:
            cols = [
                'permno',
                'date',
                'mv',
                'excess_ret_total',
                'mvel1',
                'mom12m',
                'bm',
            ]

            out = out[cols].copy()
            feature_cols = regime_config.chara_cols_benchmark


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

    plot_feature_correlation_heatmap(
        df=out,
        feature_cols=feature_cols,
        output_path=descriptives_path,
        output_name=f"{regime_config.mode}_charas",
        max_features=94
    )

    """    
    if regime_config.mode == "full" and regime_config.include_macro_interactions:
        plot_feature_correlation_heatmap(
                df=out,
                feature_cols=feature_cols,
                output_path=descriptives_path,
                output_name=f"{regime_config.mode}_all",s
                max_features=940
            )
    """

    logger.info(f"Feature panel built: {out.shape}, {len(feature_cols)} feature columns")
    if regime_config.mode == "coding":
        save_parquet(out, out_path_panel, enabled=save_feature_panel)
        save_pickle(feature_cols, out_path_cols, enabled=save_feature_panel)
    
    return out, feature_cols