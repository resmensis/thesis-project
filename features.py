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


def scale_chars_cross_sectionally_by_month(
    df: pd.DataFrame,
    characteristic_cols: list[str],
    date_col: str = "date",
    n_quantiles: int = 1000,
    random_state: int = 42,
):
    """
    Scale characteristics cross-sectionally month by month usinf the QuantileTransformer into [-1,1].
    """

    logger.debug(f"Scaling {len(characteristic_cols)} characteristics cross-sectionally")
    out = df.copy()

    qt = QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution="uniform",
        random_state=random_state,
        subsample=10000,
    )

    for col in characteristic_cols:
        lam = lambda x: qt.fit_transform(x.values.reshape(-1, 1)).ravel()
        out[col] = out.groupby(date_col)[col].transform(lam)
        
    return out


def scale_chars_cross_sectionally_by_month_bigdata_v2(
    df: pd.DataFrame,
    characteristic_cols: list[str],
    n_quantiles: int = 1000,
    random_state: int = 42,
    subsample: int = 10000,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Scale specified columns cross-sectionally within each calendar month (year, month),
    using a separate QuantileTransformer per (month, column) to keep memory bounded.

    Assumes df has a datetime-like 'date' column by default; can be overridden with date_col.
    Returns a new DataFrame with transformed columns.
    """
    if date_col not in df.columns:
        raise KeyError(f"DataFrame must contain a '{date_col}' column.")

    out = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(out[date_col]):
        out[date_col] = pd.to_datetime(out[date_col])

    # Ensure we have month-key for grouping (YYYYMM)
    out["_ym"] = out[date_col].dt.to_period("M")

    # Cache transformers per (ym, col) if you want consistent scaling across months;
    # here we keep a per-group transformer to guarantee independence across months.
    transformers = {}

    for col in characteristic_cols:
        transformed = np.full(len(out), np.nan, dtype=float)

        for ym, grp in out.groupby("_ym", sort=False):
            idx = grp.index
            X = grp[[col]].values  # shape (n_in_group, 1)
            if X.size == 0 or np.all(pd.isna(X)):
                continue

            key = (str(ym), col)
            # per-month, per-column transformer
            if key not in transformers:
                n_q = max(2, min(n_quantiles, int(X.shape[0])))
                transformers[key] = QuantileTransformer(
                    n_quantiles=n_q,
                    output_distribution="uniform",
                    random_state=random_state,
                    subsample=subsample,
                )
            qt = transformers[key]

            X_tr = qt.fit_transform(X)
            transformed[idx] = X_tr[:, 0]

        # map to [-1, 1] if desired: uncomment below
        transformed = 2.0 * transformed - 1.0

        out[col] = transformed

    # drop helper column
    out.drop(columns=["_ym"], inplace=True)

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
    logger.info(f"Building feature panel, mode={regime_config.mode}")


    # ------------------------------------------------------------------
    # 1. Create industry dummies 
    # ------------------------------------------------------------------
    out = create_industry_dummies(df, sic2_column)

    industry_cols = [c for c in out.columns if c.startswith("ind_")]
    logger.debug(f"Identified {len(cols_chara)} char, {len(cols_macro)} macro, {len(industry_cols)} industry dummies")


    # ------------------------------------------------------------------
    # 2. Scale characteristics cross-sectionally month by month to [-1, 1]
    # ------------------------------------------------------------------
    out = scale_chars_cross_sectionally_by_month(
        out,
        characteristic_cols=cols_chara,
        date_col="date",
        n_quantiles=n_quantiles,        
        random_state=random_state,
    )

    """
    scalar = GroupedQuantileTransformer(
        group_col="date",
        value_cols=cols_chara,
        output_range=(-1, 1),
        random_state=random_state,
    )

    merged = scalar.fit_transform(merged)
    merged = pd.DataFrame(merged)

    summary_stats_extended(
        merged,
        date_col="date",
        output_path=descriptives_path,
        output_name="summary_scaled",
    )
    """

    summary_stats_extended(
        out,
        date_col="date",
        output_path=descriptives_path,
        output_name="summary_timeframe_adjusted",
    )


    # ------------------------------------------------------------------
    # 3. Interactions: Chara x Macro
    # ------------------------------------------------------------------
    if regime_config.mode == "full":
        if regime_config.include_macro_interactions:
            out = create_macro_interactions(out, cols_chara, cols_macro)

        interaction_cols = [c for c in out.columns if "__x__" in c]
        feature_cols = cols_chara + interaction_cols + industry_cols


    # ------------------------------------------------------------------
    # 4. Sample reduction for coding version
    # ------------------------------------------------------------------
    else:
        out = sample_permno_coding(
            out,
            n_permnos=regime_config.coding_max_stocks,
            permno_col="permno",
            random_state=random_state
        )
        
        char_cols_coding = regime_config.chara_cols_coding
        feature_cols = char_cols_coding



    logger.info(f"Feature panel built: {out.shape}, {len(feature_cols)} feature columns")
    save_parquet(out, out_path_panel, enabled=save_feature_panel)
    save_pickle(feature_cols, out_path_cols, enabled=save_feature_panel)
    return out, feature_cols