from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from io_utils import save_parquet, save_pickle

logger = logging.getLogger("eap_ml.features")


EXCLUDED_BASE = {
    "permno", "date", "sic2", "siccd", "industry_code", "industry_code_grp",
    "ret", "dlret", "ret_total", "rf", "rf_lead", "excess_ret_lead"
}


def infer_characteristic_cols(df: pd.DataFrame):
    return [c for c in df.columns if c not in EXCLUDED_BASE and not c.startswith("ind_") and "__x__" not in c]


def infer_cols_macro(df: pd.DataFrame):
    candidates = []
    for c in df.columns:
        lc = c.lower()
        if c in EXCLUDED_BASE or c.startswith("ind_") or "__x__" in c:
            continue
        if any(k in lc for k in ["dp", "ep", "bm_mkt", "ntis", "tbl", "lty", "tms", "dfy", "svar", "infl", "rf"]):
            candidates.append(c)
    return candidates


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
    return out


def create_limited_macro_interactions(df, char_cols, cols_macro, max_interactions=12):
    logger.debug(f"Creating limited macro interactions: {max_interactions}")
    out = df.copy()
    pairs = []
    for c in char_cols:
        for m in cols_macro:
            pairs.append((c, m))
    for c, m in pairs[:max_interactions]:
        out[f"{c}__x__{m}"] = out[c] * out[m]
    interaction_cols = [f"{c}__x__{m}" for c, m in pairs[:max_interactions]]
    return out, interaction_cols


def _first_available(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def select_coding_features(df: pd.DataFrame, cols_chara, cols_macro, industry_cols, config):
    logger.debug("Selecting coding-mode features")
    required = set()
    for col in [config.ols3_size_col, config.ols3_bm_col, config.ols3_mom_col]:
        if col in df.columns:
            required.add(col)
    monthly_pick = _first_available(df, config.monthly_candidate_cols)
    quarterly_pick = _first_available(df, config.quarterly_candidate_cols)
    annual_pick = _first_available(df, config.annual_candidate_cols)
    for col in [monthly_pick, quarterly_pick, annual_pick]:
        if col is not None:
            required.add(col)
    preferred_extra = ["turnover", "dolvol", "beta", "beta_sq", "retvol", "idiovol", "prof", "inv", "op", "asset_growth"]
    for col in preferred_extra:
        if col in cols_chara:
            required.add(col)
    selected_char_cols = [c for c in cols_chara if c in required]
    selected_cols_macro = cols_macro[:config.coding_keep_macro_count]
    selected_industry_cols = industry_cols[:config.coding_keep_industry_count]
    logger.debug(f"Selected {len(selected_char_cols)} char, {len(selected_cols_macro)} macro, {len(selected_industry_cols)} industry cols")
    return {"char_cols": selected_char_cols, "cols_macro": selected_cols_macro, "industry_cols": selected_industry_cols}


def scale_chars_cross_sectionally_by_month(
    df: pd.DataFrame,
    characteristic_cols: list[str],
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
        df[col] = df.groupby("date")[col].transform(lam)



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


def build_feature_panel(
    df: pd.DataFrame,
    out_path_panel: str,
    out_path_cols: str,
    sic2_column: str,
    cols_chara: list[str],
    cols_macro: list[str],
    random_state: int,
    regime_config,
    save_feature_panel: bool,
    include_macro_interactions=True,
):
    logger.info(f"Building feature panel, mode={regime_config.mode}")

    if regime_config.mode not in ("full", "coding"):
        raise ValueError(
            f"Invalid value for variable: {regime_config.mode}. "
            "Expected 'full' or 'coding'."
        )

    cols_chara = cols_chara
    cols_macro = cols_chara

    out = create_industry_dummies(df, sic2_column)

    industry_cols = [c for c in out.columns if c.startswith("ind_")]
    logger.debug(f"Identified {len(cols_chara)} char, {len(cols_macro)} macro, {len(industry_cols)} industry cols")



    # Step 2: scale the 94 characteristics cross-sectionally month by month to [-1, 1]
    out = scale_chars_cross_sectionally_by_month(
        out,
        char_cols=cols_chara,
        date_col="date",
        n_quantiles=1000,
        random_state=random_state,
    )

    if regime_config.mode == "coding":
        sel = select_coding_features(
            out,
            cols_chara=cols_chara,
            cols_macro=cols_macro,
            industry_cols=industry_cols,
            config=regime_config,
        )
        char_cols = sel["char_cols"]
        cols_macro = sel["cols_macro"]
        industry_cols = sel["industry_cols"]
        interaction_cols = []

        if regime_config.coding_include_interactions and len(cols_macro) > 0:
            out, interaction_cols = create_limited_macro_interactions(
                out,
                char_cols=char_cols,
                cols_macro=cols_macro,
                max_interactions=regime_config.coding_max_interactions,
            )

        feature_cols = char_cols + industry_cols + interaction_cols  # Macro removed

    else:
        if include_macro_interactions:
            out = create_macro_interactions(out, cols_chara, cols_macro)

        interaction_cols = [c for c in out.columns if "__x__" in c]
        feature_cols = cols_chara + industry_cols + interaction_cols  # Macro removed

    logger.info(f"Feature panel built: {out.shape}, {len(feature_cols)} feature columns")
    save_parquet(out, out_path_panel, enabled=save_feature_panel)
    save_pickle(feature_cols, out_path_cols, enabled=save_feature_panel)
    return out, feature_cols