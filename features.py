from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

logger = logging.getLogger("eap_ml.features")


EXCLUDED_BASE = {
    "permno", "date", "sic2", "siccd", "industry_code", "industry_code_grp",
    "ret", "dlret", "ret_total", "rf", "rf_lead", "excess_ret_lead"
}


def infer_characteristic_cols(df: pd.DataFrame):
    return [c for c in df.columns if c not in EXCLUDED_BASE and not c.startswith("ind_") and "__x__" not in c]


def infer_macro_cols(df: pd.DataFrame):
    candidates = []
    for c in df.columns:
        lc = c.lower()
        if c in EXCLUDED_BASE or c.startswith("ind_") or "__x__" in c:
            continue
        if any(k in lc for k in ["dp", "ep", "bm_mkt", "ntis", "tbl", "lty", "tms", "dfy", "svar", "infl", "rf"]):
            candidates.append(c)
    return candidates


def create_industry_dummies(df: pd.DataFrame, max_dummies: int = 74):
    logger.debug(f"Creating industry dummies, max {max_dummies}")
    out = df.copy()
    counts = out["industry_code"].value_counts()
    keep = counts.index[:max_dummies].tolist()
    out["industry_code_grp"] = np.where(out["industry_code"].isin(keep), out["industry_code"], "OTHER")
    dummies = pd.get_dummies(out["industry_code_grp"], prefix="ind", dtype=float)
    logger.debug(f"Created {len(dummies.columns)} industry dummies")
    return pd.concat([out, dummies], axis=1)


def create_macro_interactions(df: pd.DataFrame, char_cols, macro_cols):
    logger.debug(f"Creating macro interactions: {len(char_cols)} chars x {len(macro_cols)} macros")
    out = df.copy()
    for z in char_cols:
        for m in macro_cols:
            out[f"{z}__x__{m}"] = out[z] * out[m]
    return out


def create_limited_macro_interactions(df, char_cols, macro_cols, max_interactions=12):
    logger.debug(f"Creating limited macro interactions: {max_interactions}")
    out = df.copy()
    pairs = []
    for c in char_cols:
        for m in macro_cols:
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


def select_coding_features(df: pd.DataFrame, all_char_cols, macro_cols, industry_cols, config):
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
        if col in all_char_cols:
            required.add(col)
    selected_char_cols = [c for c in all_char_cols if c in required]
    selected_macro_cols = macro_cols[:config.coding_keep_macro_count]
    selected_industry_cols = industry_cols[:config.coding_keep_industry_count]
    logger.debug(f"Selected {len(selected_char_cols)} char, {len(selected_macro_cols)} macro, {len(selected_industry_cols)} industry cols")
    return {"char_cols": selected_char_cols, "macro_cols": selected_macro_cols, "industry_cols": selected_industry_cols}


def winsorize_by_month(df: pd.DataFrame, cols, lower=0.01, upper=0.99):
    logger.debug(f"Winsorizing {len(cols)} columns at [{lower}, {upper}]")
    out = df.copy()
    def _clip(g):
        for c in cols:
            lo = g[c].quantile(lower)
            hi = g[c].quantile(upper)
            g[c] = g[c].clip(lo, hi)
        return g
    return out.groupby("date", group_keys=False).apply(_clip)


def scale_chars_cross_sectionally_by_month(
    df: pd.DataFrame,
    char_cols,
    date_col: str = "date",
    n_quantiles: int = 1000,
    random_state: int = 42,
):
    """
    Scale the 94 stock characteristics cross-sectionally month by month.

    For each month and each characteristic:
    1. Fit a QuantileTransformer on the non-missing cross section.
    2. Transform to Uniform[0,1].
    3. Map to [-1,1] by x -> 2*x - 1.

    Missing values remain missing.
    """
    logger.debug(f"Scaling {len(char_cols)} characteristics cross-sectionally")
    out = df.copy()

    for dt, idx in out.groupby(date_col).groups.items():
        month_idx = list(idx)

        for col in char_cols:
            s = out.loc[month_idx, col]
            mask = s.notna()

            if mask.sum() <= 1:
                continue

            x = s.loc[mask].to_numpy().reshape(-1, 1)
            nq = min(n_quantiles, len(x))

            qt = QuantileTransformer(
                n_quantiles=nq,
                output_distribution="uniform",
                random_state=random_state,
                subsample=10_000_000,
            )
            u = qt.fit_transform(x).ravel()
            out.loc[s.loc[mask].index, col] = 2.0 * u - 1.0

    return out


def build_feature_panel(df: pd.DataFrame, regime_config, include_macro_interactions=True):
    logger.info(f"Building feature panel, mode={regime_config.mode}")
    out = create_industry_dummies(df, max_dummies=74)

    all_char_cols = infer_characteristic_cols(out)
    macro_cols = infer_macro_cols(out)
    industry_cols = [c for c in out.columns if c.startswith("ind_")]
    logger.debug(f"Identified {len(all_char_cols)} char, {len(macro_cols)} macro, {len(industry_cols)} industry cols")

    # Step 1: winsorize raw characteristics by month
    out = winsorize_by_month(out, all_char_cols, lower=0.01, upper=0.99)

    # Step 2: scale the 94 characteristics cross-sectionally month by month to [-1, 1]
    out = scale_chars_cross_sectionally_by_month(
        out,
        char_cols=all_char_cols,
        date_col="date",
        n_quantiles=1000,
        random_state=42,
    )

    if regime_config.mode == "coding":
        sel = select_coding_features(
            out,
            all_char_cols=all_char_cols,
            macro_cols=macro_cols,
            industry_cols=industry_cols,
            config=regime_config,
        )
        char_cols = sel["char_cols"]
        macro_cols = sel["macro_cols"]
        industry_cols = sel["industry_cols"]
        interaction_cols = []

        if regime_config.coding_include_interactions and len(macro_cols) > 0:
            out, interaction_cols = create_limited_macro_interactions(
                out,
                char_cols=char_cols,
                macro_cols=macro_cols,
                max_interactions=regime_config.coding_max_interactions,
            )

        feature_cols = char_cols + macro_cols + industry_cols + interaction_cols

    else:
        if include_macro_interactions:
            out = create_macro_interactions(out, all_char_cols, macro_cols)

        interaction_cols = [c for c in out.columns if "__x__" in c]
        feature_cols = all_char_cols + macro_cols + industry_cols + interaction_cols

    logger.info(f"Feature panel built: {out.shape}, {len(feature_cols)} feature columns")
    return out, feature_cols