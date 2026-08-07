from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_inputs import load_datashare, load_crsp_monthly, load_macro_monthly
from io_utils import maybe_load_parquet, save_parquet

logger = logging.getLogger("eap_ml.dataset_builder")


def get_datashare_characteristic_cols(ds: pd.DataFrame) -> list[str]:
    """
    Return the 94 characteristic columns from datashare.csv.

    Assumptions on datashare.csv:
    columns 3-96: 94 characteristics. 
    column 1: permno
    column 2: date
    column 97: sic2 (to create industry dummies)

    Since pandas is 0-based, that corresponds to positions 2:95.
    """
    return ds.columns[2:95].tolist()


def compute_missingness_for_characteristics(
    df: pd.DataFrame,
    characteristic_cols: list[str],
) -> pd.DataFrame:
    """
    Compute percentage of missing values per characteristic.
    """
    missing_pct = df[characteristic_cols].isna().mean().mul(100.0)

    out = (
        missing_pct
        .rename("missing_pct")
        .reset_index()
        .rename(columns={"index": "characteristic"})
        .sort_values("missing_pct", ascending=False)
        .reset_index(drop=True)
    )
    return out


def save_missingness_histogram(
    missing_df: pd.DataFrame,
    out_jpg_path: str,
    title: str,
    bins: int = 20,
):
    """
    Save a histogram of percentage missing data per characteristic.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(missing_df["missing_pct"], bins=bins, edgecolor="black")
    plt.xlabel("Missing data percentage")
    plt.ylabel("Number of characteristics")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_jpg_path, dpi=200, bbox_inches="tight")
    plt.close()


def impute_characteristics_by_month_cross_sectional_median(
    df: pd.DataFrame,
    characteristic_cols: list[str],
) -> pd.DataFrame:
    """
    Impute missing characteristics using the cross-sectional median within each month.

    This follows the Gu, Kelly, and Xiu (2020) approach conceptually:
    missing characteristics are replaced with cross-sectional medians.
    """
    out = df.copy()

    for col in characteristic_cols:
        out[col] = out[col].fillna(out.groupby("date")[col].transform("median"))

    return out


def build_complete_dataset(
    datashare_path: str,
    crsp_path: str,
    macro_path: str,
    out_path: str,
    possible_crsp_cols: list[str],
    possible_marco_cols: list[str],
    cols_vars_monthly: list[str],
    cols_vars_quarterly: list[str],
    cols_vars_annual: list [str],
    cache_enabled: bool = True,
    force_rebuild: bool = False,    
):
    logger.info(f"Building complete dataset, output: {out_path}")
    
    if cache_enabled and not force_rebuild:
        cached = maybe_load_parquet(out_path, enabled=True)
        if cached is not None:
            logger.info(f"Loading cached complete dataset from {out_path}")
            return cached

    logger.debug("Loading source datasets")
    ds = load_datashare(datashare_path)
    crsp = load_crsp_monthly(crsp_path, possible_crsp_cols)
    macro = load_macro_monthly(macro_path, possible_marco_cols)

    # Columns 3-96 in datashare.csv = 94 characteristics.
    characteristic_cols = get_datashare_characteristic_cols(ds)
    logger.debug(f"Identified {len(characteristic_cols)} characteristic columns")

    logger.debug("Merging datashare with CRSP")
    merged = ds.merge(
        crsp[["permno", "date", "ret", "dlret", "ret_total"]],
        on=["permno", "date"],
        how="inner",
        validate="one_to_one",
    )

    logger.debug("Merging with macro data")
    merged = merged.merge(macro, on="date", how="left")
    merged["industry_code"] = merged["sic2"].astype("string").fillna("UNK")
    merged = merged.sort_values(["permno", "date"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 1. Missingness before imputation
    # ------------------------------------------------------------------
    missing_before = compute_missingness_for_characteristics(merged, characteristic_cols)

    before_csv = out_path.replace(".parquet", "_missingness_before.csv")
    before_jpg = out_path.replace(".parquet", "_missingness_before.jpg")

    missing_before.to_csv(before_csv, index=False)
    save_missingness_histogram(
        missing_before,
        before_jpg,
        title="Missing data percentage per characteristic (before imputation)",
        bins=20,
    )
    logger.debug(f"Saved missingness before imputation: {before_csv}, {before_jpg}")

    # ------------------------------------------------------------------
    # 2. Impute missing characteristics using monthly cross-sectional medians
    # ------------------------------------------------------------------
    logger.info("Imputing missing characteristics using monthly cross-sectional medians")
    merged = impute_characteristics_by_month_cross_sectional_median(merged, characteristic_cols)

    # ------------------------------------------------------------------
    # 3. Missingness after imputation
    # ------------------------------------------------------------------
    missing_after = compute_missingness_for_characteristics(merged, characteristic_cols)

    after_csv = out_path.replace(".parquet", "_missingness_after.csv")
    after_jpg = out_path.replace(".parquet", "_missingness_after.jpg")

    missing_after.to_csv(after_csv, index=False)
    save_missingness_histogram(
        missing_after,
        after_jpg,
        title="Missing data percentage per characteristic (after imputation)",
        bins=20,
    )
    logger.debug(f"Saved missingness after imputation: {after_csv}, {after_jpg}")

    # Build next-month excess return target after merge/imputation stage.
    logger.debug("Building lead excess return target")
    merged["excess_ret_lead"] = merged.groupby("permno")["ret_total"].shift(-1)


    if "rf" in merged.columns:
        if merged["rf"].abs().median() > 1:
            logger.debug("Converting rf from percentage to decimal")
            merged["rf"] = merged["rf"] / 100.0
        merged["rf_lead"] = merged.groupby("permno")["rf"].shift(-1)
        merged["excess_ret_lead"] = merged["excess_ret_lead"] - merged["rf_lead"]


    # Build shifts for monthly, quarterly and annual characteristcs
    logger.debug("Building characteristic shifts")
    merged = merged.sort_values(["permno", "date"])
    grouped = merged.groupby("permno")

    for i in merged.columns:
        if i in cols_vars_monthly:
            merged[i] = grouped[i].shift(-1)
        elif i in cols_vars_quarterly:
            merged[i] = grouped[i].shift(-3)
        elif i in cols_vars_annual:
            merged[i] = grouped[i].shift(-6)


    merged = merged.dropna(subset=["excess_ret_lead"])
    logger.info(f"Complete dataset built: {merged.shape}")
    save_parquet(merged, out_path, enabled=cache_enabled)
    return merged


def reduce_observations_for_coding(
    df: pd.DataFrame,
    max_stocks_per_month: int,
    random_state: int = 42,
):
    logger.info(f"Reducing observations for coding mode: max {max_stocks_per_month} stocks per month")
    rng = np.random.RandomState(random_state)

    def _sample_month(g):
        if len(g) <= max_stocks_per_month:
            return g
        idx = rng.choice(g.index.to_numpy(), size=max_stocks_per_month, replace=False)
        return g.loc[idx].sort_values("permno")

    out = (
        df.groupby("date", group_keys=False)
        .apply(_sample_month)
        .sort_values(["date", "permno"])
        .reset_index(drop=True)
    )
    logger.info(f"Reduced dataset: {out.shape}")
    return out