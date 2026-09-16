from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_inputs import load_datashare, load_crsp_monthly, load_macro_monthly
from io_utils import maybe_load_parquet, save_parquet

logger = logging.getLogger("eap_ml.dataset_builder")


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


def save_missingness_lineplot(
    missing_df: pd.DataFrame,
    out_jpg_path: str,
    title: str,
    color: str = "blue",
):
    """
    Save a line plot with dots showing missing data percentage per characteristic.
    """
    plt.figure(figsize=(14, 6))
    plt.plot(range(len(missing_df)), missing_df["missing_pct"], marker="o", linestyle="-", color=color, markersize=4)
    plt.xlabel("Characteristic Index (sorted by missingness)")
    plt.ylabel("Missing data percentage")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_jpg_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_missingness_comparison_plot(
    missing_before: pd.DataFrame,
    missing_after: pd.DataFrame,
    out_jpg_path: str,
    title: str = "Missing Data Percentage: Before vs After Imputation",
):
    """
    Save a comparison plot showing missingness before and after imputation.
    """
    plt.figure(figsize=(14, 6))
    
    # Merge on characteristic name
    merged = missing_before.merge(
        missing_after, 
        on="characteristic", 
        suffixes=("_before", "_after")
    )
    
    plt.plot(range(len(merged)), merged["missing_pct_before"], 
             marker="o", linestyle="-", color="red", markersize=4, label="Before Imputation", alpha=0.7)
    plt.plot(range(len(merged)), merged["missing_pct_after"], 
             marker="s", linestyle="-", color="green", markersize=4, label="After Imputation", alpha=0.7)
    
    plt.xlabel("Characteristic Index")
    plt.ylabel("Missing data percentage")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
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


import numpy as np
import pandas as pd

def summarize_columns(
    df: pd.DataFrame,
    characteristic_cols: list[str],
):
    """
    data_cfg: object with attribute 'chara_cols' (list-like)
    complete: pandas DataFrame sorted by 'date' upstream or this function will sort it
    Returns: pandas DataFrame with summary per column
    """
    summary = []
    cols_chara = characteristic_cols
    df = df.sort_values("date")

    for col in cols_chara:
        s = df[col]
        dtype = s.dtype

        first_valid_index = s.first_valid_index()
        first_value = s.loc[first_valid_index] if first_valid_index is not None else None
        first_value_type = type(first_value).__name__ if first_value is not None else None
        first_year_month = df.loc[first_valid_index, "date"] if first_valid_index is not None else None

        last_valid_index = s.last_valid_index()
        last_year_month = df.loc[last_valid_index, "date"] if last_valid_index is not None else None

        if np.issubdtype(dtype, np.number):
            col_min = s.min(skipna=True)
            col_max = s.max(skipna=True)
            col_range = col_max - col_min
        else:
            col_min = col_max = col_range = None

        summary.append({
            "column": col,
            "dtype": str(dtype),
            "first_value": first_value,
            "first_value_type": first_value_type,
            "first_year_month": first_year_month,
            "last_year_month": last_year_month,
            "min": col_min,
            "max": col_max,
            "range": col_range,
        })

    summary_df = pd.DataFrame(summary)
    return summary_df


def build_complete_dataset(
    datashare_path: str,
    crsp_path: str,
    macro_path: str,
    out_path: str,
    descriptives_path: str,
    possible_crsp_cols: list[str],
    possible_marco_cols: list[str],
    cols_chara: list[str],
    cols_vars_monthly: list[str],
    cols_vars_quarterly: list[str],
    cols_vars_annual: list [str],
    cache_enabled: bool    
):
    logger.info(f"Building complete dataset, output: {out_path}")


    logger.debug("Loading source datasets")
    """
    ds = load_datashare(datashare_path)
    """
    crsp = load_crsp_monthly(crsp_path, possible_crsp_cols)
    macro = load_macro_monthly(macro_path, possible_marco_cols)
    ds = load_datashare(datashare_path)

    # Columns 3-96 in datashare.csv = 94 characteristics.
    characteristic_cols = cols_chara
    logger.debug(f"Identified {len(characteristic_cols)} characteristic columns")

    logger.debug("Merging datashare with CRSP")
    merged = ds.merge(
        crsp[["permno", "date", "ret", "dlret"]],
        on=["permno", "date"],
        how="inner",
        validate="one_to_one",
    )

    logger.debug("Merging with macro data")
    merged = merged.merge(macro, on="date", how="left")

    """
    merged["industry_code"] = merged[sic2_column].astype("string").fillna("UNK")
    """
    merged = merged.sort_values(["permno", "date"]).reset_index(drop=True)


    # ------------------------------------------------------------------
    # ret_total
    # ------------------------------------------------------------------
    logger.debug("Creating ret_total")

    has_return_data = (merged["ret"].notna() | merged["dlret"].notna())

    merged["ret_total"] = (
        (1.0 + merged["ret"].fillna(0.0))
        * (1.0 + merged["dlret"].fillna(0.0))
        - 1.0
    ).where(has_return_data)    

    # ------------------------------------------------------------------
    # Missingness visualisation and imputation
    # ------------------------------------------------------------------
    cols_chara_and_ret_total = characteristic_cols + ["ret_total"]

    # Visualisation before imputation
    missing_before = compute_missingness_for_characteristics(merged, cols_chara_and_ret_total)

    before_csv = f"{descriptives_path}/charas_missingness_before.csv"
    missing_before.to_csv(before_csv, index=False)

    """
    before_jpg = f"{descriptives_path}/charas_missingness_before.jpg"
    save_missingness_lineplot(
        missing_before,
        before_jpg,
        title="Missing data percentage per characteristic (before imputation)",
        color="red",
    )
    """
    logger.debug(f"Saved missingness before imputation: {before_csv}")

    # Imputation
    logger.info("Imputing missing characteristics using monthly cross-sectional medians")
    merged = impute_characteristics_by_month_cross_sectional_median(merged, cols_chara_and_ret_total)

    # Visualisation after imputation
    missing_after = compute_missingness_for_characteristics(merged, cols_chara_and_ret_total)

    after_csv = f"{descriptives_path}/charas_missingness_after.csv"
    missing_after.to_csv(after_csv, index=False)

    """
    after_jpg = f"{descriptives_path}/charas_missingness_after.jpg"   
    save_missingness_lineplot(
        missing_after,
        after_jpg,
        title="Missing data percentage per characteristic (after imputation)",
        color="green",
    )    
    """
    logger.debug(f"Saved missingness after imputation: {after_csv}")

    # Comparison plot
    comparison_jpg = f"{descriptives_path}/charas_missingness_comparison.jpg"
    save_missingness_comparison_plot(
        missing_before,
        missing_after,
        comparison_jpg,
        title="Missing Data Percentage: Before vs After Imputation",
    )
    logger.debug(f"Saved missingness comparison plot: {comparison_jpg}")


    # ------------------------------------------------------------------
    # Temporal shifts 
    # ------------------------------------------------------------------

    # Build next-month excess return target after merge/imputation stage.
    logger.debug("Building lead excess return target")
    
    merged = merged.sort_values(["permno", "date"])
    merged["excess_ret_lead"] = merged.groupby("permno")["ret_total"].shift(-1)


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

    # ------------------------------------------------------------------
    # Summary 
    # ------------------------------------------------------------------
    cols_summary = characteristic_cols + ["excess_ret_lead"]
    summary = summarize_columns(merged, cols_summary)

    summary_csv = f"{descriptives_path}/summary.csv"
    summary.to_csv(summary_csv, index=False)
    
    logger.info(f"Complete dataset built: {merged.shape}")
    save_parquet(merged, out_path, enabled=cache_enabled)
    return merged