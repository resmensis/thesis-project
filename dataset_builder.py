from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt

from data_inputs import load_datashare, load_crsp_monthly, load_macro_monthly
from io_utils import save_parquet

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
        monthly_median = (
            out.groupby("date", sort=False)[col]
            .transform("median")
        )

        out[col] = out[col].fillna(monthly_median)

    return out


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
        first_year_month = df.at[first_valid_index, "date"] if first_valid_index is not None else None

        last_valid_index = s.last_valid_index()
        last_year_month = df.at[last_valid_index, "date"] if last_valid_index is not None else None

        if is_numeric_dtype(s):
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


def save_missingness_three_comparison_plot(
    missing_before: pd.DataFrame,
    missing_after_first: pd.DataFrame,
    missing_after_second: pd.DataFrame,
    out_jpg_path: str,
    title: str = "Missing Data Percentage Comparison: Before/After Imputation and After Temporal Reduction",
) -> None:
    """
    Save a comparison plot showing missingness across three processing stages.

    Each input DataFrame must contain:
        - characteristic
        - missing_pct
    """

    # Rename the percentage columns before merging.
    # This is clearer and avoids suffix conflicts.
    before = missing_before[
        ["characteristic", "missing_pct"]
    ].rename(
        columns={"missing_pct": "missing_pct_before"}
    )

    after_first = missing_after_first[
        ["characteristic", "missing_pct"]
    ].rename(
        columns={"missing_pct": "missing_pct_after_first"}
    )

    after_second = missing_after_second[
        ["characteristic", "missing_pct"]
    ].rename(
        columns={"missing_pct": "missing_pct_after_second"}
    )

    # Merge all three DataFrames on characteristic
    merged = (
        before
        .merge(after_first, on="characteristic", how="inner")
        .merge(after_second, on="characteristic", how="inner")
    )

    x = range(len(merged))

    plt.figure(figsize=(14, 6))

    plt.plot(
        x,
        merged["missing_pct_before"],
        marker="o",
        linestyle="-",
        color="red",
        markersize=4,
        label="Before Imputation",
        alpha=0.7,
    )

    plt.plot(
        x,
        merged["missing_pct_after_first"],
        marker="s",
        linestyle="-",
        color="green",
        markersize=4,
        label="After First Imputation",
        alpha=0.7,
    )

    plt.plot(
        x,
        merged["missing_pct_after_second"],
        marker="^",
        linestyle="-",
        color="blue",
        markersize=4,
        label="After Second Imputation",
        alpha=0.7,
    )

    plt.xlabel("Characteristic Index")
    plt.ylabel("Missing data percentage")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        out_jpg_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


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

    crsp = load_crsp_monthly(crsp_path, possible_crsp_cols)
    macro = load_macro_monthly(macro_path, possible_marco_cols)
    ds = load_datashare(datashare_path)

    #-------------
    date_1987_05 = pd.Period("1987-05", freq="M")
    #-------------
    #-------------
    test_1 = ds.loc[ds["date"] == date_1987_05].copy()
    #-------------

    # Columns 3-96 in datashare.csv = 94 characteristics.
    characteristic_cols = cols_chara
    logger.debug(f"Identified {len(characteristic_cols)} characteristic columns")

    logger.debug("Merging datashare with CRSP")
    merged = ds.merge(
        crsp[["permno", "date", "ret", "dlret"]],
        on=["permno", "date"],
        how="inner",
        # validate="one_to_one",
    )

    #-------------
    test_2 = merged.loc[merged["date"] == date_1987_05].copy()
    #-------------

    logger.debug("Merging with macro data")
    merged = merged.merge(macro, on="date", how="left")

    merged = merged.sort_values(["permno", "date"]).reset_index(drop=True)


    #-------------
    test_3 = merged.loc[merged["date"] == date_1987_05].copy()
    #-------------

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

    #-------------
    test_4 = merged.loc[merged["date"] == date_1987_05].copy()
    #-------------

    # ------------------------------------------------------------------
    # Missingness visualisation and imputation
    # ------------------------------------------------------------------
    cols_chara_and_ret_total = characteristic_cols + ["ret_total"]

    # Visualisation before imputation
    missing_before = compute_missingness_for_characteristics(merged, cols_chara_and_ret_total)

    before_csv = f"{descriptives_path}/charas_missingness_before.csv"
    missing_before.to_csv(before_csv, index=False)

    #-------------
    test_5 = merged.loc[merged["date"] == date_1987_05].copy()
    #-------------

 
    logger.debug(f"Saved missingness before imputation: {before_csv}")

    # Imputation
    logger.info("Imputing missing characteristics using monthly cross-sectional medians")
    merged = impute_characteristics_by_month_cross_sectional_median(merged, cols_chara_and_ret_total)

    #-------------
    test_5 = merged.loc[merged["date"] == date_1987_05].copy()
    #-------------

    # Visualisation after imputation
    missing_after = compute_missingness_for_characteristics(merged, cols_chara_and_ret_total)

    after_csv = f"{descriptives_path}/charas_missingness_after.csv"
    missing_after.to_csv(after_csv, index=False)


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


    #-------------
    test_6 = merged.loc[merged["date"] == date_1987_05].copy()
    #-------------

    # Build next-month excess return target.
    logger.debug("Building lead return target (shift ret_total)")
    
    merged = merged.sort_values(["permno", "date"])
    merged["ret_total"] = merged.groupby("permno")["ret_total"].shift(-1)


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


    #-------------
    test_7 = merged.loc[merged["date"] == date_1987_05].copy()
    #-------------

    # ------------------------------------------------------------------
    # Summary 
    # ------------------------------------------------------------------
    summary_before = summarize_columns(merged, cols_chara_and_ret_total)

    summary_before_csv = f"{descriptives_path}/summary_before.csv"
    summary_before.to_csv(summary_before_csv, index=False)

    # Define inclusive monthly boundaries
    start_period = pd.Period("1957-10", freq="M")
    end_period = pd.Period("2021-06", freq="M")

    # Keep observations from October 1957 through June 2021

    mask = (
        (merged["date"] >= start_period)
        & (merged["date"] <= end_period)
    )
    merged = merged.loc[mask].copy()

    logger.debug(f"Dataset reduced due to missing values to : {merged["date"].min()} and {merged["date"].max()}")


    #-------------
    test_8 = merged.loc[merged["date"] == date_1987_05].copy()
    #-------------

    # Visualisation of missingness after temporal cuts
    missing_after_cut = compute_missingness_for_characteristics(merged, cols_chara_and_ret_total)

    after_cut_csv = f"{descriptives_path}/charas_missingness_after_cut.csv"
    missing_after_cut.to_csv(after_cut_csv, index=False)

    logger.debug(f"Saved missingness after temporal cut: {after_cut_csv}")

    # Comparison plot
    comparison_three_jpg = f"{descriptives_path}/charas_missingness_comparison_three.jpg"
    save_missingness_three_comparison_plot(
        missing_before,
        missing_after,
        missing_after_cut,
        comparison_three_jpg,
        title="Missing Data Percentage: Before vs After Imputation vs After Temporal Reduction",
    )
    logger.debug(f"Saved missingness comparison plot: {comparison_jpg}")


    #-------------
    test_9 = merged.loc[merged["date"] == date_1987_05].copy()
    #-------------
    #-------------
    test_1_csv = f"{descriptives_path}/test_1.csv"
    test_1.to_csv(test_1_csv, index=False)

    test_2_csv = f"{descriptives_path}/test_2.csv"
    test_2.to_csv(test_2_csv, index=False)

    test_3_csv = f"{descriptives_path}/test_3.csv"
    test_3.to_csv(test_3_csv, index=False)

    test_4_csv = f"{descriptives_path}/test_4.csv"
    test_4.to_csv(test_4_csv, index=False)

    test_5_csv = f"{descriptives_path}/test_5.csv"
    test_5.to_csv(test_5_csv, index=False)

    test_6_csv = f"{descriptives_path}/test_6.csv"
    test_6.to_csv(test_6_csv, index=False)

    test_7_csv = f"{descriptives_path}/test_7.csv"
    test_7.to_csv(test_7_csv, index=False)

    test_8_csv = f"{descriptives_path}/test_8.csv"
    test_8.to_csv(test_8_csv, index=False)

    test_9_csv = f"{descriptives_path}/test_9.csv"
    test_9.to_csv(test_9_csv, index=False)
    #-------------


    
    logger.info(f"Complete dataset built: {merged.shape}")
    save_parquet(merged, out_path, enabled=cache_enabled)
    return merged