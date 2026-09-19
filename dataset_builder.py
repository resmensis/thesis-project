from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt

from data_inputs import load_datashare, load_crsp_monthly, load_macro_monthly
from io_utils import save_parquet
from transformers import GroupedMedianImputer, GroupedQuantileTransformer, GroupedLagShiftTransformer



from pathlib import Path
from typing import Optional
import warnings

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


def summary_stats_extended(
    df: pd.DataFrame,
    date_col: str,
    output_path: Optional[str] = None,
    output_name: Optional[str] = None,
    sort_df: bool = True
) -> pd.DataFrame:
    """
    Generate extended summary statistics for all columns in a DataFrame.
    
    For each column, computes:
    - dtype, min, max, range (where applicable), nunique, sum
    - missing_count, missing_pct
    - first_obs_date, last_obs_date (date of first/last non-null value)
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    date_col : str
        Name of the date column to sort by. Must be present in df and convertible
        to datetime.
    output_path : Optional[str], default None
        Directory path to save output CSV. If None, uses current working directory.
        Directory will be created if it doesn't exist.
    output_name : Optional[str], default None
        Filename (without extension) for output CSV. If None, no file is saved.
    sort_df : bool, default True
        If True, sort DataFrame by date_col internally before computing stats.
        Does not modify the original df (works on a copy).
    
    Returns
    -------
    pd.DataFrame
        DataFrame with summary statistics.
        - Rows: original df column names (variables), with 'variable' column
        - Columns: variable, dtype, min, max, range, nunique, sum, missing_count, 
                   missing_pct, first_obs_date, last_obs_date
    
    Raises
    ------
    ValueError
        If date_col is not in df or cannot be converted to datetime.
    
    Warnings
    --------
    UserWarning
        If any column has 100% missing values.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range('2020-01', periods=24, freq='ME'),
    ...     'ret': np.random.randn(24),
    ...     'me': np.random.lognormal(10, 1, 24)
    ... })
    >>> df.loc[0:2, 'ret'] = np.nan  # Leading NaNs
    >>> 
    >>> stats = summary_stats_extended(
    ...     df=df,
    ...     date_col='date',
    ...     output_path='./output',
    ...     output_name='data_quality_check',
    ...     sort_df=True
    ... )
    >>> print(stats.columns.tolist())
    ['variable', 'dtype', 'min', 'max', 'range', 'nunique', 'sum', 
     'missing_count', 'missing_pct', 'first_obs_date', 'last_obs_date']
    >>> print(stats.loc[stats['variable'] == 'ret', 'first_obs_date'].iloc[0])
    2020-04-30 00:00:00
    """
    
    # Validate date_col
    if date_col not in df.columns:
        raise ValueError(
            f"Column '{date_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Create working copy to avoid modifying original df
    df_work = df.copy()
    
    # Ensure date_col is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_work[date_col]):
        try:
            df_work[date_col] = pd.to_datetime(df_work[date_col])
        except Exception as e:
            raise ValueError(
                f"Cannot convert '{date_col}' to datetime: {e}"
            )
    
    # Sort by date if requested
    if sort_df:
        df_work = df_work.sort_values(by=date_col).reset_index(drop=True)
    
    # Initialize list to collect row dicts
    stats_rows = []
    
    # Iterate over all columns
    for col in df_work.columns:
        col_data = df_work[col]
        col_dtype = col_data.dtype
        
        # Basic stats
        dtype_val = str(col_dtype)
        nunique_val = col_data.nunique()
        missing_count = int(col_data.isna().sum())
        missing_pct = (missing_count / len(df_work)) * 100
        
        # Min, max, range (only for numeric/datetime)
        if pd.api.types.is_numeric_dtype(col_dtype) or pd.api.types.is_datetime64_any_dtype(col_dtype):
            min_val = col_data.min()
            max_val = col_data.max()
            
            # Range: only for numeric (not meaningful for datetime)
            if pd.api.types.is_numeric_dtype(col_dtype):
                if pd.notna(min_val) and pd.notna(max_val):
                    range_val = max_val - min_val
                else:
                    range_val = None
            else:
                range_val = None  # datetime columns: range not applicable
            
            # Sum: only for numeric
            if pd.api.types.is_numeric_dtype(col_dtype):
                sum_val = col_data.sum()
            else:
                sum_val = None
        else:
            # Object/categorical columns
            min_val = None
            max_val = None
            range_val = None
            sum_val = None
        
        # First/last observation date (first/last non-null value's date)
        non_null_mask = col_data.notna()
        if non_null_mask.any():
            first_obs_idx = non_null_mask.idxmax()  # First True index
            last_obs_idx = non_null_mask[::-1].idxmax()  # Last True index
            first_obs_date = df_work.loc[first_obs_idx, date_col]
            last_obs_date = df_work.loc[last_obs_idx, date_col]
        else:
            # All values are NaN
            first_obs_date = None
            last_obs_date = None
            warnings.warn(f"Column '{col}' has 100% missing values")
        
        # Collect row dict
        stats_rows.append({
            'variable': col,
            'dtype': dtype_val,
            'min': min_val,
            'max': max_val,
            'range': range_val,
            'nunique': nunique_val,
            'sum': sum_val,
            'missing_count': missing_count,
            'missing_pct': missing_pct,
            'first_obs_date': first_obs_date,
            'last_obs_date': last_obs_date
        })
    
    # Convert to DataFrame (rows = variables, columns = stats)
    stats_df = pd.DataFrame(stats_rows)
    
    # Save to CSV if output_name provided
    if output_name is not None:
        # Determine output path
        if output_path is None:
            output_dir = Path.cwd()
        else:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{output_name}.csv"
        stats_df.to_csv(output_file, index=False)
        print(f"Summary statistics saved to: {output_file}")
    
    return stats_df


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
    random_state: int,
    cache_enabled: bool    
):
    logger.info(f"Building complete dataset, output: {out_path}")


    logger.debug("Loading source datasets")

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
        # validate="one_to_one",
    )


    logger.debug("Merging with macro data")
    merged = merged.merge(macro, on="date", how="left")

    merged = merged.sort_values(["permno", "date"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # excess_ret_total
    # ------------------------------------------------------------------
    logger.debug("Creating excess_ret_total")

    has_return_data = (merged["ret"].notna() | merged["dlret"].notna())

    merged["excess_ret_total"] = (
        (1.0 + merged["ret"].fillna(0.0))
        * (1.0 + merged["dlret"].fillna(0.0))
        - 1.0 - merged["rf"]
    ).where(has_return_data)    


    summary_stats_extended(
        merged,
        date_col="date",
        output_path=descriptives_path,
        output_name="summary_1",
    )
    # ------------------------------------------------------------------
    # Missingness visualisation and imputation
    # ------------------------------------------------------------------
    cols_chara_and_excess_ret_total = characteristic_cols + ["excess_ret_total"]

    # Visualisation before imputation
    missing_before = compute_missingness_for_characteristics(merged, cols_chara_and_excess_ret_total)

    before_csv = f"{descriptives_path}/charas_missingness_before.csv"
    missing_before.to_csv(before_csv, index=False)

 
    logger.debug(f"Saved missingness before imputation: {before_csv}")

    # Imputation
    logger.info("Imputing missing characteristics using monthly cross-sectional medians")

    imputer = GroupedMedianImputer(
        group_cols=["date"],
        value_cols=cols_chara_and_excess_ret_total,
        fallback="constant",
        fill_value=0.0
    )

    merged = imputer.fit_transform(merged)
    merged = pd.DataFrame(merged)

    """
    merged = impute_characteristics_by_month_cross_sectional_median(merged, cols_chara_and_excess_ret_total)
    """


    # Visualisation after imputation
    missing_after = compute_missingness_for_characteristics(merged, cols_chara_and_excess_ret_total)

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

    summary_stats_extended(
        merged,
        date_col="date",
        output_path=descriptives_path,
        output_name="summary_2",
    )
    # ------------------------------------------------------------------
    # Quantile Transformation 
    # ------------------------------------------------------------------
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
        output_name="summary_3",
    )
    """
    # ------------------------------------------------------------------
    # Temporal shifts 
    # ------------------------------------------------------------------
    merged = merged.sort_values(["permno", "date"]).reset_index(drop=True)
    shift_1month = ["excess_ret_total", *cols_vars_monthly]


    transformer_1month = GroupedLagShiftTransformer(
        lag=-1,
        group_col="permno",
        value_cols=shift_1month,
        create_new_cols=True,
        rename_mode="none",
        suffix=None,
        original_suffix="_original",
    )
    transformer_3months = GroupedLagShiftTransformer(
        lag=-3,
        group_col="permno",
        value_cols=cols_vars_quarterly,
        create_new_cols=True,
        rename_mode="none",
        suffix=None,
        original_suffix="_original",
    )
    transformer_6months = GroupedLagShiftTransformer(
        lag=-6,
        group_col="permno",
        value_cols=cols_vars_annual,
        create_new_cols=True,
        rename_mode="none",
        suffix=None,
        original_suffix="_original",
    )

    logger.info("Creating 1-month shift.")
    merged = transformer_1month.fit_transform(merged)
    logger.debug("1-month shift created.")
    logger.info("Creating 3-month shift.")
    merged = transformer_3months.fit_transform(merged)
    logger.debug("3-month shift created.")
    logger.info("Creating 6-month shift.")
    merged = transformer_6months.fit_transform(merged)
    logger.debug("6-month shift created.")
    merged = pd.DataFrame(merged)
    logger.debug("Df after shifts recreated.")

    summary_stats_extended(
        merged,
        date_col="date",
        output_path=descriptives_path,
        output_name="summary_4",
    )


    # Define inclusive monthly boundaries
    start_period = pd.Period("1957-03-01")
    end_period = pd.Period("2021-06-01")

    # Keep observations from October 1957 through June 2021

    mask = (
        (merged["date"] >= start_period)
        & (merged["date"] <= end_period)
    )
    merged = merged.loc[mask].copy()

    logger.debug(f"Dataset reduced due to missing values to : {merged["date"].min()} and {merged["date"].max()}")


    # Visualisation of missingness after temporal cuts
    missing_after_cut = compute_missingness_for_characteristics(merged, cols_chara_and_excess_ret_total)

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

    
    logger.info(f"Complete dataset built: {merged.shape}")
    save_parquet(merged, out_path, enabled=cache_enabled)
    return merged