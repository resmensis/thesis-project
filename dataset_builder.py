"""
Dataset builder for equity prediction.

This module builds the complete dataset by:
1. Loading and merging datashare (characteristics), CRSP (returns), and macro data
2. Imputing missing characteristics (cross-sectional median by month)
3. Building excess_ret_lead (target variable)
4. Applying temporal shifts based on characteristic frequency
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional

from config import CharacteristicsFrequency


def load_and_merge_data(
    datashare_path: str,
    crsp_path: str,
    macro_path: str,
    possible_crsp_cols: List[str],
    possible_marco_cols: List[str],
    sic2_column: str,
) -> pd.DataFrame:
    """
    Load and merge the three data sources.
    
    Args:
        datashare_path: Path to datashare.csv (94 characteristics)
        crsp_path: Path to crsp_monthly.csv (returns)
        macro_path: Path to Data2024_monthly_goyal.csv (macro variables)
        possible_crsp_cols: List of possible column names in CRSP data
        possible_marco_cols: List of possible column names in macro data
        sic2_column: Column name for industry codes in datashare.csv
    
    Returns:
        Merged DataFrame with characteristics, returns, and macro variables
    """
    # Load datashare
    datashare = pd.read_csv(datashare_path)
    logger.info(f"Loaded datashare: {datashare.shape}")
    
    # Load CRSP
    crsp = pd.read_csv(crsp_path)
    logger.info(f"Loaded CRSP: {crsp.shape}")
    
    # Load macro
    macro = pd.read_csv(macro_path)
    logger.info(f"Loaded macro: {macro.shape}")
    
    # Merge datashare + CRSP
    df = pd.merge(datashare, crsp, on=["permno", "yyyymm"], how="left")
    logger.info(f"After merging datashare+CRSP: {df.shape}")
    
    # Merge with macro
    df = pd.merge(df, macro, on="yyyymm", how="left")
    logger.info(f"After merging with macro: {df.shape}")
    
    return df


def impute_missing_characteristics(df: pd.DataFrame, freq_config: CharacteristicsFrequency) -> pd.DataFrame:
    """
    Impute missing characteristics using cross-sectional median by month.
    
    Args:
        df: DataFrame with characteristics
        freq_config: CharacteristicsFrequency config with column lists
    
    Returns:
        DataFrame with imputed characteristics
    """
    # Get all characteristic columns
    all_chars = (
        freq_config.cols_vars_monthly +
        freq_config.cols_vars_quarterly +
        freq_config.cols_vars_annual
    )
    
    # Impute missing values by month (cross-sectional median)
    for char in all_chars:
        if char in df.columns:
            df[char] = df.groupby("yyyymm")[char].transform(
                lambda x: x.fillna(x.median())
            )
    
    logger.info(f"Imputed missing characteristics")
    return df


def build_excess_ret_lead(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the target variable: excess return lead (excess_ret_lead).
    
    excess_ret_lead = ret_lead - tbl_lead
    where ret_lead is the lead return and tbl_lead is the lead T-bill rate
    
    Args:
        df: DataFrame with ret and tbl columns
    
    Returns:
        DataFrame with excess_ret_lead column
    """
    # Calculate excess return lead
    df["excess_ret_lead"] = df["ret_lead"] - df["tbl_lead"]
    
    logger.info(f"Built excess_ret_lead: mean={df['excess_ret_lead'].mean():.4f}, std={df['excess_ret_lead'].std():.4f}")
    return df


def apply_temporal_shifts(df: pd.DataFrame, freq_config: CharacteristicsFrequency) -> pd.DataFrame:
    """
    Apply temporal shifts to characteristics based on their frequency.
    
    Monthly characteristics: shift(-1) - available next month
    Quarterly characteristics: shift(-3) - available next quarter
    Annual characteristics: shift(-6) - available next 6 months (conservative)
    
    Args:
        df: DataFrame with characteristics
        freq_config: CharacteristicsFrequency config with column lists
    
    Returns:
        DataFrame with shifted characteristics
    """
    # Shift monthly characteristics
    for char in freq_config.cols_vars_monthly:
        if char in df.columns:
            df[char] = df.groupby("permno")[char].shift(-1)
    
    # Shift quarterly characteristics
    for char in freq_config.cols_vars_quarterly:
        if char in df.columns:
            df[char] = df.groupby("permno")[char].shift(-3)
    
    # Shift annual characteristics
    for char in freq_config.cols_vars_annual:
        if char in df.columns:
            df[char] = df.groupby("permno")[char].shift(-6)
    
    logger.info(f"Applied temporal shifts")
    return df


def build_complete_dataset(
    datashare_path: str,
    crsp_path: str,
    macro_path: str,
    possible_crsp_cols: List[str],
    possible_marco_cols: List[str],
    cols_vars_monthly: List[str],
    cols_vars_quarterly: List[str],
    cols_vars_annual: List[str],
    sic2_column: str,
) -> pd.DataFrame:
    """
    Build the complete dataset from raw data sources.
    
    This is a pure function that always builds the dataset (no caching logic).
    Caching should be handled by the caller (run_experiments.py).
    
    Args:
        datashare_path: Path to datashare.csv
        crsp_path: Path to crsp_monthly.csv
        macro_path: Path to Data2024_monthly_goyal.csv
        possible_crsp_cols: List of possible column names in CRSP data
        possible_marco_cols: List of possible column names in macro data
        cols_vars_monthly: List of monthly characteristic column names
        cols_vars_quarterly: List of quarterly characteristic column names
        cols_vars_annual: List of annual characteristic column names
        sic2_column: Column name for industry codes
    
    Returns:
        Complete dataset DataFrame
    """
    logger.info("Building complete dataset from raw data sources.")
    
    # Create frequency config
    freq_config = CharacteristicsFrequency(
        cols_vars_monthly=cols_vars_monthly,
        cols_vars_quarterly=cols_vars_quarterly,
        cols_vars_annual=cols_vars_annual,
    )
    
    # Step 1: Load and merge data
    df = load_and_merge_data(
        datashare_path=datashare_path,
        crsp_path=crsp_path,
        macro_path=macro_path,
        possible_crsp_cols=possible_crsp_cols,
        possible_marco_cols=possible_marco_cols,
        sic2_column=sic2_column,
    )
    
    # Step 2: Impute missing characteristics
    df = impute_missing_characteristics(df, freq_config)
    
    # Step 3: Build target variable
    df = build_excess_ret_lead(df)
    
    # Step 4: Apply temporal shifts
    df = apply_temporal_shifts(df, freq_config)
    
    logger.info(f"Complete dataset built: {df.shape}")
    
    return df


def reduce_observations_for_coding(
    df: pd.DataFrame,
    max_stocks_per_month: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Reduce the dataset to max_stocks_per_month for coding mode.
    
    This samples up to max_stocks_per_month for each month independently,
    which is different from coding_reduced mode where the same 500 stocks
    are kept constant across all months.
    
    Args:
        df: Complete dataset
        max_stocks_per_month: Maximum number of stocks per month (default: 500)
        random_state: Random seed for reproducibility (default: 42)
    
    Returns:
        Reduced dataset with max_stocks_per_month per month
    """
    logger.info(f"Reducing to {max_stocks_per_month} stocks per month for coding mode.")
    
    # Set random state
    rng = np.random.RandomState(random_state)
    
    # Sample max_stocks_per_month for each month
    def sample_month(group):
        if len(group) <= max_stocks_per_month:
            return group
        else:
            sampled_idx = rng.choice(group.index, size=max_stocks_per_month, replace=False)
            return group.loc[sampled_idx]
    
    reduced = df.groupby("yyyymm").apply(sample_month).reset_index(drop=True)
    
    logger.info(f"Reduced dataset: {reduced.shape} (from {df.shape})")
    
    return reduced


def select_constant_stocks_for_coding_reduced(
    df: pd.DataFrame,
    n_stocks: int = 500,
    random_state: int = 42,
) -> List[int]:
    """
    Select a constant set of stocks (permno) for coding_reduced mode.
    
    Unlike the monthly sampling in reduce_observations_for_coding(), this function
    selects exactly n_stocks that are kept constant across the entire time period.
    This ensures the same stocks are used for all dates, maintaining temporal consistency.
    
    Selection is based on the globally set random seed to ensure reproducibility.
    
    Args:
        df: Complete dataset with 'permno' column
        n_stocks: Number of stocks to select (default: 500)
        random_state: Random seed for reproducibility (default: 42)
    
    Returns:
        List of selected permno values
    
    Example:
        >>> selected_permno = select_constant_stocks_for_coding_reduced(complete_df, n_stocks=500, random_state=42)
        >>> reduced_df = complete_df[complete_df["permno"].isin(selected_permno)]
    """
    # Get unique permno
    unique_permno = df["permno"].unique().tolist()
    
    # Set random state for reproducibility
    rng = np.random.RandomState(random_state)
    
    # Select n_stocks randomly
    if len(unique_permno) <= n_stocks:
        logger.info(f"Only {len(unique_permno)} unique stocks available, keeping all")
        selected_permno = unique_permno
    else:
        selected_permno = rng.choice(unique_permno, size=n_stocks, replace=False).tolist()
        logger.info(f"Selected {len(selected_permno)} constant stocks for coding_reduced mode")
    
    return selected_permno