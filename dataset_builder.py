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
from typing import List, Optional

from config import CharacteristicsFrequency


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
    # Load datashare
    datashare = pd.read_csv(datashare_path)
    
    # Load CRSP
    crsp = pd.read_csv(crsp_path)
    
    # Load macro
    macro = pd.read_csv(macro_path)
    
    # Merge datashare + CRSP
    df = pd.merge(datashare, crsp, on=["permno", "yyyymm"], how="left")
    
    # Merge with macro
    df = pd.merge(df, macro, on="yyyymm", how="left")
    
    # Get all characteristic columns
    all_chars = cols_vars_monthly + cols_vars_quarterly + cols_vars_annual
    
    # Impute missing values by month (cross-sectional median)
    for char in all_chars:
        if char in df.columns:
            df[char] = df.groupby("yyyymm")[char].transform(
                lambda x: x.fillna(x.median())
            )
    
    # Calculate excess return lead
    df["excess_ret_lead"] = df["ret_lead"] - df["tbl_lead"]
    
    # Shift monthly characteristics
    for char in cols_vars_monthly:
        if char in df.columns:
            df[char] = df.groupby("permno")[char].shift(-1)
    
    # Shift quarterly characteristics
    for char in cols_vars_quarterly:
        if char in df.columns:
            df[char] = df.groupby("permno")[char].shift(-3)
    
    # Shift annual characteristics
    for char in cols_vars_annual:
        if char in df.columns:
            df[char] = df.groupby("permno")[char].shift(-6)
    
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
    
    return reduced