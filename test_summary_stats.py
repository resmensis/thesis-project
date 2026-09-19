"""
Extended summary statistics for DataFrames with date-based temporal coverage.

This module provides a function to generate comprehensive summary statistics
for all columns in a pandas DataFrame, with special handling for date columns
and temporal coverage analysis (first/last non-null observation dates).

Typical use case: CRSP/Compustat panel data quality checks before asset-pricing analysis.

Output format:
- Rows: original df column names (variables)
- Columns: dtype, min, max, range, nunique, sum, missing_count, missing_pct, 
          first_obs_date, last_obs_date
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import warnings


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


# Example usage and test
if __name__ == "__main__":
    print("Testing summary_stats_extended function...\n")
    
    # Create test DataFrame (simulating CRSP-like panel)
    test_df = pd.DataFrame({
        'date': pd.date_range('2020-01-31', periods=24, freq='ME'),
        'ret': np.random.randn(24) * 0.05,
        'me': np.random.lognormal(mean=10, sigma=1, size=24),
        'sector': np.random.choice(['Tech', 'Finance', 'Health'], size=24),
        'bm': np.random.uniform(0.5, 2.0, size=24)
    })
    
    # Introduce realistic missing data patterns
    test_df.loc[0:2, 'ret'] = np.nan      # Leading NaNs (e.g., returns need lag)
    test_df.loc[20:23, 'bm'] = np.nan     # Trailing NaNs (incomplete recent data)
    test_df.loc[5:7, 'me'] = np.nan       # Middle NaNs (data gaps)
    
    # Run function
    result = summary_stats_extended(
        df=test_df,
        date_col='date',
        output_path='./output',
        output_name='test_summary_stats',
        sort_df=True
    )
    
    print("\nResult DataFrame (stats as columns):")
    print(result.to_string())
    print("\n\nTemporal coverage analysis:")
    ret_row = result[result['variable'] == 'ret']
    me_row = result[result['variable'] == 'me']
    bm_row = result[result['variable'] == 'bm']
    print(f"ret: {ret_row['first_obs_date'].iloc[0]} to {ret_row['last_obs_date'].iloc[0]}")
    print(f"me:  {me_row['first_obs_date'].iloc[0]} to {me_row['last_obs_date'].iloc[0]}")
    print(f"bm:  {bm_row['first_obs_date'].iloc[0]} to {bm_row['last_obs_date'].iloc[0]}")
    print("\nTest completed successfully!")