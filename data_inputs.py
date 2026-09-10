from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from pandas import Timestamp
from typing import Any

logger = logging.getLogger("eap_ml.data_inputs")


def standardize_to_monthly(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Standardize date column to YYYY-MM format (first day of month).
    
    Handles two input formats efficiently:
    - YYYYMM (e.g., 196001, 196012)
    - YYYYMMDD (e.g., 19600101, 19601231)
    
    All dates are converted to the first day of the month (YYYY-MM-01).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with date column to standardize
    date_col : str, default 'date'
        Name of the date column
    
    Returns
    -------
    pd.DataFrame
        DataFrame with standardized date column (datetime64[ns] dtype)
    
    Example
    -------
    >>> df = pd.DataFrame({'date': [196001, 196012, 19610115, 19611231]})
    >>> df = standardize_to_monthly(df, 'date')
    >>> df['date']
    0   1960-01-01
    1   1960-12-01
    2   1961-01-01
    3   1961-12-01
    Name: date, dtype: datetime64[ns]
    """
    df = df.copy()
    dates = df[date_col]
    
    # Convert to string for uniform processing
    date_str = dates.astype(str).str.strip()
    
    # Detect format by string length
    # YYYYMM = 6 chars, YYYYMMDD = 8 chars
    is_monthly = date_str.str.len() == 6
    is_daily = date_str.str.len() == 8
    
    result = pd.Series(index=dates.index, dtype='datetime64[ns]')
    
    # Process YYYYMM format (6 digits)
    if is_monthly.any():
        monthly_dates = date_str.loc[is_monthly]
        # Parse as YYYY-MM by inserting dash
        result.loc[is_monthly] = pd.to_datetime(
            monthly_dates.str[:4] + '-' + monthly_dates.str[4:],
            format='%Y-%m'
        )
    
    # Process YYYYMMDD format (8 digits)
    if is_daily.any():
        daily_dates = date_str.loc[is_daily]
        # Parse full date then convert to month start
        result.loc[is_daily] = pd.to_datetime(
            daily_dates,
            format='%Y%m%d'
        ).dt.to_period('M').dt.to_timestamp('M', how='start')
    
    # Handle any remaining values (try generic parsing)
    mask = result.isna() & dates.notna()
    if mask.any():
        remaining = dates.loc[mask]
        # Try YYYYMMDD first
        parsed = pd.to_datetime(remaining, format='%Y%m%d', errors='coerce')
        # For failures, try YYYYMM
        failed_mask = parsed.isna()
        if failed_mask.any():
            parsed.loc[failed_mask] = pd.to_datetime(
                remaining.loc[failed_mask].astype(str).str[:4] + '-' + 
                remaining.loc[failed_mask].astype(str).str[4:6],
                format='%Y-%m',
                errors='coerce'
            )
        # Convert to month start
        result.loc[mask] = parsed.dt.to_period('M').dt.to_timestamp('M', how='start')
    
    df[date_col] = result
    return df


def load_datashare(path: str) -> pd.DataFrame:
    logger.debug(f"Loading datashare from {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    date_col = "date" if "date" in df.columns else "yyyymm"
    df = df.rename(columns={date_col: "date"})
    df = standardize_to_monthly(df, "date")
    df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")
    logger.debug(f"Datashare loaded: {len(df)} rows, {len(df.columns)} columns")
    return df


def load_crsp_monthly(
        path: str,
        possible_crsp_cols: list[str]
    ) -> pd.DataFrame:
    logger.debug(f"Loading CRSP monthly data from {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    cols = possible_crsp_cols
    df = df.loc[:, df.columns.isin(cols)]
    date_col = "date" if "date" in df.columns else "yyyymm"
    df = df.rename(columns={date_col: "date"})
    df = standardize_to_monthly(df, "date")
    for col in ["permno", "ret", "dlret"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["dlret"] = df["dlret"].fillna(0.0)
    df["ret"] = df["ret"].fillna(0.0)
    df["ret_total"] = (1.0 + df["ret"]) * (1.0 + df["dlret"]) - 1.0
    logger.debug(f"CRSP loaded: {len(df)} rows")
    return df


def load_macro_monthly(
        path: str,
        possible_macro_cols: list[str]
    ) -> pd.DataFrame:
    logger.debug(f"Loading macro data from {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    cols = possible_macro_cols
    df = df.loc[:, df.columns.isin(cols)]
    date_col = "date" if "date" in df.columns else "yyyymm"
    df = df.rename(columns={date_col: "date"})
    df = standardize_to_monthly(df, "date")
    logger.debug(f"Macro loaded: {len(df)} rows, columns: {list(df.columns)}")
    return df