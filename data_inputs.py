from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from pandas import Timestamp
from typing import Any

logger = logging.getLogger("eap_ml.data_inputs")


import pandas as pd


def standardize_to_monthly(
    df: pd.DataFrame,
    date_col: str = "date",
    *,
    copy: bool = False,
) -> pd.DataFrame:
    """
    Convert one consistently encoded date column to a monthly Period column.

    Supported input formats per dataset:
    - YYYYMM:   198605  -> Period('1986-05', 'M')
    - YYYYMMDD: 19860530 -> Period('1986-05', 'M')

    The function detects the format once and parses the full column vectorially.
    It is designed for large datasets: it does not use row-wise apply().

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    date_col : str, default "date"
        Date-column name.
    copy : bool, default False
        If True, return a copy. If False, modify df in place.

    Returns
    -------
    pd.DataFrame
        DataFrame whose date_col has dtype period[M].

    Raises
    ------
    KeyError
        If date_col is absent.
    ValueError
        If the column is empty, contains no valid date values, contains
        unsupported values, or mixes YYYYMM and YYYYMMDD formats.
    """
    if date_col not in df.columns:
        raise KeyError(f"Column '{date_col}' not found in DataFrame.")

    if copy:
        df = df.copy()

    raw = df[date_col]

    # Preserve missing values while giving every observed entry one uniform form.
    date_str = raw.astype("string").str.strip()

    # `198605.0` can arise if a numeric CSV column contains missing values.
    date_str = date_str.str.replace(r"\.0$", "", regex=True)

    non_missing = date_str.dropna()
    if non_missing.empty:
        raise ValueError(f"Column '{date_col}' contains no non-missing dates.")

    lengths = non_missing.str.len()
    unique_lengths = set(lengths.unique())

    if unique_lengths == {6}:
        parsed = pd.to_datetime(date_str, format="%Y%m", errors="coerce")

    elif unique_lengths == {8}:
        parsed = pd.to_datetime(date_str, format="%Y%m%d", errors="coerce")

    elif unique_lengths.issubset({6, 8}):
        raise ValueError(
            f"Column '{date_col}' mixes YYYYMM and YYYYMMDD formats. "
            "Each dataset must use exactly one format."
        )

    else:
        bad_examples = non_missing.loc[~lengths.isin([6, 8])].head(5).tolist()
        raise ValueError(
            f"Unsupported date format in '{date_col}'. "
            f"Expected YYYYMM or YYYYMMDD; examples: {bad_examples}"
        )

    invalid = raw.notna() & parsed.isna()
    if invalid.any():
        bad_examples = date_str.loc[invalid].head(5).tolist()
        raise ValueError(
            f"Invalid calendar dates in '{date_col}'; examples: {bad_examples}"
        )

    # Monthly key: suitable for matching, grouping, and monthly forecasts.
    df[date_col] = parsed.dt.to_period("M")

    return df

def load_datashare(path: str) -> pd.DataFrame:
    logger.debug(f"Loading datashare from {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    date_col = "date" if "date" in df.columns else "yyyymm"
    df = df.rename(columns={date_col: "date"})
    logger.debug("Standardizing date of datashare")
    df = standardize_to_monthly(df, date_col="date")
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
    logger.debug("Standardizing date of CRSP")
    df = standardize_to_monthly(df, date_col="date")
    for col in ["permno", "ret", "dlret"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.debug("Creating ret_total")
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
    logger.debug("Standardizing date of Macro")
    df = standardize_to_monthly(df, date_col="date")
    logger.debug(f"Macro loaded: {len(df)} rows, columns: {list(df.columns)}")
    return df