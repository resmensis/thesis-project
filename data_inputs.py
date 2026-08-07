from __future__ import annotations

import pandas as pd
import numpy as np
from pandas import Timestamp
from typing import Any


def parse_mixed_monthly_date(x: Any) -> Any:
    if pd.isna(x):
        return pd.NaT

    s = str(x).strip()

    if s.isdigit() and len(s) == 6:
        dt = pd.to_datetime(s + "01", format="%Y%m%d", errors="coerce")
        if pd.isna(dt):
            return pd.NaT
        return dt.to_period("M").to_timestamp("M")

    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if pd.isna(dt):
        dt = pd.to_datetime(s, errors="coerce", yearfirst=True)

    if pd.isna(dt):
        return pd.NaT

    return dt.to_period("M").to_timestamp("M")


def harmonize_monthly_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = out[date_col].apply(lambda x: parse_mixed_monthly_date(x))
    return out


def load_datashare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    date_col = "date" if "date" in df.columns else "yyyymm"
    df = df.rename(columns={date_col: "date"})
    df = harmonize_monthly_dates(df, "date")
    df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")
    return df


def load_crsp_monthly(
        path: str,
        possible_crsp_cols: list[str]
    ) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    cols = possible_crsp_cols
    df = df.loc[:, df.columns.isin(cols)]
    date_col = "date" if "date" in df.columns else "yyyymm"
    df = df.rename(columns={date_col: "date"})
    df = harmonize_monthly_dates(df, "date")
    for col in ["permno", "ret", "dlret"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["dlret"] = df["dlret"].fillna(0.0)
    df["ret_total"] = (1.0 + df["ret"].fillna(0.0)) * (1.0 + df["dlret"]) - 1.0
    return df


def load_macro_monthly(
        path: str,
        possible_macro_cols: list[str]
    ) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    cols = possible_macro_cols
    df = df.loc[:, df.columns.isin(cols)]
    date_col = "date" if "date" in df.columns else "yyyymm"
    df = df.rename(columns={date_col: "date"})
    df = harmonize_monthly_dates(df, "date")
    return df
