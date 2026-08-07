import logging
import pandas as pd

logger = logging.getLogger("eap_ml.sample_splits")


def subset_timeframe(df: pd.DataFrame, end_date: str) -> pd.DataFrame:
    end_date = pd.Timestamp(end_date)
    logger.debug(f"Subsetting timeframe to <= {end_date}")
    result = df[df["date"] <= end_date].copy()
    logger.debug(f"Subset result: {result.shape}")
    return result


def chronological_split(df: pd.DataFrame, train_end: str, val_end: str, test_end: str | None = None):
    logger.info(f"Splitting data chronologically: train<={train_end}, val<={val_end}, test<={test_end}")
    train_end = pd.Timestamp(train_end)
    val_end = pd.Timestamp(val_end)
    test_end = pd.Timestamp(test_end) if test_end else df["date"].max()
    train = df[df["date"] <= train_end].copy()
    val = df[(df["date"] > train_end) & (df["date"] <= val_end)].copy()
    test = df[(df["date"] > val_end) & (df["date"] <= test_end)].copy()
    logger.info(f"Split sizes: train={len(train)}, val={len(val)}, test={len(test)}")
    return {"train": train, "val": val, "test": test}