import pandas as pd


def subset_timeframe(df: pd.DataFrame, end_date: str) -> pd.DataFrame:
    end_date = pd.Timestamp(end_date)
    return df[df["date"] <= end_date].copy()


def chronological_split(df: pd.DataFrame, train_end: str, val_end: str, test_end: str | None = None):
    train_end = pd.Timestamp(train_end)
    val_end = pd.Timestamp(val_end)
    test_end = pd.Timestamp(test_end) if test_end else df["date"].max()
    train = df[df["date"] <= train_end].copy()
    val = df[(df["date"] > train_end) & (df["date"] <= val_end)].copy()
    test = df[(df["date"] > val_end) & (df["date"] <= test_end)].copy()
    return {"train": train, "val": val, "test": test}
