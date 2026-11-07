"""Step 3a: build lag/calendar features from monthly sales.
Run:  python -m src.features"""
import pandas as pd

from src.data_loader import load_clean


def monthly_sales(df: pd.DataFrame) -> pd.Series:
    """Total sales per month (month-start dates), the series we will forecast."""
    return df.set_index("Order_Date")["Sales"].resample("MS").sum()


def build_features(series: pd.Series) -> pd.DataFrame:
    """Create a DataFrame with lag and calendar features from a monthly series."""
    df = pd.DataFrame({"sales": series})
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["lag_1"] = df["sales"].shift(1)
    df["lag_2"] = df["sales"].shift(2)
    df["lag_3"] = df["sales"].shift(3)
    df["lag_12"] = df["sales"].shift(12)
    df["roll_mean_3"] = df["sales"].shift(1).rolling(3).mean()
    return df.dropna()


def time_split(df: pd.DataFrame, test_months: int = 6):
    """Split into train/test by time. Last `test_months` months are test."""
    cutoff = df.index[-test_months]
    return df[df.index < cutoff], df[df.index >= cutoff]


if __name__ == "__main__":
    X = build_features(monthly_sales(load_clean()))
    print("Feature matrix shape:", X.shape)
    print(X.head(3))
    print("...\n", X.tail(3))
    train, test = time_split(X)
    print(f"\nTrain: {len(train)} months, Test: {len(test)} months")
    print(f"Train period: {train.index[0].date()} -> {train.index[-1].date()}")
    print(f"Test  period: {test.index[0].date()} -> {test.index[-1].date()}")
