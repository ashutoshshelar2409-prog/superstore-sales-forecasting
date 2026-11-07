"""Step 1: load and clean the raw Superstore CSV."""
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "superstore.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "superstore_clean.csv"


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Read the CSV exactly as it is on disk."""
    if path is None:
        if RAW_PATH.exists():
            path = RAW_PATH
        else:
            # Fallback to any csv in data/raw if named differently
            raw_csvs = list((BASE_DIR / "data" / "raw").glob("*.csv"))
            if raw_csvs:
                path = raw_csvs[0]
            else:
                path = RAW_PATH
    # The Kaggle file is not UTF-8, so we must say which encoding it uses.
    return pd.read_csv(path, encoding="latin-1")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix types and obvious problems. Returns a new DataFrame."""
    df = df.copy()

    # Dates are text like '8/11/2017' meaning 8 November (day first).
    for col in ["Order_Date", "Ship_Date"]:
        df[col] = pd.to_datetime(df[col], dayfirst=True)

    # Postal codes are identifiers, not numbers: keep as text, no '.0'.
    df["Postal_Code"] = (
        df["Postal_Code"].astype("Int64").astype("string")
    )

    # Remove exact duplicate rows if any.
    df = df.drop_duplicates()

    # Sort by time: essential for any forecasting work later.
    df = df.sort_values("Order_Date").reset_index(drop=True)
    return df


def load_clean() -> pd.DataFrame:
    return clean(load_raw())


if __name__ == "__main__":
    df = load_clean()
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print("Shape:", df.shape)
    print("Date range:", df["Order_Date"].min().date(), "->", df["Order_Date"].max().date())
    print("\nColumn types:\n", df.dtypes)
    print("\nMissing values:\n", df.isna().sum()[df.isna().sum() > 0])
    print("\nTotal sales:", round(df["Sales"].sum(), 2))