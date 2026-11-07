"""Step 2: explore the data. Run:  python -m src.explore
Saves charts into reports/figures/ and prints the key numbers."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save charts to files instead of opening windows
import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import load_clean

FIG = Path("reports/figures")
FIG.mkdir(parents=True, exist_ok=True)


def monthly_sales(df: pd.DataFrame) -> pd.Series:
    """Total sales per month (month-start dates), the series we will forecast."""
    return df.set_index("Order_Date")["Sales"].resample("MS").sum()


def main() -> None:
    df = load_clean()
    m = monthly_sales(df)

    print("Months in series:", len(m), "| any missing months:", m.isna().any())
    print("\nSales by year:\n", df.groupby(df["Order_Date"].dt.year)["Sales"].sum().round(0))
    print("\nSales by category:\n", df.groupby("Category")["Sales"].sum().round(0))
    print("\nSales by region:\n", df.groupby("Region")["Sales"].sum().round(0))
    print("\nAverage sales by calendar month:\n",
          m.groupby(m.index.month).mean().round(0))

    # 1) Monthly sales over time: look for trend and repeating peaks
    ax = m.plot(figsize=(10, 4), marker="o", title="Monthly sales")
    ax.set_ylabel("Sales ($)")
    plt.tight_layout(); plt.savefig(FIG / "monthly_sales.png", dpi=120); plt.close()

    # 2) Seasonality: average sales for each calendar month
    ax = m.groupby(m.index.month).mean().plot(kind="bar", figsize=(8, 4),
                                              title="Average sales by calendar month")
    ax.set_xlabel("Month"); ax.set_ylabel("Avg sales ($)")
    plt.tight_layout(); plt.savefig(FIG / "seasonality.png", dpi=120); plt.close()

    # 3) Category split per year
    pivot = df.pivot_table(index=df["Order_Date"].dt.year, columns="Category",
                           values="Sales", aggfunc="sum")
    pivot.plot(kind="bar", figsize=(8, 4), title="Sales by category per year")
    plt.tight_layout(); plt.savefig(FIG / "category_by_year.png", dpi=120); plt.close()
    print("\nCharts saved in", FIG)


if __name__ == "__main__":
    main()