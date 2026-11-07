"""Step 3b: simple baselines. Any real model must beat these."""
import pandas as pd

from src.data_loader import load_clean
from src.features import build_features, monthly_sales, time_split
from src.metrics import scores


def main() -> None:
    X = build_features(monthly_sales(load_clean()))
    train, test = time_split(X)
    y = test["sales"]

    baselines = {
        "Mean of training data": pd.Series(train["sales"].mean(), index=test.index),
        "Naive (last month)": test["lag_1"],
        "Seasonal naive (same month last year)": test["lag_12"],
    }
    results = pd.DataFrame({name: scores(y, p) for name, p in baselines.items()}).T
    print(results)


if __name__ == "__main__":
    main()