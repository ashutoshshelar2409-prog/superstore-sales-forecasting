"""Step 5: walk-forward validation, a more trustworthy test than one 6-month split.
Idea: pretend it is month T, train on everything before T, predict month T,
then move forward one month and repeat. Run:  python -m src.walk_forward"""
import pandas as pd

from src.metrics import scores
from src.data_loader import load_clean
from src.features import build_features, monthly_sales
from src.models import FEATURES, get_models

MIN_TRAIN = 18  # first forecast is made after seeing 18 months of feature rows


def evaluate(X: pd.DataFrame):
    """Returns (actual, predictions) where predictions maps name -> Series."""
    models = get_models()
    preds = {name: [] for name in models}
    preds["Baseline: last month"] = []
    preds["Baseline: same month last year"] = []
    actual, index = [], []

    for t in range(MIN_TRAIN, len(X)):
        train, row = X.iloc[:t], X.iloc[[t]]
        actual.append(row["sales"].iloc[0]); index.append(row.index[0])
        for name, model in models.items():
            model.fit(train[FEATURES], train["sales"])
            preds[name].append(model.predict(row[FEATURES])[0])
        preds["Baseline: last month"].append(row["lag_1"].iloc[0])
        preds["Baseline: same month last year"].append(row["lag_12"].iloc[0])

    y = pd.Series(actual, index=index)
    return y, {n: pd.Series(p, index=index) for n, p in preds.items()}


def main() -> None:
    X = build_features(monthly_sales(load_clean()))
    y, preds = evaluate(X)
    table = pd.DataFrame(
        {n: scores(y, p) for n, p in preds.items()}
    ).T.sort_values("MAE")
    print(f"Walk-forward forecasts: {len(y)} months ({y.index[0].date()} -> {y.index[-1].date()})\n")
    print(table)

    # How often does each model beat the best baseline, month by month?
    base_err = (y - preds["Baseline: same month last year"]).abs()
    print("\nMonths where model beats 'same month last year' baseline:")
    for n in get_models():
        err = (y - preds[n]).abs()
        print(f"  {n}: {(err < base_err).sum()} of {len(y)}")


if __name__ == "__main__":
    main()