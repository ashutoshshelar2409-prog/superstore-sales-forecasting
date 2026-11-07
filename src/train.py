"""Step 4 + 6a: pick the best model by WALK-FORWARD score, refit on all data, save it.
Run:  python -m src.train"""
from pathlib import Path

import joblib
import pandas as pd

from src.metrics import scores
from src.data_loader import load_clean
from src.features import build_features, monthly_sales
from src.models import FEATURES, get_models
from src.walk_forward import evaluate

MODEL_DIR = Path("models")


def main() -> None:
    X = build_features(monthly_sales(load_clean()))
    y, preds = evaluate(X)

    table = pd.DataFrame({n: scores(y, p) for n, p in preds.items()}).T.sort_values("MAE")
    print(table, "\n")

    # Choose among real models only (baselines are just references)
    real = list(get_models())
    best = min(real, key=lambda n: scores(y, preds[n])["MAE"])

    # Typical forecast error of the best model: used for the app's uncertainty band
    errors = y - preds[best]
    band = {"low": float(errors.quantile(0.10)), "high": float(errors.quantile(0.90))}

    final = get_models()[best]
    final.fit(X[FEATURES], X["sales"])  # refit on ALL data before deploying
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"name": best, "model": final, "features": FEATURES, "band": band,
                 "walk_forward_mae": scores(y, preds[best])["MAE"]},
                MODEL_DIR / "best_model.joblib")
    print(f"Best by walk-forward MAE: {best} -> saved to models/best_model.joblib")
    print(f"Error band (10th/90th percentile of past errors): {band}")


if __name__ == "__main__":
    main()