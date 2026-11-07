"""Step 6a: forecast future months with the saved model.
Each predicted month is fed back in as the 'last month' for the next one."""
import joblib
import pandas as pd

MODEL_PATH = "models/best_model.joblib"


def load_model(path: str = MODEL_PATH) -> dict:
    return joblib.load(path)


def forecast(history: pd.Series, bundle: dict, horizon: int = 6) -> pd.DataFrame:
    """history: monthly sales (month-start index). Returns forecast + band."""
    series = history.copy()
    rows = []
    for _ in range(horizon):
        t = series.index[-1] + pd.offsets.MonthBegin(1)
        feats = pd.DataFrame([{
            "month": t.month,
            "quarter": t.quarter,
            "lag_1": series.iloc[-1],
            "lag_2": series.iloc[-2],
            "lag_3": series.iloc[-3],
            "lag_12": series.iloc[-12],
            "roll_mean_3": series.iloc[-3:].mean(),
        }])[bundle["features"]]
        pred = max(float(bundle["model"].predict(feats)[0]), 0.0)
        series.loc[t] = pred
        rows.append({
            "month": t,
            "forecast": pred,
            "low": max(pred + bundle["band"]["low"], 0.0),
            "high": pred + bundle["band"]["high"],
        })
    return pd.DataFrame(rows).set_index("month")