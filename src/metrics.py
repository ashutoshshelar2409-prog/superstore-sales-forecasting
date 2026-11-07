"""Scoring utilities shared across modules."""
import numpy as np


def scores(y_true, y_pred) -> dict:
    err = y_true - y_pred
    return {
        "MAE": round(float(np.mean(np.abs(err))), 0),
        "RMSE": round(float(np.sqrt(np.mean(err ** 2))), 0),
        "MAPE_%": round(float(np.mean(np.abs(err / y_true)) * 100), 1),
    }

