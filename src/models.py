"""Model definitions shared by training, validation and the app."""
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

FEATURES = ["month", "quarter", "lag_1", "lag_2", "lag_3", "lag_12", "roll_mean_3"]


def get_models() -> dict:
    return {
        # Ridge = linear regression with a penalty that stops it overfitting
        "Ridge (linear)": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=4, min_samples_leaf=2, random_state=42),
        "XGBoost": XGBRegressor(
            n_estimators=200, max_depth=2, learning_rate=0.05,
            subsample=0.8, random_state=42),
    }