"""Step 6b: Streamlit dashboard. Run from the project root:
    streamlit run app/streamlit_app.py"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)              # so relative data/model paths work from anywhere
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import streamlit as st

from src.data_loader import load_clean
from src.features import monthly_sales
from src.forecast import forecast, load_model

st.set_page_config(page_title="Superstore Sales Forecast", layout="wide")
st.title("Superstore monthly sales forecast")


@st.cache_data
def get_history():
    return monthly_sales(load_clean())


@st.cache_resource
def get_bundle():
    return load_model()


try:
    history, bundle = get_history(), get_bundle()
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Run `python -m src.train` first.")
    st.stop()

horizon = st.sidebar.slider("Months to forecast", 1, 12, 6)
st.sidebar.markdown(f"**Model:** {bundle['name']}")
st.sidebar.markdown(f"**Typical error (back-test MAE):** ${bundle['walk_forward_mae']:,.0f} / month")

fc = forecast(history, bundle, horizon)

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(history.index, history.values, label="Actual", color="#1F3864")
ax.plot(fc.index, fc["forecast"], label="Forecast", color="#E07B00", marker="o")
ax.fill_between(fc.index, fc["low"], fc["high"], color="#E07B00", alpha=0.2,
                label="Likely range (10th-90th pct of past errors)")
ax.set_ylabel("Sales ($)"); ax.legend(); ax.grid(alpha=0.3)
st.pyplot(fig)

c1, c2 = st.columns(2)
c1.metric("Forecast total", f"${fc['forecast'].sum():,.0f}")
c2.metric("Best month", fc["forecast"].idxmax().strftime("%b %Y"))

table = fc.copy()
table.index = table.index.strftime("%b %Y")
st.dataframe(table.round(0).astype(int).rename(columns={
    "forecast": "Forecast ($)", "low": "Low ($)", "high": "High ($)"}))

st.caption("Trained on 4 years of monthly data (48 months). Forecasts are uncertain; "
           "the model only beats a 'same month last year' guess by a small margin. "
           "Later months are less reliable because each one builds on earlier forecasts.")