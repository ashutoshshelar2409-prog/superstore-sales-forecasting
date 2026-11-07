"""Step 6b: Streamlit dashboard. Run from the project root:
    streamlit run app/streamlit_app.py"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)              # so relative data/model paths work from anywhere
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data_loader import load_clean
from src.features import monthly_sales
from src.forecast import forecast, load_model

st.set_page_config(page_title="Superstore Sales Forecast", layout="wide", page_icon="📈")

st.title("📈 Superstore Sales Forecasting Dashboard")
st.markdown("Interactive machine learning demand forecast powered by historical transactional patterns (2015–2018).")


@st.cache_data
def get_clean_data():
    return load_clean()


@st.cache_data
def get_history(df):
    return monthly_sales(df)


@st.cache_resource
def get_bundle():
    return load_model()


try:
    df_clean = get_clean_data()
    history = get_history(df_clean)
    bundle = get_bundle()
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Run `python -m src.train` first.")
    st.stop()

# Sidebar Controls
st.sidebar.header("⚙️ Forecast Settings")
horizon = st.sidebar.slider("Forecast Horizon (Months)", min_value=1, max_value=12, value=6)

st.sidebar.divider()
st.sidebar.subheader("🤖 Active Model")
st.sidebar.markdown(f"**Model Type:** {bundle.get('name', 'Ridge (linear)')}")
st.sidebar.markdown(f"**Backtest MAE:** ${bundle.get('walk_forward_mae', 13759):,.0f} / month")
st.sidebar.caption("Evaluated via 18-month rolling walk-forward cross validation.")

# Generate Forecast
fc = forecast(history, bundle, horizon)

# Top KPI Metrics Row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
total_hist_rev = df_clean["Sales"].sum()
avg_monthly_rev = history.mean()
fc_total = fc["forecast"].sum()
best_fc_month = fc["forecast"].idxmax().strftime("%b %Y")

kpi1.metric("Historical Revenue", f"${total_hist_rev:,.0f}")
kpi2.metric("Historical Avg / Month", f"${avg_monthly_rev:,.0f}")
kpi3.metric(f"Projected Next {horizon} Mo.", f"${fc_total:,.0f}")
kpi4.metric("Peak Forecast Month", best_fc_month, f"${fc['forecast'].max():,.0f}")

st.divider()

# Navigation Tabs
tab_forecast, tab_seasonality, tab_performance = st.tabs([
    "📊 Forecast & Projections",
    "📅 Seasonality & Exploration",
    "🏆 Model Evaluation"
])

with tab_forecast:
    col_plot, col_table = st.columns([1.6, 1.0])

    with col_plot:
        st.subheader("Sales History & Forward Projections")
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.plot(history.index, history.values, label="Historical Actuals", color="#1F3864", linewidth=2)
        ax.plot(fc.index, fc["forecast"], label="Model Forecast", color="#E07B00", marker="o", linewidth=2)
        ax.fill_between(
            fc.index,
            fc["low"],
            fc["high"],
            color="#E07B00",
            alpha=0.2,
            label="Confidence Band (10th–90th pct)"
        )
        ax.set_ylabel("Monthly Revenue ($)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper left")
        plt.tight_layout()
        st.pyplot(fig)

    with col_table:
        st.subheader("Monthly Projections")
        table = fc.copy()
        table.index = table.index.strftime("%b %Y")
        formatted_table = table.round(0).astype(int).rename(columns={
            "forecast": "Forecast ($)",
            "low": "Lower Bound ($)",
            "high": "Upper Bound ($)"
        })
        st.dataframe(formatted_table, use_container_width=True)

        # Download Forecast as CSV button
        csv_data = formatted_table.to_csv().encode("utf-8")
        st.download_button(
            label="📥 Download Forecast as CSV",
            data=csv_data,
            file_name=f"superstore_forecast_{horizon}months.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.info(
        "💡 **Forecasting Note:** Forecasts use recursive multi-step autoregression with lag features "
        "(`lag_1`, `lag_2`, `lag_3`, `lag_12`, and 3-month rolling mean). "
        "Confidence bands represent the empirical 10th and 90th percentile errors from walk-forward validation."
    )

with tab_seasonality:
    st.subheader("Calendar Seasonality & Historical Drivers")
    c_seas, c_cat = st.columns(2)

    with c_seas:
        st.markdown("**Average Revenue by Calendar Month**")
        m_avg = history.groupby(history.index.month).mean()
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        
        fig_s, ax_s = plt.subplots(figsize=(8, 4))
        ax_s.bar(month_labels, m_avg, color="#3470A3", edgecolor="#1F3864")
        ax_s.set_ylabel("Avg Sales ($)")
        ax_s.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig_s)
        st.caption("Sales peak substantially in November and December during holiday shopping surges.")

    with c_cat:
        st.markdown("**Revenue by Category & Region**")
        cat_data = df_clean.groupby("Category")["Sales"].sum()
        reg_data = df_clean.groupby("Region")["Sales"].sum()

        fig_b, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.pie(cat_data, labels=cat_data.index, autopct="%1.1f%%", colors=["#2E86AB", "#A23B72", "#F18F01"])
        ax1.set_title("By Category")
        ax2.pie(reg_data, labels=reg_data.index, autopct="%1.1f%%", colors=["#3E92CC", "#D8315B", "#E0A96D", "#4B5358"])
        ax2.set_title("By Region")
        plt.tight_layout()
        st.pyplot(fig_b)

with tab_performance:
    st.subheader("Model Evaluation & Backtest Scoreboard")
    st.markdown(
        "Models were tested using **18-month rolling walk-forward validation** "
        "(training strictly on history prior to month $T$, predicting month $T$, and rolling forward)."
    )

    perf_df = pd.DataFrame([
        {"Model": "Ridge (linear)", "MAE ($)": "13,759", "RMSE ($)": "16,669", "MAPE (%)": "23.3%", "Status": "Selected (Best)"},
        {"Model": "Random Forest", "MAE ($)": "14,492", "RMSE ($)": "16,502", "MAPE (%)": "25.9%", "Status": "Candidate"},
        {"Model": "Seasonal Naive (same month last year)", "MAE ($)": "14,498", "RMSE ($)": "17,876", "MAPE (%)": "23.8%", "Status": "Baseline"},
        {"Model": "XGBoost", "MAE ($)": "16,023", "RMSE ($)": "18,200", "MAPE (%)": "30.6%", "Status": "Candidate"},
        {"Model": "Naive (last month)", "MAE ($)": "20,729", "RMSE ($)": "25,335", "MAPE (%)": "38.5%", "Status": "Baseline"},
    ])
    st.table(perf_df)

    st.markdown(
        "> **Why Ridge Won:** With 48 monthly observations, tree-based ensembles (XGBoost / Random Forest) "
        "have a tendency to overfit noisy monthly peaks. Ridge regression with L2 regularization captures "
        "the underlying linear trend and seasonal annual cycle reliably without variance blow-up."
    )
