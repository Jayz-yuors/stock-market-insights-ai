import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import plotly.graph_objects as go
import plotly.express as px
from plotting import plot_correlation
from calculations import (
    fetch_prices, fetch_current_price, fetch_company_info,
    compute_sma, compute_ema, detect_abrupt_changes,
    volatility_and_risk, correlation_analysis,
    compare_companies, get_close_price_column,
    add_technical_indicators,
)
from data_fetcher import get_company_list, run_fetching


# ============== BASIC PAGE CONFIG ==============
st.set_page_config(
    page_title="📈 Stocks Insights",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============== AUTO UPDATE ==============
@st.cache_resource(ttl=24 * 60 * 60)
def silent_update():
    run_fetching()

with st.spinner("Syncing latest stock data…"):
    silent_update()


# ============== SIDEBAR FILTERS ==============
company_list = get_company_list()
if not company_list:
    st.stop()

selected_companies = st.sidebar.multiselect(
    "Select Company Tickers",
    options=company_list,
    default=company_list[:2],
)

if not selected_companies:
    st.stop()

min_date = date(2015, 1, 1)
max_date = datetime.today().date()
start_date = st.sidebar.date_input("Start Date", min_date)
end_date = st.sidebar.date_input("End Date", max_date)

if start_date >= end_date:
    st.sidebar.error("Invalid Date")
    st.stop()


# Helper: CSV Download
def download_csv(df, name):
    st.download_button(
        label="📥 Download Data (CSV)",
        data=df.to_csv(index=False),
        file_name=f"{name}.csv",
        mime="text/csv",
    )


# Helper: Budget options
def build_budget_options():
    small = list(range(0, 100001, 10000))
    large = list(range(200000, 1000001, 100000))
    values = small + large
    labels = [f"₹{v:,.0f}" for v in values]
    return values, labels


# Helper: Trend Confidence
def analyze_trend_confidence(df, col_close, horizon):
    if len(df) < 15:
        return 50, "Hold", 0, 0
    lookback = 60 if horizon == "Short Term" else 180
    recent = df.tail(min(len(df), lookback))
    x = np.arange(len(recent))
    y = recent[col_close].values
    slope, _ = np.polyfit(x, y, 1)
    pct_change = (y[-1] - y[0]) / y[0] * 100
    vol = recent[col_close].pct_change().std() * 100
    conf = 50 + pct_change/2 - vol/4
    conf = float(max(5, min(95, conf)))

    if pct_change > 5 and slope > 0: label = "Strong Buy"
    elif pct_change > 1 and slope > 0: label = "Buy"
    elif pct_change < -5 and slope < 0: label = "Risky"
    else: label = "Hold"

    return conf, label, pct_change, vol


def project_future(df, col_close, horizon):
    lookback = 60 if horizon == "Short Term" else 180
    recent = df.tail(min(len(df), lookback))
    if len(recent) < 10:
        return pd.DataFrame(), pd.DataFrame()
    x = np.arange(len(recent))
    y = recent[col_close].values
    slope, intercept = np.polyfit(x, y, 1)
    future_days = 15 if horizon == "Short Term" else 60
    last_date = df["trade_date"].iloc[-1]
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=future_days)
    x_future = np.arange(len(recent), len(recent) + len(future_dates))
    future_prices = intercept + slope * x_future
    future_df = pd.DataFrame({"trade_date": future_dates, col_close: future_prices})
    future_df["SMA"] = future_df[col_close].rolling(20, min_periods=1).mean()
    return (future_df[future_df[col_close] < future_df["SMA"]],
            future_df[future_df[col_close] > future_df["SMA"]])


# ==========================================================
# TABS
# ==========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Price Trends",
    "⚡ Abrupt Changes",
    "📉 Risk & Volatility",
    "🔗 Compare & Correlate",
    "🧠 Smart Insights"
])


# TAB 1 — UPDATED
with tab1:
    st.subheader("📈 Price Trend + Technical Indicators")

    view_mode = st.radio(
        "Indicator Display Mode",
        ["Overlay on Chart", "Separate Panels"],
        horizontal=True,
    )

    ma_options = {
        "20 SMA": "SMA_20",
        "50 SMA": "SMA_50",
        "200 SMA": "SMA_200",
        "20 EMA": "EMA_20",
        "50 EMA": "EMA_50",
    }

    overlay_selected = st.multiselect(
        "Select Moving Averages",
        list(ma_options.keys()),
        default=["20 SMA", "50 SMA"]
    )

    osc_options = ["RSI (14)", "MACD"]
    osc_selected = st.multiselect(
        "Oscillators (always separate charts)",
        osc_options,
        default=["RSI (14)"]
    )

    for ticker in selected_companies:
        df = fetch_prices(ticker, start_date, end_date)
        if df is None or df.empty: continue
        df = add_technical_indicators(df)
        col_close = get_close_price_column(df)

        st.markdown(f"### 📌 {ticker}")

        if view_mode == "Overlay on Chart":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.trade_date, y=df[col_close], name="Close"))

            for label in overlay_selected:
                fig.add_trace(go.Scatter(
                    x=df.trade_date, y=df[ma_options[label]], name=label
                ))

            fig.update_layout(
                title=f"{ticker} — Price with Indicators",
                xaxis_title="Date", yaxis_title="Price (₹)",
                legend_title="Legend"
            )
            st.plotly_chart(fig, use_container_width=True)

        else:  # Separate panels
            # Close Price Panel
            st.markdown("#### 🟦 Price Trend")
            st.line_chart(df.set_index("trade_date")[col_close], height=350)

            # One panel per selected indicator
            for label in overlay_selected:
                col = ma_options[label]
                if col in df.columns:
                    st.markdown(f"#### 📌 {label} Trend")
                    st.line_chart(df.set_index("trade_date")[[col_close, col]], height=300)

        # Oscillator charts (separate in both modes)
        if "RSI (14)" in osc_selected:
            st.markdown("#### 🔄 RSI (14)")
            st.line_chart(df.set_index("trade_date")["RSI_14"], height=250)

        if "MACD" in osc_selected:
            st.markdown("#### 📉 MACD Indicator")
            st.line_chart(df.set_index("trade_date")["MACD"], height=250)

        # CSV Download
        download_csv(df, f"{ticker}_indicators")


# TAB 2 — TAB 5 (NO CHANGES)
##########################################################
# (Your existing code for Tab2, Tab3, Tab4, Tab5 remains untouched)
##########################################################
