import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import plotly.graph_objects as go
import plotly.express as px

from plotting import plot_correlation
from calculations import (
    fetch_prices, compute_sma,
    detect_abrupt_changes, volatility_and_risk,
    correlation_analysis, compare_companies,
    get_close_price_column, add_technical_indicators
)
from data_fetcher import get_company_list, run_fetching


# ============== PAGE CONFIG ==============
st.set_page_config(
    page_title="📈 Stocks Insights",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============== DB AUTO UPDATE ==============
@st.cache_resource(ttl=24 * 60 * 60)
def silent_update():
    run_fetching()

with st.spinner("⏳ Syncing latest stock data…"):
    silent_update()


# ============== SIDEBAR FILTERS ==============
company_list = get_company_list()
if not company_list:
    st.error("⚠ No tickers found — insert companies first!")
    st.stop()

selected_companies = st.sidebar.multiselect(
    "Select Companies",
    options=company_list,
    default=company_list[:2],
)

min_date = date(2015, 1, 1)
max_date = datetime.today().date()
start_date = st.sidebar.date_input("Start Date", min_date)
end_date = st.sidebar.date_input("End Date", max_date)

if start_date >= end_date:
    st.sidebar.error("Start date must be before End date.")
    st.stop()


# ============== CSV DOWNLOAD HELPER ==============
def download_csv(df, name):
    csv = df.to_csv(index=False)
    st.download_button(
        "📥 Download Data as CSV",
        csv,
        mime="text/csv",
        file_name=f"{name}.csv"
    )


# ============== Helper Functions TAB 5 ==============
def build_budget_options():
    small = list(range(0, 100001, 10000))
    large = list(range(200000, 1000001, 100000))
    values = small + large
    labels = [f"₹{v:,.0f}" for v in values]
    return values, labels


def analyze_trend_confidence(df, col_close, horizon):
    if len(df) < 15: return 50, "Hold", 0, 0
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
    elif pct_change < -5 and slope < 0: label = "Risky / Avoid"
    else: label = "Hold"
    return conf, label, pct_change, vol


def project_future(df, col_close, horizon):
    lookback = 60 if horizon == "Short Term" else 180
    if len(df) < lookback: lookback = len(df)
    recent = df.tail(lookback)
    if len(recent) < 10: return pd.DataFrame(), pd.DataFrame()
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


# ============== TABS ==============
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Price Trends",
    "⚡ Abrupt Changes",
    "📉 Risk & Volatility",
    "🔗 Compare & Correlate",
    "🧠 Smart Insights"
])


# TAB 1 — Price Trend + Indicators (WITH CSV)
with tab1:
    st.subheader("📈 Price Trend with Indicators")
    view_mode = st.radio("Indicator View Mode", ["Overlay", "Separate Panels"], horizontal=True)

    ma_opts = {"20 SMA": "SMA_20", "50 SMA": "SMA_50"}
    ma_selected = st.multiselect("Select Moving Averages", list(ma_opts.keys()), default=list(ma_opts.keys()))

    osc_list = ["RSI (14)", "MACD"]
    osc_selected = st.multiselect("Select Oscillators (Separate Panels)", osc_list, default=["RSI (14)"])

    for ticker in selected_companies:
        df = fetch_prices(ticker, start_date, end_date)
        if df is None or df.empty: continue
        df = add_technical_indicators(df)
        col = get_close_price_column(df)

        st.markdown(f"### {ticker}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.trade_date, y=df[col], name="Close"))

        for label in ma_selected:
            fig.add_trace(go.Scatter(x=df.trade_date, y=df[ma_opts[label]], name=label))

        fig.update_layout(title=f"{ticker} Price Trend", xaxis_title="Date", yaxis_title="Price ₹")
        st.plotly_chart(fig, use_container_width=True)

        download_csv(df, f"{ticker}_price_indicators")

        # Oscillators (ALWAYS separate)
        if "RSI (14)" in osc_selected:
            st.line_chart(df.set_index("trade_date")["RSI_14"], height=250)

        if "MACD" in osc_selected:
            st.line_chart(df.set_index("trade_date")["MACD"], height=250)


# TAB 2 — Abrupt Price Changes (RESTORED)
with tab2:
    st.subheader("⚡ Sudden Market Shocks Detection")
    threshold = st.slider("Threshold % Movement", 3, 20, 7) / 100

    for ticker in selected_companies:
        df = fetch_prices(ticker, start_date, end_date)
        if df is None or df.empty: continue

        abrupt = detect_abrupt_changes(df, threshold)
        st.markdown(f"### {ticker}")

        if abrupt.empty:
            st.info("No major sudden movements detected.")
            continue

        fig = px.bar(
            abrupt, x="trade_date", y="pct_change",
            color="pct_change", color_continuous_scale="RdYlGn",
            title=f"Sudden % Jumps — {ticker}"
        )
        fig.update_yaxes(title="% Change")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(abrupt)


# TAB 3 — Volatility Risk
with tab3:
    st.subheader("📉 Volatility & Market Risk Analysis")
    window = st.slider("Rolling Window Size", 5, 55, 20)

    for ticker in selected_companies:
        df = fetch_prices(ticker)
        if df is None or df.empty: continue

        vr = volatility_and_risk(df, window)
        st.markdown(f"### {ticker}")
        st.line_chart(vr.set_index("trade_date")[["volatility", "risk"]], height=350)


# TAB 4 — Comparison & Correlation
with tab4:
    st.subheader("🔗 Market Comparison & Correlation Matrix")

    if len(selected_companies) >= 2:
        merged = compare_companies(selected_companies, start_date, end_date)
        st.line_chart(merged)

        corr = correlation_analysis(selected_companies)
        st.dataframe(corr.style.background_gradient(cmap="coolwarm"))
        plot_correlation(corr)


# TAB 5 — Smart Insights — FULLY RESTORED
with tab5:
    st.subheader("🧠 Smart Investment Suggestions")

    budget_vals, budget_labels = build_budget_options()
    col1, col2, col3 = st.columns(3)

    with col1:
        budget_label = st.selectbox("Investment Budget", budget_labels, index=4)
        budget = budget_vals[budget_labels.index(budget_label)]
    with col2:
        horizon = st.radio("Investment Horizon", ["Short Term", "Long Term"], horizontal=True)
    with col3:
        st.metric("Forecast Window", "15 days" if horizon == "Short Term" else "60 days")

    for ticker in selected_companies:
        df = fetch_prices(ticker)
        if df is None or df.empty: continue
        df = compute_sma(df)
        col = get_close_price_column(df)

        conf, label, pct, vol = analyze_trend_confidence(df, col, horizon)
        latest_price = df[col].iloc[-1]
        shares = int(budget / latest_price)

        st.markdown(f"---\n### {ticker}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Signal", label)
        m2.metric("Confidence", f"{conf:.1f}%")
        m3.metric("Trend Movement", f"{pct:+.2f}%")
        m4.metric("Volatility", f"{vol:.2f}%")

        buy, sell = project_future(df, col, horizon)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.trade_date, y=df[col], mode="lines", name="Close"))
        fig.add_trace(go.Scatter(x=df.trade_date, y=df["SMA"], mode="lines", name="SMA"))

        fig.update_layout(title=f"{ticker} Forecast Zones")
        st.plotly_chart(fig, use_container_width=True)
