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


# ================== BASIC PAGE CONFIG ==================
st.set_page_config(
    page_title="📈 Stocks Insights",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ================== THEME TOGGLE ==================
if "theme" not in st.session_state:
    st.session_state["theme"] = "Dark"

theme = st.sidebar.radio(
    "Theme",
    ["Dark", "Light"],
    index=0 if st.session_state["theme"] == "Dark" else 1,
)
st.session_state["theme"] = theme


# ================== DOWNLOAD BUTTON HELPER ==================
def download_button(df, filename="data.csv", label="📥 Download CSV"):
    if df is not None and not df.empty:
        csv = df.to_csv(index=False)
        st.download_button(
            label=label,
            data=csv,
            file_name=filename,
            mime="text/csv"
        )


# ================== AUTO UPDATE DB ==================
@st.cache_resource(ttl=24 * 60 * 60)
def silent_update():
    run_fetching()

with st.spinner("Syncing latest stock data…"):
    silent_update()


# ================== SIDEBAR ==================
company_list = get_company_list()
if not company_list:
    st.error("⚠ No tickers found in DB — run insert_companies first!")
    st.stop()

selected_companies = st.sidebar.multiselect(
    "Select Company Tickers",
    options=company_list,
    default=company_list[:2],
)

min_date = date(2015, 1, 1)
max_date = datetime.today().date()
start_date = st.sidebar.date_input("Start Date", min_date)
end_date = st.sidebar.date_input("End Date", max_date)

date_valid = True
if start_date >= end_date:
    st.sidebar.error("Start date must be earlier than end date.")
    date_valid = False


# ================== UTILS ==================
def build_budget_options():
    small = list(range(0, 100001, 10000))
    large = list(range(200000, 1000001, 100000))
    values = small + large
    labels = [f"₹{v:,.0f}" for v in values]
    return values, labels


# ================== TABS ==================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📈 Price Trends",
        "⚡ Abrupt Changes",
        "📉 Risk & Volatility",
        "🔗 Compare & Correlate",
        "🧠 Smart Insights"
    ]
)


# ================== TAB 1 — PRICE TRENDS ==================
with tab1:
    st.subheader("Price Trend + Indicators")

    if date_valid:

        view_mode = st.radio(
            "Indicator View Mode",
            ["Overlay", "Separate Panels"],
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
            "Moving Averages",
            list(ma_options.keys()),
            default=["20 SMA", "50 SMA"],
        )

        osc_options = ["RSI (14)", "MACD"]
        osc_selected = st.multiselect(
            "Oscillators",
            osc_options,
            default=["RSI (14)"],
        )

        for ticker in selected_companies:
            df = fetch_prices(ticker, start_date, end_date)
            if df is None or df.empty:
                continue

            df = add_technical_indicators(df)
            col_close = get_close_price_column(df)

            st.markdown(f"### {ticker}")

            # Main chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["trade_date"], y=df[col_close], mode="lines", name="Close"))

            if view_mode == "Overlay":
                for label in overlay_selected:
                    col = ma_options[label]
                    fig.add_trace(go.Scatter(x=df["trade_date"], y=df[col], mode="lines", name=label))

                st.plotly_chart(fig, use_container_width=True)
                download_button(df, f"{ticker}_price_with_overlay.csv")

            else:
                st.plotly_chart(fig, use_container_width=True)
                download_button(df[["trade_date", col_close]], f"{ticker}_price_only.csv")

                for label in overlay_selected:
                    col = ma_options[label]
                    if col in df.columns:
                        panel_df = df[["trade_date", col_close, col]]
                        st.line_chart(panel_df.set_index("trade_date"))
                        download_button(panel_df, f"{ticker}_{label}.csv")

            # Oscillator panels
            if "RSI (14)" in osc_selected and "RSI_14" in df.columns:
                st.line_chart(df.set_index("trade_date")[["RSI_14"]])
                download_button(df[["trade_date", "RSI_14"]], f"{ticker}_RSI.csv")

            if "MACD" in osc_selected and "MACD" in df.columns:
                st.line_chart(df.set_index("trade_date")[["MACD"]])
                download_button(df[["trade_date", "MACD"]], f"{ticker}_MACD.csv")


# ================== TAB 2 — ABRUPT MOVES ==================
with tab2:
    st.subheader("Sudden Price Movements")

    threshold_pct = st.slider("Threshold (%)", 3, 20, 7) / 100

    for ticker in selected_companies:
        df = fetch_prices(ticker, start_date, end_date)
        if df is None or df.empty:
            continue

        abrupt = detect_abrupt_changes(df, threshold_pct)
        if abrupt.empty:
            continue

        st.dataframe(abrupt)
        download_button(abrupt, f"{ticker}_abrupt_changes.csv")


# ================== TAB 3 — VOLATILITY ==================
with tab3:
    st.subheader("Volatility & Risk")

    window = st.slider("Rolling Window", 5, 55, 20)

    for ticker in selected_companies:
        df = fetch_prices(ticker)
        if df is None or df.empty:
            continue

        vr = volatility_and_risk(df, window)
        st.line_chart(vr.set_index("trade_date")[["volatility", "risk"]])
        download_button(vr, f"{ticker}_volatility_risk.csv")


# ================== TAB 4 — COMPARE & CORRELATE ==================
with tab4:
    st.subheader("Market Relationship")

    if len(selected_companies) >= 2:
        merged = compare_companies(selected_companies, start_date, end_date)
        st.line_chart(merged)
        download_button(merged, "compare_companies.csv")

        corr = correlation_analysis(selected_companies)
        st.dataframe(corr)
        download_button(corr, "correlation_matrix.csv")


# ================== TAB 5 — FORECASTING ==================
with tab5:
    st.subheader("Future Trend Forecasting")

    for ticker in selected_companies:
        df = fetch_prices(ticker)
        if df is None or df.empty:
            continue

        df = compute_sma(df)
        col_close = get_close_price_column(df)

        st.line_chart(df.set_index("trade_date")[[col_close, "SMA"]])
        download_button(df, f"{ticker}_forecast_data.csv")
