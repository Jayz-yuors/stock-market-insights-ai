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

# ================= Page Config ==================
st.set_page_config(
    page_title="📈 Stocks Insights",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= Theme Switcher ==================
if "theme" not in st.session_state:
    st.session_state["theme"] = "Dark"

theme = st.sidebar.radio("Theme", ["Dark", "Light"],
                         index=0 if st.session_state["theme"] == "Dark" else 1)
st.session_state["theme"] = theme

# ================= Silent DB Update ==================
@st.cache_resource(ttl=24*60*60)
def silent_update():
    run_fetching()

with st.spinner("Syncing latest stock data…"):
    silent_update()

# ================= Sidebar Sections ==================
company_list = get_company_list()
if not company_list:
    st.error("⚠ No tickers found. Please insert companies first.")
    st.stop()

selected_companies = st.sidebar.multiselect(
    "Select companies",
    options=company_list,
    default=company_list[:2]
)

min_date = date(2015,1,1)
max_date = datetime.today().date()
start_date = st.sidebar.date_input("Start Date", min_date)
end_date = st.sidebar.date_input("End Date", max_date)

if start_date >= end_date:
    st.sidebar.error("Start must be before End.")
    st.stop()

# ================= Helper for CSV Download ==================
def add_csv_download(df, filename):
    csv = df.to_csv(index=False)
    st.download_button(
        "📥 Download CSV",
        csv,
        file_name=filename,
        mime="text/csv"
    )

# ================= Tabs Layout ==================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Price Trends", "⚡ Abrupt Changes", "📉 Risk & Volatility",
     "🔗 Compare & Correlate", "🧠 Smart Insights"]
)

# ============= Tab 1 - Technical Indicators =============
with tab1:
    st.subheader("Price Trends with Indicators")

    view_mode = st.radio("Indicator View", ["Overlay on main chart", "Separate indicator panels"])

    ma_options = {
        "20 SMA": "SMA_20", "50 SMA": "SMA_50", "200 SMA": "SMA_200",
        "20 EMA": "EMA_20", "50 EMA": "EMA_50"
    }

    overlay_sel = st.multiselect(
        "Moving Averages", list(ma_options.keys()),
        default=["20 SMA", "50 SMA"]
    )

    osc_sel = st.multiselect("Oscillators", ["RSI (14)", "MACD"], default=["RSI (14)"])

    for ticker in selected_companies:
        df = fetch_prices(ticker, start_date, end_date)
        if df is None or df.empty:
            continue

        df = add_technical_indicators(df)
        col = get_close_price_column(df)

        st.markdown(f"### 📌 {ticker}")
        st.metric("Latest Price", f"₹{df[col].iloc[-1]:.2f}")

        # Graph
        if view_mode == "Overlay on main chart":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.trade_date, y=df[col], mode="lines", name="Close"))

            for label in overlay_sel:
                c = ma_options[label]
                if c in df:
                    fig.add_trace(go.Scatter(
                        x=df.trade_date, y=df[c], mode="lines", name=label
                    ))

            fig.update_layout(
                title=f"{ticker} Price + Indicators",
                xaxis_title="Date", yaxis_title="Price (₹)"
            )
            st.plotly_chart(fig, use_container_width=True)

            add_csv_download(df, f"{ticker}_technical_data.csv")

        else:
            # Price-only chart
            st.line_chart(df.set_index("trade_date")[[col]])
            add_csv_download(df[["trade_date", col]], f"{ticker}_price_only.csv")

            # Individual Panels
            for label in overlay_sel:
                c = ma_options[label]
                if c not in df: continue
                panel = df[["trade_date", col, c]]
                st.line_chart(panel.set_index("trade_date"))
                add_csv_download(panel, f"{ticker}_{label}.csv")

        # Oscillators
        if "RSI (14)" in osc_sel and "RSI_14" in df:
            st.line_chart(df.set_index("trade_date")[["RSI_14"]])
            add_csv_download(df[["trade_date", "RSI_14"]], f"{ticker}_rsi.csv")

        if "MACD" in osc_sel and "MACD" in df:
            st.line_chart(df.set_index("trade_date")[["MACD"]])
            add_csv_download(df[["trade_date", "MACD"]], f"{ticker}_macd.csv")

# ============= Tab 2 - Abrupt Changes =============
with tab2:
    st.subheader("Shock Movements Detection")

    threshold_pct = st.slider("Threshold %", 3, 20, 7) / 100

    for ticker in selected_companies:
        df = fetch_prices(ticker, start_date, end_date)
        if df is None or df.empty:
            continue

        abrupt = detect_abrupt_changes(df, threshold_pct)
        st.markdown(f"### {ticker}")

        if abrupt.empty:
            st.info("No shocks detected.")
            continue

        fig = px.bar(
            abrupt, x="trade_date", y="pct_change",
            title=f"{ticker} Sudden % Movement",
            color="pct_change",
            color_continuous_scale="RdYlGn"
        )
        fig.update_yaxes(title="% Change")

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(abrupt)

# ============= Tab 3 - Risk & Volatility =============
with tab3:
    st.subheader("Volatility & Market Risk")
    window = st.slider("Rolling Window", 5, 55, 20)

    for ticker in selected_companies:
        df = fetch_prices(ticker)
        if df is None or df.empty: continue

        vr = volatility_and_risk(df, window)

        st.markdown(f"### {ticker}")
        st.line_chart(
            vr.set_index("trade_date")[["volatility", "risk"]],
            height=400
        )

# ============= Tab 4 - Comparison =============
with tab4:
    st.subheader("Compare Companies & Correlation Matrix")

    if len(selected_companies) >= 2:
        merged = compare_companies(selected_companies, start_date, end_date)
        st.line_chart(merged)
        corr = correlation_analysis(selected_companies)
        st.dataframe(corr.style.background_gradient(cmap="coolwarm"))
        plot_correlation(corr)

# ============= Tab 5 - Smart Insights =============
with tab5:
    st.subheader("Actionable Smart Investment Suggestions")

    budget_vals, budget_labels = build_budget_options()
    colA, colB, colC = st.columns(3)

    with colA:
        label = st.selectbox("Budget", budget_labels, index=4)
        budget = budget_vals[budget_labels.index(label)]
    with colB:
        horizon = st.radio("Type", ["Short Term", "Long Term"], horizontal=True)
    with colC:
        st.metric("Forecast Window", "15 days" if horizon == "Short Term" else "60 days")

    st.caption("Not financial advice — purely educational!")

    for ticker in selected_companies:
        df = fetch_prices(ticker)
        if df is None or df.empty: continue

        df = compute_sma(df)
        col = get_close_price_column(df)

        # Trend analytics preserved
        from app import analyze_trend_confidence, project_future  # <-- keep original behavior

        conf, sig, pct, vol = analyze_trend_confidence(df, col, horizon)
        latest = df[col].iloc[-1]
        shares = int(budget / latest)

        st.markdown(f"---\n### {ticker}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Signal", sig)
        m2.metric("Confidence", f"{conf:.1f}%")
        m3.metric("Trend %", f"{pct:+.2f}%")
        m4.metric("Volatility", f"{vol:.2f}%")

        future_buy, future_sell = project_future(df, col, horizon)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.trade_date, y=df[col], mode="lines", name="Close"))
        fig.add_trace(go.Scatter(x=df.trade_date, y=df["SMA"], mode="lines", name="SMA"))

        fig.update_layout(title=f"{ticker} — Forecast Future Signals")
        st.plotly_chart(fig, use_container_width=True)

