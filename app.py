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

# ============== THEME TOGGLE ==============
if "theme" not in st.session_state:
    st.session_state["theme"] = "Dark"

theme = st.sidebar.radio(
    "Theme",
    ["Dark", "Light"],
    index=0 if st.session_state["theme"] == "Dark" else 1,
)
st.session_state["theme"] = theme

if theme == "Dark":
    bg_color = "#060814"
    card_color = "#101623"
    text_color = "#fafafa"
    accent = "#00b4d8"
else:
    bg_color = "#f3f8ff"
    card_color = "#ffffff"
    text_color = "#111827"
    accent = "#0077ff"

# ============== DEV BANNER IN SIDEBAR ==============
st.sidebar.markdown(
    """
    <div style="
        margin-top: 12px;
        padding: 10px 15px;
        text-align: center;
        border-radius: 8px;
        background: linear-gradient(90deg, #0099ff, #00cc99);
        font-size: 13px;
        font-weight: 600;
        color: white;
    ">
        👨‍💻 Developed by<br>
        <a href="https://www.linkedin.com/in/jay-keluskar-b17601358"
           target="_blank"
           style="color:white; text-decoration:none;">
            Jay Keluskar
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============== HEADER ==============
st.markdown(
    """
    <h1>📊 Stock Insights</h1>
    <p style='color:rgba(148,163,184,0.9); font-size:0.95rem;'>
        Live Nifty50 analytics with MongoDB + Yahoo Finance integration.
    </p>
    """,
    unsafe_allow_html=True
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
    st.error("⚠ No tickers found in DB — run insert_companies first!")
    st.stop()

selected_companies = st.sidebar.multiselect(
    "Select Company Tickers",
    options=company_list,
    default=company_list[:2],
)

if not selected_companies:
    st.warning("Select at least one company.")
    st.stop()

min_date = date(2015, 1, 1)
max_date = datetime.today().date()
start_date = st.sidebar.date_input("Start Date", min_date)
end_date = st.sidebar.date_input("End Date", max_date)

date_valid = True
if start_date >= end_date:
    st.sidebar.error("Start date must be earlier than end date.")
    date_valid = False


# ============== Helper Functions ==============
def build_budget_options():
    small = list(range(0, 100001, 10000))
    large = list(range(200000, 1000001, 100000))
    values = small + large
    labels = [f"₹{v:,.0f}" for v in values]
    return values, labels


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

    conf = 50 + pct_change / 2 - vol / 4
    conf = float(max(5, min(95, conf)))

    if pct_change > 5 and slope > 0:
        label = "Strong Buy"
    elif pct_change > 1 and slope > 0:
        label = "Buy"
    elif pct_change < -5 and slope < 0:
        label = "Risky / Avoid"
    else:
        label = "Hold"

    return conf, label, pct_change, vol


def project_future(df, col_close, horizon):
    lookback = 60 if horizon == "Short Term" else 180
    if len(df) < lookback:
        lookback = len(df)

    recent = df.tail(lookback)
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

    buy = future_df[future_df[col_close] < future_df["SMA"]]
    sell = future_df[future_df[col_close] > future_df["SMA"]]
    return buy, sell


def download_csv(df: pd.DataFrame, name: str):
    """Reusable download helper for tabular data."""
    st.download_button(
        label="📥 Download this data as CSV",
        data=df.to_csv(index=False),
        file_name=f"{name}.csv",
        mime="text/csv",
    )


# ============== TABS ==============
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Price Trends",
    "⚡ Abrupt Changes",
    "📉 Risk & Volatility",
    "🔗 Compare & Correlate",
    "🧠 Smart Insights"
])


# ============== TAB 1 — Price Trends + Indicators ==============
# ============== TAB 1 — Price Trends + Indicators ==============
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
        "Oscillators (separate charts)",
        osc_options,
        default=["RSI (14)"]
    )

    for ticker in selected_companies:
        df = fetch_prices(ticker, start_date, end_date)
        if df is None or df.empty:
            continue

        df = add_technical_indicators(df)
        col_close = get_close_price_column(df)

        st.markdown(f"### 📌 {ticker}")
        st.metric("Latest Close Price", f"₹ {df[col_close].iloc[-1]:.2f}")

        if view_mode == "Overlay on Chart":

            fig = go.Figure()

            # Close Price Line
            fig.add_trace(go.Scatter(
                x=df.trade_date,
                y=df[col_close],
                name="Close Price",
                mode="lines"
            ))

            # Add SMA/EMA overlays
            for label in overlay_selected:
                col = ma_options[label]
                if col in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.trade_date,
                        y=df[col],
                        name=label,
                        mode="lines"
                    ))

            fig.update_layout(
                title=f"{ticker} — Price + Indicators",
                height=500,
                margin=dict(t=80, b=10),
                legend=dict(
                    title="Indicators",
                    orientation="h",
                    yanchor="bottom",
                    y=1.10,
                    xanchor="center",
                    x=0.5,
                ),
                xaxis_title="Date",
                yaxis_title="Price (₹)"
            )

            st.plotly_chart(fig, use_container_width=True)
            download_csv(df, f"{ticker}_overlay_data")
        else:  # Separate Panels Mode

            st.markdown("#### 📌 Close Price")
            st.line_chart(df.set_index("trade_date")[col_close], height=350)

            for label in overlay_selected:
                col = ma_options[label]
                if col in df.columns:
                    st.markdown(f"#### 📊 {label}")
                    st.line_chart(df.set_index("trade_date")[[col_close, col]], height=300)

        # Oscillators (always separate)
        if "RSI (14)" in osc_selected:
            st.markdown("#### 🔄 RSI (14)")
            st.line_chart(df.set_index("trade_date")["RSI_14"], height=250)

        if "MACD" in osc_selected:
            st.markdown("#### 📉 MACD Indicator")
            st.line_chart(df.set_index("trade_date")["MACD"], height=250)

        # Download Button for Data
        with st.expander("📄 View & Download Data"):
            st.dataframe(df, use_container_width=True)
            download_csv(df, f"{ticker}_indicators")



# ============== TAB 2 — Improved Abrupt Changes ==============
with tab2:
    st.subheader("Sudden Price Jumps & Falls ✨")

    threshold_pct = st.slider("Threshold (%)", 3, 20, 7) / 100

    for ticker in selected_companies:
        df = fetch_prices(ticker, start_date, end_date)
        if df is None or df.empty:
            continue
        col_close = get_close_price_column(df)

        abrupt = detect_abrupt_changes(df, threshold_pct)
        st.markdown(f"### {ticker} — Abrupt Moves")

        if abrupt.empty:
            st.info("No major movements detected.")
            continue

        fig = px.bar(
            abrupt,
            x="trade_date",
            y="pct_change",
            color="pct_change",
            color_continuous_scale="RdYlGn",
            title=f"Abrupt % Movements — {ticker}",
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Daily % Change",
            coloraxis_colorbar_title="% Change",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(abrupt, use_container_width=True)


# ============== TAB 3 — Risk View ==============
with tab3:
    st.subheader("Volatility & Risk Index")

    window = st.slider("Volatility Window", 5, 55, 20)

    for ticker in selected_companies:
        df = fetch_prices(ticker)
        if df is None or df.empty:
            continue
        vr = volatility_and_risk(df, window)
        st.markdown(f"### {ticker} — Volatility vs Risk")

        st.line_chart(
            vr.set_index("trade_date")[["volatility", "risk"]],
            use_container_width=True,
        )


# ============== TAB 4 — Comparison View ==============
with tab4:
    st.subheader("Compare Price Trends & Correlation")

    if len(selected_companies) >= 2:
        merged = compare_companies(selected_companies, start_date, end_date)

        st.markdown("### 📈 Normalized Price Comparison")

        # Plotly Multi-Line Chart (Restored Correctly)
        fig = go.Figure()
        for ticker in selected_companies:
            fig.add_trace(go.Scatter(
                x=merged.index,
                y=merged[ticker],
                mode="lines",
                name=ticker
            ))

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Normalized Price",
            legend_title="Companies",
            title="Price Comparison Across Selected Stocks",
            height=500,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        # CSV Download
        download_csv(
            merged.reset_index(),
            "normalized_price_comparison"
        )

        st.markdown("### 🔢 Correlation Matrix")

        corr = correlation_analysis(selected_companies)
        st.dataframe(corr.style.background_gradient(cmap="coolwarm"))

        plot_correlation(corr)


# ============== TAB 5 — Smart Insights ==============
with tab5:
    st.subheader("Smart Insights, Opportunities & Forecast")

    budget_vals, budget_labels = build_budget_options()
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        budget_label = st.selectbox("Budget", budget_labels, index=4)
        budget = budget_vals[budget_labels.index(budget_label)]
    with col_b:
        horizon = st.radio("Type", ["Short Term", "Long Term"], horizontal=True)
    with col_c:
        forecast_window = 15 if horizon == "Short Term" else 60
        st.metric("Forecast Window", f"{forecast_window} days")

    st.caption(
        "Based on DB trends — Not financial advice. "
        "Do not invest solely on this. Just for educational purpose!"
    )

    for ticker in selected_companies:
        df = fetch_prices(ticker)
        if df is None or df.empty:
            continue
        df = compute_sma(df)  # keep old logic for this tab
        col_close = get_close_price_column(df)

        conf, label, pct, vol = analyze_trend_confidence(df, col_close, horizon)
        latest = df[col_close].iloc[-1]
        shares = int(budget / latest)

        st.markdown(f"---\n### {ticker}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Signal", label)
        m2.metric("Confidence", f"{conf:.1f}%")
        m3.metric("Trend %", f"{pct:+.2f}%")
        m4.metric("Volatility", f"{vol:.2f}%")

        st.caption(f"With {budget_label}, Approx shares: **{shares}**")

        buy_future, sell_future = project_future(df, col_close, horizon)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["trade_date"], y=df[col_close], mode="lines", name="Close"))
        fig.add_trace(go.Scatter(x=df["trade_date"], y=df["SMA"], mode="lines", name="SMA"))
        fig.add_trace(go.Scatter(x=buy_future["trade_date"], y=buy_future[col_close],
                                 mode="markers", marker_color="green", name="Future Buy"))
        fig.add_trace(go.Scatter(x=sell_future["trade_date"], y=sell_future[col_close],
                                 mode="markers", marker_color="red", name="Future Sell"))

        fig.update_layout(
            title=f"{ticker} — Forecasted Buy/Sell Zones",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("🔮 Forecasted Opportunities"):
            c1, c2 = st.columns(2)
            c1.markdown("#### 🟢 Future Buy Opportunities")
            if buy_future is None or buy_future.empty:
                c1.info("No future buy signals detected 🚫")
            else:
                c1.dataframe(buy_future[["trade_date", col_close]], use_container_width=True)

            c2.markdown("#### 🔴 Future Sell Opportunities")
            if sell_future is None or sell_future.empty:
                c2.info("No future sell signals detected 🚫")
            else:
                c2.dataframe(sell_future[["trade_date", col_close]], use_container_width=True)




