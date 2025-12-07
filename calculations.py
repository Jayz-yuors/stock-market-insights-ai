from mongo_config import get_db
import pandas as pd

db = get_db()


# ========== Utility: Correct Close Price Column ==========
def get_close_price_column(df):
    for col in df.columns:
        if col.lower() == "close":
            return col
    raise KeyError("Close price column not found!")


# ========== FETCH PRICE HISTORY ==========
def fetch_prices(ticker_symbol, start_date=None, end_date=None):
    query = {"ticker": ticker_symbol}

    if start_date and end_date:
        query["date"] = {
            "$gte": pd.to_datetime(start_date),
            "$lte": pd.to_datetime(end_date),
        }

    data = list(db["stock_prices"].find(query).sort("date", 1))
    if not data:
        return None

    df = pd.DataFrame(data)
    df.rename(columns={"date": "trade_date"}, inplace=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    return df


def fetch_current_price(ticker_symbol):
    rec = db["stock_prices"].find_one(
        {"ticker": ticker_symbol},
        sort=[("date", -1)]
    )
    return float(rec["close"]) if rec else None


def fetch_company_info(ticker):
    return db["companies"].find_one({"ticker": ticker}, {"_id": 0})


# ========== PRICE INDICATORS ==========
def compute_sma(df, window=20):
    """Existing helper used in multiple places – kept as-is."""
    col = get_close_price_column(df)
    df["SMA"] = df[col].rolling(window, min_periods=1).mean()
    return df


def compute_ema(df, window=20):
    """Existing helper used in multiple places – kept as-is."""
    col = get_close_price_column(df)
    df["EMA"] = df[col].ewm(span=window, adjust=False).mean()
    return df


def detect_abrupt_changes(df, threshold=0.05):
    col = get_close_price_column(df)
    df["pct_change"] = df[col].pct_change()
    return df[abs(df["pct_change"]) > threshold]


def add_technical_indicators(df):
    """
    Add richer technical indicators on top of raw OHLCV data.

    Indicators:
    - SMA_20, SMA_50, SMA_200
    - EMA_20, EMA_50
    - RSI_14
    - MACD, MACD_signal, MACD_hist
    - Golden_Cross (1 if SMA_50 > SMA_200 else 0)
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    col = get_close_price_column(df)
    close = pd.to_numeric(df[col], errors="coerce")

    # ---- Moving Averages ----
    df["SMA_20"] = close.rolling(window=20, min_periods=1).mean()
    df["SMA_50"] = close.rolling(window=50, min_periods=1).mean()
    df["SMA_200"] = close.rolling(window=200, min_periods=1).mean()

    df["EMA_20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA_50"] = close.ewm(span=50, adjust=False).mean()

    # ---- RSI (14) ----
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    df["RSI_14"] = rsi.fillna(50)  # neutral where insufficient data

    # ---- MACD (12, 26, 9) ----
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["MACD"] = macd
    df["MACD_signal"] = macd_signal
    df["MACD_hist"] = macd - macd_signal

    # ---- Golden Cross (50 SMA above 200 SMA) ----
    gc = (
        (df["SMA_50"] > df["SMA_200"])
        & df["SMA_50"].notna()
        & df["SMA_200"].notna()
    )
    df["Golden_Cross"] = gc.astype(int)

    return df


# ========== VOLATILITY & RISK ==========
def volatility_and_risk(df, window=20):
    col = get_close_price_column(df)
    df["volatility"] = df[col].rolling(window).std()
    df["risk"] = df["volatility"] / df[col]
    return df


# ========== CORRELATION & COMPARISON ==========
def correlation_analysis(tickers):
    series_list = []
    names = []

    for ticker in tickers:
        df = fetch_prices(ticker)
        if df is None:
            continue
        col = get_close_price_column(df)
        series_list.append(df.set_index("trade_date")[col].rename(ticker))
        names.append(ticker)

    if not series_list:
        return pd.DataFrame()

    merged = pd.concat(series_list, axis=1, join="inner")
    return merged.corr()


def compare_companies(tickers, start_date=None, end_date=None):
    series_list = []

    for ticker in tickers:
        df = fetch_prices(ticker, start_date, end_date)
        if df is None:
            continue
        col = get_close_price_column(df)
        series_list.append(df.set_index("trade_date")[col].rename(ticker))

    return pd.concat(series_list, axis=1, join="inner") if series_list else pd.DataFrame()


# ========== INVESTMENT STRATEGY ==========
def best_time_to_invest(df):
    col = get_close_price_column(df)

    if "SMA" not in df.columns:
        df = compute_sma(df)

    return df[df[col] > df["SMA"]][["trade_date", col]]
