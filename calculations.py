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
        query["date"] = {"$gte": pd.to_datetime(start_date),
                         "$lte": pd.to_datetime(end_date)}

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
    col = get_close_price_column(df)
    df["SMA"] = df[col].rolling(window, min_periods=1).mean()
    return df


def compute_ema(df, window=20):
    col = get_close_price_column(df)
    df["EMA"] = df[col].ewm(span=window, adjust=False).mean()
    return df


def detect_abrupt_changes(df, threshold=0.05):
    col = get_close_price_column(df)
    df["pct_change"] = df[col].pct_change()
    return df[abs(df["pct_change"]) > threshold]


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
        if df is None: continue
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
        if df is None: continue
        col = get_close_price_column(df)
        series_list.append(df.set_index("trade_date")[col].rename(ticker))

    return pd.concat(series_list, axis=1, join="inner") if series_list else pd.DataFrame()


# ========== INVESTMENT STRATEGY ==========
def best_time_to_invest(df):
    col = get_close_price_column(df)

    if "SMA" not in df.columns:
        df = compute_sma(df)

    return df[df[col] > df["SMA"]][["trade_date", col]]
