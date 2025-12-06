from mongo_config import get_db
import pandas as pd
import matplotlib.pyplot as plt

db = get_db()


def fetch_company_info(ticker):
    return db["companies"].find_one(
        {"ticker": ticker},
        {"_id": 0}
    )


def fetch_prices(ticker, start_date=None, end_date=None):
    query = {"ticker": ticker}
    
    if start_date and end_date:
        query["date"] = {"$gte": pd.to_datetime(start_date), "$lte": pd.to_datetime(end_date)}

    data = list(db["stock_prices"].find(query).sort("date", 1))
    if not data:
        return None

    df = pd.DataFrame(data)
    df.rename(columns={"date": "trade_date"}, inplace=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def fetch_current_price(ticker):
    rec = db["stock_prices"].find_one(
        {"ticker": ticker},
        sort=[("date", -1)]
    )
    return rec["close"] if rec else None


def compute_sma(df, window=20):
    df["SMA"] = df["close"].rolling(window=window, min_periods=1).mean()
    return df


def compute_ema(df, window=20):
    df["EMA"] = df["close"].ewm(span=window, adjust=False).mean()
    return df


def detect_abrupt_changes(df, threshold=0.05):
    df["pct_change"] = df["close"].pct_change()
    return df[abs(df["pct_change"]) > threshold]


def volatility_and_risk(df, window=20):
    df["volatility"] = df["close"].rolling(window=window).std()
    df["risk"] = df["volatility"] / df["close"]
    return df


def correlation_analysis(tickers):
    merged = None
    for t in tickers:
        df = fetch_prices(t)
        if df is None: continue
        df = df.set_index("trade_date")["close"]
        merged = df if merged is None else merged.join(df, how="inner", rsuffix=f"_{t}")
    return merged.corr() if merged is not None else None
