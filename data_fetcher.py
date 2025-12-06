from mongo_config import get_db
import yfinance as yf
import pandas as pd
from datetime import datetime, date, timedelta
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

START_DATE = datetime(2015, 1, 1)
REQUEST_PAUSE_SEC = 0.5  # small pause for safety


def get_company_list():
    db = get_db()
    companies = list(db["companies"].find({}, {"ticker": 1, "_id": 0}))
    return [c["ticker"] for c in companies]


def get_latest_date(ticker):
    """
    Returns the latest datetime stored for this ticker, or None.
    """
    db = get_db()
    rec = db["stock_prices"].find_one(
        {"ticker": ticker},
        sort=[("date", -1)]
    )
    return rec["date"] if rec and rec.get("date") else None


def fetch_yfinance(ticker, start_date):
    """
    Fetch daily data from Yahoo Finance from start_date → today.
    start_date may be datetime or date, we normalize it.
    """
    if isinstance(start_date, datetime):
        start_dt = start_date
    elif isinstance(start_date, date):
        start_dt = datetime.combine(start_date, datetime.min.time())
    else:
        # fallback: try converting
        start_dt = pd.to_datetime(start_date)

    # If we've already got data up to (or beyond) today, skip
    today = datetime.today()
    if start_dt.date() >= today.date():
        logging.info(f"{ticker}: already up to date, skipping fetch.")
        return None

    start_str = start_dt.strftime("%Y-%m-%d")
    logging.info(f"{ticker}: fetching from {start_str}...")

    df = yf.download(
        ticker,
        start=start_str,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        logging.warning(f"{ticker}: Yahoo returned EMPTY dataframe.")
        return None

    logging.info(f"{ticker}: fetched {len(df)} rows from Yahoo.")
    return df


def insert_prices(df, ticker):
    """
    Insert or update prices in MongoDB.
    Handles Series/scalar values safely.
    """
    db = get_db()
    sp = db["stock_prices"]

    # Ensure datetime index
    df = df.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")

    def safe_float(v):
        # scalar numbers
        if isinstance(v, (int, float)):
            return float(v)
        # pandas Series (e.g., from multi-index)
        if isinstance(v, pd.Series):
            v = v.dropna()
            if not v.empty:
                return float(v.iloc[0])
            return None
        # try generic conversion
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def safe_int(v):
        # scalar numbers
        if isinstance(v, (int, float)):
            return int(v)
        # pandas Series
        if isinstance(v, pd.Series):
            v = v.dropna()
            if not v.empty:
                return int(v.iloc[0])
            return 0
        # generic
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0

    count = 0
    for ts, row in df.iterrows():
        if pd.isna(ts):
            continue

        trade_dt = ts.to_pydatetime()

        doc = {
            "ticker": ticker,
            "date": trade_dt,
            "open": safe_float(row.get("Open")),
            "high": safe_float(row.get("High")),
            "low": safe_float(row.get("Low")),
            "close": safe_float(row.get("Close")),
            "volume": safe_int(row.get("Volume")),
        }

        sp.update_one(
            {"ticker": ticker, "date": trade_dt},
            {"$set": doc},
            upsert=True
        )
        count += 1

    logging.info(f"{ticker}: inserted/updated {count} rows in MongoDB.")


def run_fetching():
    """
    Incremental updater:
    - For each ticker, find last stored date.
    - Fetch only data AFTER that date.
    - If no data, fetch from START_DATE.
    Safe to call from Streamlit (cached) and from CLI.
    """
    tickers = get_company_list()
    logging.info(f"Updating stock data for {len(tickers)} companies...")

    for ticker in tickers:
        last_dt = get_latest_date(ticker)

        if last_dt:
            start_date = last_dt + timedelta(days=1)
        else:
            start_date = START_DATE

        df = fetch_yfinance(ticker, start_date)
        if df is not None:
            insert_prices(df, ticker)
        else:
            logging.info(f"{ticker}: no new data to insert.")

        time.sleep(REQUEST_PAUSE_SEC)

    logging.info("✨ Stock data auto-update completed.")
