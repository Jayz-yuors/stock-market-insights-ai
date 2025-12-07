from mongo_config import get_db
from insert_companies import insert_companies
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
REQUEST_PAUSE_SEC = 0.5  # small delay to avoid rate limits


# 🔥 Auto-insert companies if DB is empty
def get_company_list():
    db = get_db()
    companies = list(db["companies"].find({}, {"ticker": 1, "_id": 0}))

    if not companies:
        logging.warning("⚠ No companies found — inserting default list!")
        insert_companies()
        companies = list(db["companies"].find({}, {"ticker": 1, "_id": 0}))
        logging.info("✔ Default companies inserted!")

    return [c["ticker"] for c in companies]


def get_latest_date(ticker):
    """Returns the most recent stored date for a ticker."""
    db = get_db()
    rec = db["stock_prices"].find_one(
        {"ticker": ticker},
        sort=[("date", -1)]
    )
    return rec["date"] if rec else None


def fetch_yfinance(ticker, start_date):
    """
    Fetch stock data from Yahoo Finance from start_date until today.
    """
    today = datetime.today()

    if isinstance(start_date, datetime):
        start_dt = start_date
    else:
        start_dt = datetime.combine(start_date, datetime.min.time())

    # If already up to date
    if start_dt.date() >= today.date():
        logging.info(f"{ticker}: already updated — skipping")
        return None

    logging.info(f"{ticker}: fetching data from {start_dt.date()}...")

    df = yf.download(
        ticker,
        start=start_dt.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        logging.warning(f"{ticker}: No data returned from Yahoo")
        return None

    logging.info(f"{ticker}: fetched {len(df)} rows")
    return df


def insert_prices(df, ticker):
    """
    Insert fetched data into MongoDB using upsert.
    """
    db = get_db()
    sp = db["stock_prices"]

    df = df.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")

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

    logging.info(f"{ticker}: inserted/updated {count} rows")


def run_fetching():
    """
    Update only missing days for each ticker — always safe to call.
    """
    tickers = get_company_list()
    logging.info(f"🔄 Updating stock DB for {len(tickers)} companies...")

    for ticker in tickers:
        last_dt = get_latest_date(ticker)
        start_date = last_dt + timedelta(days=1) if last_dt else START_DATE

        df = fetch_yfinance(ticker, start_date)
        if df is not None:
            insert_prices(df, ticker)

        time.sleep(REQUEST_PAUSE_SEC)

    logging.info("✨ Stock data update finished!")

