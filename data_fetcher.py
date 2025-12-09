from mongo_config import get_db
from insert_companies import insert_companies
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, date
import logging
import time

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

START_DATE = date(2015, 1, 1)
REQUEST_PAUSE_SEC = 0.3
SAFETY_LOOKBACK_DAYS = 7  # Helps recover missing gaps from Yahoo


# --------------------------
# Safe numeric conversion
# --------------------------
def safe_float(v):
    try:
        return float(v)
    except:
        return None


def safe_int(v):
    try:
        return int(float(v))
    except:
        return 0


# --------------------------
# Company List Loader
# --------------------------
def get_company_list():
    db = get_db()
    companies = list(db["companies"].find({}, {"ticker": 1, "_id": 0}))

    if not companies:
        logging.warning("⚠ No companies in DB — inserting default list")
        insert_companies()
        companies = list(db["companies"].find({}, {"ticker": 1, "_id": 0}))
        logging.info("✔ Default companies inserted!")

    return [c["ticker"] for c in companies]


# --------------------------
# Get latest stored date
# --------------------------
def get_latest_date(ticker):
    db = get_db()
    rec = db["stock_prices"].find_one(
        {"ticker": ticker},
        sort=[("date", -1)]
    )
    if rec:
        return rec["date"].date()  # Convert MongoDB datetime → date only
    return None


# --------------------------
# Fetch new data from Yahoo
# --------------------------
def fetch_yfinance(ticker, start_date):
    today = datetime.now().date()

    if start_date >= today:
        logging.info(f"{ticker}: Already updated — no fetch needed")
        return None

    logging.info(f"{ticker}: Fetching from {start_date} to {today} …")

    df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=today.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if df.empty:
        logging.warning(f"{ticker}: 🚫 No new data received")
        return None

    df.index = pd.to_datetime(df.index, utc=False)
    logging.info(f"{ticker}: ✔ Downloaded {len(df)} rows")
    return df


# --------------------------
# Insert New Data into Mongo
# --------------------------
def insert_prices(df, ticker):
    db = get_db()
    sp = db["stock_prices"]

    inserted = 0
    for ts, row in df.iterrows():
        trade_date = ts.to_pydatetime()

        # ensure BSON-compatible type
        trade_date = trade_date.replace(tzinfo=None)

        doc = {
            "ticker": ticker,
            "date": trade_date,
            "open": safe_float(row.get("Open")),
            "high": safe_float(row.get("High")),
            "low": safe_float(row.get("Low")),
            "close": safe_float(row.get("Close")),
            "volume": safe_int(row.get("Volume")),
        }

        sp.update_one(
            {"ticker": ticker, "date": trade_date},
            {"$set": doc},
            upsert=True
        )
        inserted += 1

    logging.info(f"{ticker}: 🔄 {inserted} rows inserted/updated")



# --------------------------
# MAIN DAILY UPDATER
# --------------------------
def run_fetching():
    tickers = get_company_list()
    logging.info(f"🚀 Running DB Sync for {len(tickers)} tickers")

    today = datetime.now().date()

    for ticker in tickers:
        last_dt = get_latest_date(ticker)

        # If DB has history, fetch little overlap for fixes
        if last_dt:
            start_date = last_dt - timedelta(days=SAFETY_LOOKBACK_DAYS)
        else:
            start_date = START_DATE

        df = fetch_yfinance(ticker, start_date)

        if df is not None:
            insert_prices(df, ticker)

        time.sleep(REQUEST_PAUSE_SEC)

    logging.info("✨ DB Sync Finished Successfully!")

