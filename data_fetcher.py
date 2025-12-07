from mongo_config import get_db
from insert_companies import insert_companies
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
import time

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

START_DATE = datetime(2015, 1, 1)
REQUEST_PAUSE_SEC = 0.3
SAFETY_LOOKBACK_DAYS = 7    # <-- Important Recovery Feature


# ---------------------------------
# NUMERIC FIXERS
# ---------------------------------
def safe_float(v):
    try:
        return float(v) if v is not None else None
    except:
        return None


def safe_int(v):
    try:
        return int(float(v)) if v is not None else 0
    except:
        return 0


# ---------------------------------
# COMPANY LIST AUTO-SETUP
# ---------------------------------
def get_company_list():
    db = get_db()
    companies = list(db["companies"].find({}, {"ticker": 1, "_id": 0}))

    if not companies:
        logging.warning("⚠ No companies found — inserting defaults")
        insert_companies()
        companies = list(db["companies"].find({}, {"ticker": 1, "_id": 0}))
        logging.info("✔ Default companies inserted")

    return [c["ticker"] for c in companies]


# ---------------------------------
# HELPER: MOST RECENT DATE IN DB
# ---------------------------------
def get_latest_date(ticker):
    db = get_db()
    rec = db["stock_prices"].find_one(
        {"ticker": ticker},
        sort=[("date", -1)]
    )
    return rec["date"] if rec else None


# ---------------------------------
# FETCH DATA
# ---------------------------------
def fetch_yfinance(ticker, start_date):
    today = datetime.now()

    if start_date.date() >= today.date():
        logging.info(f"{ticker}: Already updated — no fetch needed")
        return None

    logging.info(f"{ticker}: Fetching from {start_date.date()} ...")

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
        logging.warning(f"{ticker}: 🚫 No new data from Yahoo!")
        return None

    df.index = pd.to_datetime(df.index, utc=False)
    logging.info(f"{ticker}: ✔ {len(df)} days fetched")

    return df


# ---------------------------------
# INSERT INTO DB
# ---------------------------------
def insert_prices(df, ticker):
    db = get_db()
    sp = db["stock_prices"]

    count = 0
    for ts, row in df.iterrows():
        if pd.isna(ts):
            continue

        doc = {
            "ticker": ticker,
            "date": ts.to_pydatetime(),
            "open": safe_float(row.get("Open")),
            "high": safe_float(row.get("High")),
            "low": safe_float(row.get("Low")),
            "close": safe_float(row.get("Close")),
            "volume": safe_int(row.get("Volume")),
        }

        sp.update_one(
            {"ticker": ticker, "date": doc["date"]},
            {"$set": doc},
            upsert=True
        )
        count += 1

    logging.info(f"{ticker}: 🔄 {count} rows inserted/updated")


# ---------------------------------
# MAIN UPDATE CALL
# ---------------------------------
def run_fetching():
    tickers = get_company_list()
    logging.info(f"🚀 Syncing DB for {len(tickers)} tickers")

    today = datetime.now().date()

    for ticker in tickers:
        last_dt = get_latest_date(ticker)

        # Fallback: first full fetch
        if not last_dt:
            start_date = START_DATE
        else:
            # Always fetch extra 7-days → repairs missing gaps
            start_date = (last_dt - timedelta(days=SAFETY_LOOKBACK_DAYS))

        df = fetch_yfinance(ticker, start_date)

        if df is not None:
            insert_prices(df, ticker)

        time.sleep(REQUEST_PAUSE_SEC)

    logging.info("✨ Daily DB Sync Complete!")
