from mongo_config import get_db

# ---------- Default Nifty50 Companies ----------
DEFAULT_COMPANIES = [
    ("Reliance Industries", "RELIANCE.NS"),
    ("HDFC Bank", "HDFCBANK.NS"),
    ("Tata Consultancy Services", "TCS.NS"),
    ("Bharti Airtel", "BHARTIARTL.NS"),
    ("ICICI Bank", "ICICIBANK.NS"),
    ("State Bank of India", "SBIN.NS"),
    ("Infosys", "INFY.NS"),
    ("Hindustan Unilever", "HINDUNILVR.NS"),
    ("Life Insurance Corporation of India", "LICI.NS"),
    ("Bajaj Finance", "BAJFINANCE.NS"),
    ("ITC", "ITC.NS"),
    ("Larsen & Toubro", "LT.NS"),
    ("Maruti Suzuki India", "MARUTI.NS"),
    ("HCL Technologies", "HCLTECH.NS"),
    ("Sun Pharmaceutical", "SUNPHARMA.NS"),
    ("Kotak Mahindra Bank", "KOTAKBANK.NS"),
    ("Mahindra & Mahindra", "M&M.NS"),
    ("UltraTech Cement", "ULTRACEMCO.NS"),
    ("Axis Bank", "AXISBANK.NS"),
    ("NTPC Limited", "NTPC.NS"),
    ("Titan Company", "TITAN.NS"),
    ("Bajaj Finserv", "BAJAJFINSV.NS"),
    ("Hindustan Aeronautics", "HAL.NS"),
    ("Oil & Natural Gas", "ONGC.NS"),
    ("Adani Ports & SEZ", "ADANIPORTS.NS"),
    ("Bharat Electronics", "BEL.NS"),
    ("Wipro", "WIPRO.NS"),
    ("JSW Steel", "JSWSTEEL.NS"),
    ("Tata Motors", "TATAMOTORS.NS"),
    ("Asian Paints", "ASIANPAINT.NS"),
    ("Coal India", "COALINDIA.NS"),
    ("Nestlé India", "NESTLEIND.NS"),
    ("Grasim Industries", "GRASIM.NS"),
    ("Hindalco Industries", "HINDALCO.NS"),
    ("Tata Steel", "TATASTEEL.NS"),
    ("Ambuja Cement", "AMBUJACEM.NS"),
]


def insert_companies():
    """Auto-insert default companies safely (idempotent)."""
    db = get_db()
    col = db["companies"]

    inserted_count = 0

    for name, ticker in DEFAULT_COMPANIES:
        result = col.update_one(
            {"ticker": ticker},
            {"$set": {"name": name, "ticker": ticker}},
            upsert=True
        )
        if result.upserted_id:
            inserted_count += 1

    print(f"✔ Company list synced! (Newly inserted: {inserted_count})")


if __name__ == "__main__":
    insert_companies()
