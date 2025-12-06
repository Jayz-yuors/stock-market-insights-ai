from mongo_config import get_db

def insert_companies(companies):
    db = get_db()
    col = db["companies"]

    for name, ticker in companies:
        col.update_one(
            {"ticker": ticker},
            {"$set": {"name": name, "ticker": ticker}},
            upsert=True
        )

    print("Companies inserted successfully!")


if __name__ == "__main__":
    companies_list = [
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
    insert_companies(companies_list)
