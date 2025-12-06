from mongo_config import get_db

db = get_db()

def check_companies():
    print(list(db["companies"].find({}, {"_id":0})))

def check_price_counts():
    for ticker in db["companies"].distinct("ticker"):
        count = db["stock_prices"].count_documents({"ticker": ticker})
        print(f"{ticker}: {count}")

if __name__ == "__main__":
    check_companies()
    check_price_counts()
