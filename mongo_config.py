import os
from pymongo import MongoClient

def get_db():
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DBNAME", "Nifty50")

    if not mongo_uri:
        raise RuntimeError("❌ MONGO_URI not found in Streamlit Secrets!")

    client = MongoClient(mongo_uri)
    return client[db_name]
