import os
from pymongo import MongoClient

# Load MongoDB connection from Streamlit Secrets
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise Exception("❌ MongoDB connection URL not found in Streamlit secrets!")

client = MongoClient(MONGO_URI)

# Select your database
db = client["StockDB"]  # Ensure this matches your actual DB name


def get_db():
    """
    Returns MongoDB database reference.
    Called by other modules to fetch collections.
    """
    return db
