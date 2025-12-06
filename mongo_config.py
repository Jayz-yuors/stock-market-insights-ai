import os
from pymongo import MongoClient

mongo_uri = (
    f"mongodb+srv://{os.getenv('MONGO_USER')}:{os.getenv('MONGO_PASS')}"
    f"@{os.getenv('MONGO_CLUSTER')}/?retryWrites=true&w=majority"
)

client = MongoClient(mongo_uri)
db = client[os.getenv("MONGO_DBNAME")]