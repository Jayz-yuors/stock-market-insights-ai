import os
from datetime import datetime
from data_fetcher import run_fetching

def log(msg):
    print(f"[{datetime.now()}] {msg}")

if __name__ == "__main__":
    log("🚀 Daily DB Stock Update Started...")
    try:
        run_fetching()
        log("✔ DB Successfully Updated!")
    except Exception as e:
        log(f"❌ Update Failed: {e}")
