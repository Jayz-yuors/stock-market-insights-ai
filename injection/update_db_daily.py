import os
import sys

# Ensure root project folder is in Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(ROOT_DIR)

from data_fetcher import run_fetching

if __name__ == "__main__":
    print("🚀 Running daily DB update…")
    run_fetching()
    print("✨ DB sync completed successfully!")
