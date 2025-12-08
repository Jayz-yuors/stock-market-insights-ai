import os
import sys

# Ensure repo root is available in import path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from data_fetcher import run_fetching

if __name__ == "__main__":
    print("🚀 Running daily DB update…")
    run_fetching()
    print("✨ DB sync completed successfully!")
