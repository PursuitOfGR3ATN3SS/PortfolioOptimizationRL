#!/usr/bin/env python3
"""
get_data.py

Downloads “Massive Stock News Analysis DB for NLP/backtests” dataset,
  extracts it into SentimentAnalysis/data/
"""

import zipfile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

# Constants
DATASET = "miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests"
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
ZIP_PATH = DATA_DIR / "dataset.zip"

# Download Kaggle dataset using API
def download_dataset():
    print("[*] Authenticating with Kaggle...")
    api = KaggleApi()
    api.authenticate()

    print("[*] Downloading dataset...")

    api.dataset_download_files(DATASET, path=str(DATA_DIR), quiet=False, force=False)

    zips = list(DATA_DIR.glob("*.zip"))
    if not zips:
        raise FileNotFoundError("Dataset zip not found after download.")

    zips[0].rename(ZIP_PATH)
    print(f"[+] Saved to {ZIP_PATH}")

# Extract contents, then delete ZIP
def extract_dataset():
    print("[*] Extracting...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DATA_DIR)
    ZIP_PATH.unlink() # remove zip to save space
    print(f"[+] Extracted to {DATA_DIR.resolve()}")

def main():
    DATA_DIR.mkdir(exist_ok=True)
    if not any(DATA_DIR.glob("*.csv")):
        download_dataset()
        extract_dataset()
    else:
        print(f"[i] Data already present in {DATA_DIR}. Skipping download.")

    print("Download complete.")

if __name__ == "__main__":
    main()
