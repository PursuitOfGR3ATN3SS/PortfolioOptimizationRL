#!/usr/bin/env python3
"""
filter_data2.py

Reads all CSVs in SentimentAnalysis/data/, prints:
  # rows per calendar year
Creates `SentimentAnalysis/data/filtered_analyst_ratings.csv`
"""

from pathlib import Path
import argparse
import pandas as pd
from tqdm.autonotebook import tqdm

# Constants
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUT_FILE = DATA_DIR / "filtered_analyst_ratings.csv"
DATE_COL_CANDIDATES = [
    "date", "publishedAt", "timestamp", "datetime", "created_utc"
]
# Specified range of dates
DEFAULT_START = "2019-05-01"
DEFAULT_END = "2020-3-31"

parser = argparse.ArgumentParser(
    description="Filter dataset by date range; optional random sample."
)
parser.add_argument( # specify through commandline
    "--start", type=str, default=DEFAULT_START,
    help=f"Start date (YYYY-MM-DD). Default: {DEFAULT_START}"
)
parser.add_argument(
    "--end", type=str, default=DEFAULT_END,
    help=f"End date (YYYY-MM-DD). Default: {DEFAULT_END}"
)
parser.add_argument(
    "-n", "--rows", type=int, default=None,
    help="Maximum number of rows to keep (randomly sampled) in the range"
)
ARGS = parser.parse_args()

# find usable date column from known candidates
def detect_date_column(df: pd.DataFrame) -> str:
    for c in DATE_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError("No date column found in dataframe.")

def main():
    # get all CSVs in data directory
    csv_files = list(DATA_DIR.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {DATA_DIR}. Run get_data.py first.")

    start_date = pd.to_datetime(ARGS.start)
    end_date   = pd.to_datetime(ARGS.end)

    print(f"Filtering rows from {start_date.date()} to {end_date.date()}")

    frames = []

    # count rows per year / year‑month
    for csv in tqdm(csv_files, desc="Collecting"):
        df = pd.read_csv(csv)
        date_col = detect_date_column(df)

        df["__date"] = (
            pd.to_datetime(df[date_col], errors="coerce", utc=True)
              .dt.tz_convert(None)
        )
        mask = (df["__date"] >= start_date) & (df["__date"] <= end_date)
        subset = df.loc[mask].drop(columns="__date")
        if not subset.empty:
            frames.append(subset)

    if not frames:
        raise RuntimeError("No rows found in the specified date range.")

    combined = pd.concat(frames, ignore_index=True)

    # optional random sample
    if ARGS.rows is not None:
        sample_n = min(ARGS.rows, len(combined))
        combined = combined.sample(n=sample_n, random_state=42)
        print(f"[i] Randomly sampled {sample_n:,} rows")

    combined.to_csv(OUT_FILE, index=False)
    print(f"Wrote filtered CSV --> {OUT_FILE.resolve()} ({len(combined):,} rows)")

if __name__ == "__main__":
    main()
