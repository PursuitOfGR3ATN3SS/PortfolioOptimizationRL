#!/usr/bin/env python3
"""
filter_data.py

Reads all CSVs in SentimentAnalysis/data/, prints:
  # rows per calendar year
  busiest month inside the busiest year
Creates `SentimentAnalysis/data/filtered_analyst_ratings.csv`
  with only rows from <busiest‑year>/<busiest‑month>
"""

from pathlib import Path
from collections import Counter
import argparse
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

# Constants
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUT_FILE = DATA_DIR / "filtered_analyst_ratings.csv"
DATE_COL_CANDIDATES = [
    "date", "publishedAt", "timestamp", "datetime", "created_utc"
]

# parser for row sampling (n = ...)
parser = argparse.ArgumentParser(
    description="Filter dataset to busiest year‑month; optional random sample."
)
parser.add_argument(
    "-n", "--rows", type=int, default=None,
    help="Maximum number of rows to keep from the busiest month"
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

    year_counts = Counter()
    ym_counts = Counter()

    # count rows per year / year‑month
    for csv in tqdm(csv_files, desc="Counting"):
        df = pd.read_csv(csv, usecols=lambda c: c in DATE_COL_CANDIDATES)
        date_col = detect_date_column(df)

        dates = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_convert(None)
        years = dates.dt.year.dropna().astype(int)
        months = dates.dt.month.dropna().astype(int)

        year_counts.update(years)
        ym_counts.update(zip(years, months))

    print("Rows per year:")
    for y, n in sorted(year_counts.items()):
        print(f"  {y}: {n:,}")

    # find busiest year and month (most rows)
    busiest_year, _ = max(year_counts.items(), key=lambda x: x[1])
    busiest_month = max(
        ((y, m) for (y, m), cnt in ym_counts.items() if y == busiest_year),
        key=lambda p: ym_counts[p]
    )[1]

    print(f"\nBusiest year: {busiest_year}")
    print(f"Busiest month: {busiest_month:02d} (rows: {ym_counts[busiest_year, busiest_month]:,})")

    # create filtered CSV
    month_frames = []

    for csv in tqdm(csv_files, desc="Collecting"):
        df = pd.read_csv(csv)
        date_col = detect_date_column(df)
        df["__date"] = (
            pd.to_datetime(df[date_col], errors="coerce", utc=True)
              .dt.tz_convert(None)
        )
        mask = (
            (df["__date"].dt.year == busiest_year) &
            (df["__date"].dt.month == busiest_month)
        )
        subset = df.loc[mask].drop(columns="__date")
        if not subset.empty:
            month_frames.append(subset)

    if not month_frames:
        raise RuntimeError("No rows found for busiest month (unexpected).")

    month_df = pd.concat(month_frames, ignore_index=True)

    # optional random sample
    if ARGS.rows is not None:
        sample_n = min(ARGS.rows, len(month_df))
        month_df = month_df.sample(n=sample_n, random_state=42)
        print(f"[i] Randomly sampled {sample_n:,} rows")

    month_df.to_csv(OUT_FILE, index=False)
    print(f"\nWrote filtered CSV --> {OUT_FILE.resolve()} ({len(month_df):,} rows)")

if __name__ == "__main__":
    main()
