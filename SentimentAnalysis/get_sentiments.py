#!/usr/bin/env python3
"""
get_sentiments.py

Runs FinBERT on news headlines/content, combines average sentiment
  per calendar day (float in [-1, 1])
Fills missing days by forward filling most recent sentiment
Outputs outputs/daily_sentiment.csv
  columns: date, sentiment
"""

from pathlib import Path
from collections import defaultdict
import pandas as pd
from tqdm.autonotebook import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Constants
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUT_DIR = ROOT_DIR / "outputs"
OUT_FILE = OUT_DIR / "daily_sentiment.csv"
MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 32
TOKEN_MAX = 128
TEXT_COL_CANDIDATES = ["title", "headline", "content", "text", "body", "message"]
DATE_COL_CANDIDATES = ["date", "publishedAt", "timestamp", "datetime", "created_utc"]

# Load FinBERT model and tokenizer
def make_finbert():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, truncation=True)

# Find usable text and date columns in DataFrame
def detect_columns(df: pd.DataFrame):
    text_col = next((c for c in TEXT_COL_CANDIDATES if c in df.columns), None)
    date_col = next((c for c in DATE_COL_CANDIDATES if c in df.columns), None)
    if not text_col or not date_col:
        raise ValueError("Suitable text/date columns not found.")
    return text_col, date_col

# Split text rows into chunks for batch processing
def iter_batches(series, size=BATCH_SIZE):
    buf = []
    for txt in series:
        buf.append(txt)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf

def main():
    OUT_DIR.mkdir(exist_ok=True)
    senti_pipe = make_finbert()

    daily_scores = defaultdict(list)

    # filtered file if it exists, otherwise process all CSVs
    filtered = DATA_DIR / "filtered_analyst_ratings.csv"
    csv_files = [filtered] if filtered.exists() else list(DATA_DIR.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV files found to process.")

    # process each file
    for csv in csv_files:
        print(f"[*] Processing {csv.name} …")
        df = pd.read_csv(csv).dropna(how="all")
        text_col, date_col = detect_columns(df)
        df = df[[text_col, date_col]].dropna()

        # normalise to datetime.date
        df["__date"] = (
            pd.to_datetime(df[date_col], errors="coerce", utc=True)
              .dt.tz_convert(None)
              .dt.date
        )
        df = df.dropna(subset=["__date"])

        # sentiment analysis in batches
        for batch in tqdm(iter_batches(df[text_col]), total=len(df)//BATCH_SIZE + 1):
            preds = senti_pipe(batch, max_length=TOKEN_MAX)
            signed = [
                1 if p["label"] == "positive"
                else -1 if p["label"] == "negative"
                else 0
                for p in preds
            ]
            for d, s in zip(df["__date"].iloc[:len(signed)], signed):
                daily_scores[d].append(s)

    # combine to daily mean sentiment (float)
    rows = [
        (pd.to_datetime(d), sum(vals) / len(vals))
        for d, vals in daily_scores.items()
    ]
    out_df = (
        pd.DataFrame(rows, columns=["date", "sentiment"])
          .set_index("date")
          .sort_index()
    )

    # fill missing days (forward fill)
    # UNCOMMENT
    # full_range = pd.date_range(out_df.index.min(), out_df.index.max(), freq="D")
    # out_df = out_df.reindex(full_range).ffill()

    # final formatting
    out_df.reset_index(inplace=True)
    out_df.rename(columns={"index": "date"}, inplace=True)
    out_df["date"] = out_df["date"].dt.strftime("%Y-%m-%d")

    out_df.to_csv(OUT_FILE, index=False)
    print(f"Wrote {len(out_df)} rows --> {OUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
