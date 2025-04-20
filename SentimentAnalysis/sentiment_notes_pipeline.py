"""
sentiment_notes_pipeline.py

## Data Location
Unzip file from https://www.sec.gov/data-research/sec-markets-data/financial-statement-notes-data-sets
Into: /SentimentAnalysis/<year>_<month>_notes

Runs FinBert sentiment, exports two CSVs.

SEC data structure (what is used):
  txt.tsv:
    has raw content from filings
    each row represents a disclosure block (like a paragraph or table from a 10-K or 10-Q filing)
      - adsh
        Accession number
      - tag
        Name of XBRL element (eg. us-gaap:AccountingPoliciesTextBlock)
      - ddate
        date of disclosure
      - value
        textual content (this is sent to FinBERT!)

  ren.tsv:
    rendering info to find how this is displayed in the SEC viewer
    each adsh is matched with one or more menu categories
      - adsh
        Accession number
      - menucat
        Menu category
        F = Financial Statements, N = Notes, M = MD&A (Management Discussion & Analysis)

  sub.tsv:
    contains metadata for each filing
      - adsh
        Accession number
      - cik
        Central Index Key (UID for company)
      - name
        Company name
      - fp
        Filing period (Q1, FY, etc.)
      - fye
        Fiscal year-end (1231 for Dec 31)
"""

import os
from pathlib import Path
from typing import Tuple, Iterator, List, Dict

import pandas as pd
from tqdm.autonotebook import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline # FinBERT - pytorch based model

"""
CONFIG VARIABLES
"""

# Path to the directory that holds txt.tsv / ren.tsv / sub.tsv
DATA_DIR = Path(__file__).parent / "2025_03_notes" # Can change filename here (different months / years)
OUT_DIR = Path(__file__).parent / "outputs" # where the CSVs will go
BATCH_SIZE = 16
NOTE_TOKEN_LIMIT = 512
MODEL_NAME = "ProsusAI/finbert"



# reads a tsv file into a pandas dataframe
# makes sure to apply default for SEC data
def load_tsv(path: Path, **kwargs) -> pd.DataFrame:
    """Read a .tsv into a DataFrame with SEC‑friendly defaults."""
    return pd.read_csv(
        path,
        sep="\t",
        low_memory=False,
        dtype=str,
        na_values=["", "NULL"], # all columns as string, treats 'NULL' as NaN
        **kwargs,
    )

# splits long list of text inputs into smaller batches
# each of those gets sent to senti_pipe()
def iter_chunks(series: pd.Series, batch: int = BATCH_SIZE) -> Iterator[List[str]]:
    buf: List[str] = []
    for txt in series:
        buf.append(txt)
        if len(buf) == batch:
            yield buf; buf = []
    if buf:
        yield buf



# loads tsv files
# TSV files: txt.tsv (facts), ren.tsv (rendering), sub.tsv (submission metadata)
def load_dataframes() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {"txt.tsv", "ren.tsv", "sub.tsv"}
    missing = required - {p.name for p in DATA_DIR.iterdir()}
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {DATA_DIR}. "
            "Make sure you unzipped the notes data here."
        )

    txt = load_tsv(DATA_DIR / "txt.tsv", usecols=["adsh", "tag", "ddate", "value"])
    ren = load_tsv(DATA_DIR / "ren.tsv", usecols=["adsh", "menucat"])
    sub = load_tsv(DATA_DIR / "sub.tsv", usecols=["adsh", "cik", "name", "fp", "fye"])
    return txt, ren, sub

# filters raw text data to keep only note sections from 'Notes' menu category
# focuses on 'TextBlock' - has full narrative disclosures
def filter_note_blocks(txt: pd.DataFrame, ren: pd.DataFrame) -> pd.DataFrame:
    notes_filings = ren.loc[ren["menucat"] == "N", "adsh"].unique()
    txt = txt[txt["adsh"].isin(notes_filings)]
    txt = txt[txt["tag"].str.endswith("TextBlock", na=False)]
    txt = txt.dropna(subset=["value"]).reset_index(drop=True)
    print(f"[filter] note TextBlocks kept: {len(txt):,}")
    return txt

# loads FinBERT model and tokenizer as Hugging Face inference pipeline
def make_sentiment_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, truncation=True)

# applies sentiment analysis to each note block
# adds 'sentiment' and 'score' columns to DataFrame
def run_sentiment(df: pd.DataFrame, senti_pipe) -> pd.DataFrame:
    sentiments, scores = [], []
    for chunk in tqdm(iter_chunks(df["value"]), total=len(df)//BATCH_SIZE + 1):
        preds = senti_pipe(chunk, max_length=NOTE_TOKEN_LIMIT)
        sentiments += [p["label"] for p in preds]
        scores += [p["score"] for p in preds]
    df = df.assign(sentiment=sentiments, score=scores)
    return df

# combines sentiment from per-note to per-filing
# calculate signed average score, bullish/bearish/neutral
def aggregate(df: pd.DataFrame, sub: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scale = {"positive": 1, "negative": -1, "neutral": 0}
    df["score_signed"] = df["sentiment"].map(scale)

    per_filing = (
        df.groupby("adsh", as_index=False)
          .agg(
              n_notes=("sentiment", "size"),
              pos=("sentiment", lambda s: (s=="positive").sum()),
              neg=("sentiment", lambda s: (s=="negative").sum()),
              neu=("sentiment", lambda s: (s=="neutral").sum()),
              mean_signed=("score_signed", "mean"),
          )
          .merge(sub, on="adsh", how="left")
    )

    per_filing["overall_sentiment"] = pd.cut(
        per_filing["mean_signed"],
        bins=[-1, -0.05, 0.05, 1],
        labels=["bearish", "neutral", "bullish"],
        include_lowest=True,
    )
    return df, per_filing

def main():
    # folder exists
    OUT_DIR.mkdir(exist_ok=True)

    # load, filter data
    txt, ren, sub = load_dataframes()
    notes = filter_note_blocks(txt, ren)

    # run FinBERT on notes
    senti_pipe = make_sentiment_pipeline()
    notes_sent = run_sentiment(notes, senti_pipe)

    # aggregate and save results
    per_note, per_filing = aggregate(notes_sent, sub)
    per_note.to_csv(OUT_DIR / "2025_03_per_note.csv", index=False)
    per_filing.to_csv(OUT_DIR / "2025_03_per_filing.csv", index=False)

    print("[done] CSVs written to", OUT_DIR.resolve())

if __name__ == "__main__":
    main()
