# Sentiment Analysis Documentation

## Makefile Commands

Run these commands from the project root:

- `make`: Shows available targets
- `make get-data`: Download dataset from Kaggle
- `make filter-data n=10000`: Filter to busiest month and sample 10,000 rows (optional `n`)
- `make get-sentiments`: Run FinBERT sentiment pipeline dataset from data
- `make sentiment n=10000`: Run full pipeline (download --> filter --> analyze)

---

## Scripts

### get_data.py

Downloads the **"Massive Stock News Analysis DB for NLP/backtests"** from Kaggle:  
`https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests`

Places into `/SentimentAnalysis/data/`

Requires:

- Kaggle credentials file at `~/.kaggle/kaggle.json`, or
- Environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`

---

### filter_data.py

Filters the dataset to the **busiest month** within the **busiest year**  
(based on number of news records).

- Optional: Sample fixed number of random rows from that month using the `-n` or `n=...` argument.
- Creates `/SentimentAnalysis/data/filtered_analyst_ratings.csv`

The file will be used as input for the sentiment pipeline if it exists.

---

### get_sentiments.py

Runs **FinBERT** (ProsusAI/finbert) on all headlines/text content in the dataset.

- Combines **daily average sentiment scores** from headlines:
  - `+1 = positive`
  - `-1 = negative`
  - `0 = neutral`
- Daily scores are **averaged** (as float) and forward-filled across calendar dates.

Creates `/SentimentAnalysis/outputs/daily_sentiment.csv`

---

## Output File

### `daily_sentiment.csv`

One unique row per day. If a date has no headlines, the last known sentiment is forward-filled.

| date       | sentiment |
|------------|-----------|
| 2019-10-25 | -0.67     |
| 2019-10-26 | -0.42     |
| 2019-10-27 | -0.42     |
| 2019-10-28 |  0.18     |

- `date`: Calendar date in `YYYY-MM-DD` format
- `sentiment`: Average sentiment (float in range [-1, 1])

---
