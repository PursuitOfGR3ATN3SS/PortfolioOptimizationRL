# Portfolio Optimization with Reinforcement Learning

This project uses **Reinforcement Learning** (PPO) to optimize stock portfolio allocation and combines it with **sentiment analysis** from real-world financial news.

---

## Sentiment Analysis

We use **FinBERT** (a financial-domain BERT model) to classify the sentiment of stock-related news articles.

The sentiment signal is then used to inform portfolio decisions, reflecting the daily tone of the market based on recent news coverage.

The pipeline:

1. Downloads a large dataset of historical stock news headlines
2. Filters to the busiest month
3. Applies FinBERT sentiment analysis
4. Aggregates into a daily sentiment time series

Output:

- `daily_sentiment.csv` — one row per day with an average sentiment score (float between -1 and 1)

---

## Makefile Commands

Run these commands from the project root:

- `make`: Show available commands
- `make get-data`: Download the stock news dataset from Kaggle
- `make filter-data n=10000`: Filter to the busiest month and keep 10,000 random rows
- `make get-sentiments`: Run FinBERT sentiment pipeline on filtered (or full) dataset
- `make sentiment n=10000`: Run full pipeline: download --> filter --> analyze

---

## Requirements

See `env.yaml` to set up the Conda environment:

```bash
conda env create -f env.yaml
conda activate RL
```

Also make sure to configure your Kaggle API credentials:

1. Go to `https://www.kaggle.com/account`
2. Create a new API token
3. Place the downloaded `kaggle.json` file into:

```bash
~/.kaggle/kaggle.json
```

Or set the following environment variables:

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
```
