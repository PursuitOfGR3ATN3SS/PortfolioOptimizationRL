# Portfolio Optimization with Reinforcement Learning

This project uses **Reinforcement Learning** (PPO) to optimize stock portfolio allocation and combines it with **sentiment analysis** from real-world financial news.

---

## Environment Requirements

All of the following environment requirements can be found in [env.yaml](https://github.com/PursuitOfGR3ATN3SS/PortfolioOptimizationUsingRLandNLP/blob/main/env.yaml)

`python >= 3.10`
`gymnasium`
`pandas`
`numpy`
`yfinance`
`stable-baselines3`
`tqdm`
`pip`
`transformers`
`datasets`
`kaggle`

---

## System Requirements

Developers used the following devices:

- Lenovo Thinkpad T14s Gen2
- Macbook

We utilized no GPUs while training. Sentiment analysis can be faster with gpu, it currently takes around 30 minutes. Training with 100,000 timesteps takes around 2 minutes.

---

## Command Line Arguments

The following arguments are (optional) used in the program when running `main.py`.
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num_portfolio_stocks` | `int` | `20` | Number of stocks to include in the portfolio. |
| `--start_date` | `str` | `"2019-05-01"` | Starting date for the portfolio's timeframe (format: YYYY-MM-DD). |
| `--end_date` | `str` | `"2020-03-25"` | Ending date for the portfolio's timeframe (format: YYYY-MM-DD). |
| `--stock_index` | `str` | `"nasdaq"` | The stock index to fetch tickers from. Options: `"nasdaq"`, `"nyse"`, or `"all"`. |
| `--random_seed` | `int` | `42` | Random seed for reproducibility. |
| `--cache_dir` | `str` | `"./cache/"` | Directory path to store cached content. |
| `--use_sentiment` | `int` | `0` | Whether to include news sentiment in the optimization strategy. Set to `1` to use sentiment, or `0` to ignore. |
| `--best_model_path` | `str` | `"./cache/best_model"` | Directory path to store the best-performing model. |
| `--eval_dir` | `str` | `"./cache/eval"` | Directory path to store evaluation callback results. |

### Example Usage

```bash
python main.py --num_portfolio_stocks 25 --start_date 2020-01-01 --end_date 2021-01-01 --stock_index nyse --use_sentiment 1
```

---

## Market Data Pipeline

1. Fetch current list of tickers available on NASDAQ, NYSE, or both
2. Verify presence of tickers in yfinance api
3. All verified tickers are treated as `valid_tickers`

Results are [cached](https://github.com/PursuitOfGR3ATN3SS/PortfolioOptimizationUsingRLandNLP/tree/main/cache), both the valid and invalid tickers, to speed up developement.

4. Samples `n` tickers from `valid_tickers` and treats them as the `portfolio_stocks`

- The tickers are re-validated when sampled due to issues with some tickers getting past verification 1st time

---

## RL Environment

The environment uses OpenAI's Gymnasium and stable-baseline3 for logging, evaluation, and model implementation.

#### Portfolio

The [portfolio](https://github.com/PursuitOfGR3ATN3SS/PortfolioOptimizationUsingRLandNLP/blob/main/src/env.py) is the both the stock portfolio and market, as it considers assets and market conditions.

#### Agent

The agent uses a Multi-Layer Perceptron policy and PPO algorithm to optimize asset allocation.

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

### Makefile Commands

Run these commands from the project root:

- `make`: Show available commands
- `make get-data`: Download the stock news dataset from Kaggle
- `make filter-data n=10000`: Filter to the busiest month and keep 10,000 random rows
- `make get-sentiments`: Run FinBERT sentiment pipeline on filtered (or full) dataset
- `make sentiment n=10000`: Run full pipeline: download --> filter --> analyze

---

### Requirements

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

#### Note: You must download the data from kaggle, through API, then run sentiment pipeline separtely of the market data pipeline. After creating the `daily_sentiment.csv` file in the `data` directory, you can run `main.py`.

---

## Results

| Sentiment | Period                   | Cum. Return | Avg Return | Volatility | Sharpe (Simple) | Sharpe (Log) |
| --------- | ------------------------ | ----------- | ---------- | ---------- | --------------- | ------------ |
| No        | 2019-05-01 to 2020-03-25 | 0.8391      | -0.0097    | 0.0617     | -0.6289         | -0.7474      |
| Yes       | 2019-05-01 to 2020-03-25 | 0.8574      | -0.0084    | 0.0598     | -0.5619         | -0.6804      |
| No        | 2019-05-01 to 2019-12-31 | 1.0140      | 0.0024     | 0.0090     | 0.6957          | 0.6839       |
| Yes       | 2019-05-01 to 2019-12-31 | 1.0302      | 0.0050     | 0.0088     | 1.5024          | 1.4964       |

---

## Future Work

1. More algorithms outside of PPO
2. Implementation of PPO (not using stable-baseline3)
3. Larger range of sentiment data used.
4. Store sentiment data and cached data in database.
5. More robust evaluation pipeline.
6. Better logging.
7. Use of technical indicators in data.

---

## Resources

- [Average portfolio size](https://www.investopedia.com/ask/answers/05/optimalportfoliosize.asp#:~:text=The%20more%20equities%20you%20hold,portfolio%20of%20only%20two%20stocks.)
- [All U.S. Stock Tickers](https://github.com/rreichel3/US-Stock-Symbols/blob/main/all/all_tickers.txt)
- [Deep Reinforcement Learning for Optimal Portfolio Allocation: A Comparative
  Study with Mean-Variance Optimization](https://icaps23.icaps-conference.org/papers/finplan/FinPlan23_paper_4.pdf) -[News data for sentiment analysis](https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests/data)
- [FinBERT Hugging Face docs](https://huggingface.co/ProsusAI/finbert)
- [Stable-baselines3 docs](https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html)
