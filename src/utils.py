import argparse
import yfinance as yf
import numpy as np
from numpy import ndarray
import os, time, random, requests
import pandas as pd
import matplotlib.pyplot as plt

def parse_cl_args():
  parser = argparse.ArgumentParser(description="A stock portfolio that leverages reinforcement learning to reach optimality.")
  parser.add_argument(
    "--num_portfolio_stocks",
    type=int,
    default=20,
    help="Number of stocks to include in the stock portfolio"
  )
  parser.add_argument(
    "--start_date",
    type=str,
    default="2019-05-01",
    help="Starting date for portfolio timeframe",
  )
  parser.add_argument(
    "--end_date",
    type=str,
    default="2020-03-25",
    help="Ending date for portfolio timeframe",
  )
  parser.add_argument(
    "--stock_index",
    type=str,
    default="nasdaq",
    choices=["nasdaq", "nyse", "all"],
    help="The stock index to fetch tickers from",
  )
  parser.add_argument(
    "--random_seed",
    type=int,
    default=42,
    help="Random seed for reproducability"
  )
  parser.add_argument(
    "--cache_dir",
    type=str,
    default="./cache/",
    help="Where to store cached content"
  )
  parser.add_argument(
    "--use_sentiment",
    type=bool,
    default=False,
    choices=[True, False],
    help="Determines whether or not add news sentiment into optimization strategy"
  )
  parser.add_argument(
    "--best_model_path",
    type=str,
    default="./cache/model/best_model/best_model.zip",
    help="Where to store the best model",
  )
  parser.add_argument(
    "--eval_dir",
    type=str,
    default="./cache/model/best_model/best_model.zip",
    help="Where to store EvalCallback results",
  )
  return parser.parse_args()


def return_cl_args(args) -> tuple[any]:
  return args.num_portfolio_stocks, args.start_date, args.end_date, args.stock_index, args.random_seed, args.cache_dir, args.use_sentiment, args.best_model_path, args.eval_dir


def get_all_tickers(cached_path: str | None, index:str="all") -> list[str]:
  """
  Gets all of the stock tickers in the United States per [US-Stock-Symbols](https://github.com/rreichel3/US-Stock-Symbols/tree/main).

  Args:
    cached_path (str | None): File containing tickers that have already been parsed.
    index (str): Index to collect tickers from.
    - Default is all
    - Allows
      - all, nasdaq, nyse
  Returns:
    list[str] -> All of the parsed tickers.
  """
  url: str
  file_content: str
  status_code: int = 404
  skip = False

  if cached_path is None:
    skip = True

  if os.path.exists(cached_path) and not skip:
    with open(cached_path, "r") as file:
      file_content = file.read()
      status_code = 200
  else:
    match (index):
      case "nasdaq":
        url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_tickers.txt"
      case "nyse":
        url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nyse/nyse_tickers.txt"
      case _: # Default
        url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"

    response = requests.get(url=url)
    status_code = response.status_code
    if status_code == 200:
      file_content = response.text

  if status_code == 200:
    # Successful request or file read
    return file_content.splitlines()

  print(f"Request to recieve ticker data failed per {status_code}")
  return [] # Unsuccessful request or file read


def is_valid_ticker(ticker: str) -> bool:
  """
  Confirms whether a ticker is in the yfinance api.

  Args:
    ticker (str): Ticker name

  Returns:
    bool -> True if the ticker is within the yfinance api, False otherwise.
  """
  try:
    info = yf.Ticker(ticker).info
    return 'regularMarketPrice' in info and info['regularMarketPrice'] is not None
  except:
    print(f"{ticker} is not available in yfinance.")
    return False


def validate_tickers(
    tickers: list[str],
    start_date:str,
    end_date:str,
    batch_size: int = 25,
    interval: str = "1d",
    delay: int = 5,
) -> tuple[list[str]]:
    """
    Removes any tickers that are not available in the yfinance API.

    Args:
      tickers (list[str]): Tickers to be evaluated.
      batch_size (int): Number of tickers to request per call.
      interval (str): Sampling interval, e.g. '1d', '1m'.
      delay (int): Delay between batches (in seconds).

    Returns:
      list[str]: Tickers that return valid, non-empty price data.
      list[str]: Tickst that return invalid price data
    """
    n = len(tickers)
    assert batch_size <= n, f"The given batch size ({batch_size}) is larger than the number of tickers ({n})."

    valid_tickers = []
    invalid_tickers = []

    tickers = [t for t in tickers if not t.endswith(("W", "R", "U"))] # Omit warrants, rights, and units

    for i in range(0, n, batch_size):
      batch = tickers[i:i + batch_size]
      try:
        data = yf.download(
          batch,
          interval=interval,
          start=start_date,
          end=end_date,
          progress=False,
          threads=True
        )

        if isinstance(data.columns, pd.MultiIndex) and "Close" in data.columns.levels[0]:
          close_data = data["Close"]
          for ticker in close_data.columns:
            series = close_data[ticker]
            if series.dropna().shape[0] > 0:
              valid_tickers.append(ticker)
            else:
              invalid_tickers.append(ticker)
              print(f"[!] {ticker} has only NaNs from {start_date} to {end_date}")
        else:
          print(f"[!] Invalid format or missing 'Close' for batch: {batch}")

      except Exception as e:
        print(f"[X] Batch failed: {batch} - {e}")

    time.sleep(delay)

    return list(set(valid_tickers)), list(set(invalid_tickers))


def cache_tickers(tickers: list[str], out_path: str, start_date:str, end_date, validate: bool=False, APPEND_TO_CACHE:bool=False) -> None:
  """
  Stores the given tickers in a txt file within the output directory

  Args:
    tickers (list[str]): Tickers to be stored
    validate (bool): Whether to validate tickers before storage.
    out_path (bool): Where to write cache to.
    APPEND_TO_CACHE (bool): Determines whether to overwrite current cache or add list of tickers to cache.
  """
  if validate:
    tickers = validate_tickers(tickers=tickers, start_date=start_date, end_date=end_date)

  file_flag: str = "a" if APPEND_TO_CACHE else "w"

  # Add tickers to cache file
  with open(out_path, file_flag) as cache_file:
    cache_file.write("\n".join(tickers))
  print(f"{len(tickers)} have been written to {out_path}")


def load_tickers(
  start_date: str,
  end_date: str,
  cache_dir: str | None = "./cache",
  cache_filename:str | None = "tickers.cache",
  invalid_cache_filename:str | None="invalid_tickers.cache",
  index: str = "all",
  batch_size:int=25,
  request_delay:int=3,
  ) -> list[str]:
  """
  Performs pipeline to either load tickers from cache or get new set and update cache.

  Args:
    cache_dir (str | None): : Where the cache is located
      - Default is "./cache"
    cache_filename (str | None): Cache filename
      - Default is "tickers.cache"
    invalid_cache_filename (str | None): Cache filename for invalid tickers
      - Default is "invalid_tickers.cache"
    index (str): The index to request tickers from
      - Default is all
    batch_size (int): Number of tickers to check in single yfinance call.
      - Default is 25

  Returns:
    list[str] -> Validated set of tickers available in yfinance api.
  """
  cached_path: str = cache_dir + "/" + cache_filename if (cache_filename and cache_dir) else None
  print(cached_path)
  initial_tickers: list[str] = get_all_tickers(cached_path=cached_path, index=index)

  if cached_path and os.path.exists(cached_path):
    print(f"Tickers are being fetched from cache.")
    return initial_tickers
  else:
    print(f"Cache is not available, fetching new data.\n")
    validated_tickers, invalidated_tickers = validate_tickers(
      tickers=initial_tickers,
      batch_size=batch_size,
      start_date=start_date,
      end_date=end_date,
      delay=request_delay
    )
    cache_tickers(tickers=validated_tickers, out_path=cached_path, start_date=start_date, end_date=end_date)
    cache_tickers(tickers=invalidated_tickers, out_path=f"{cache_dir}/{invalid_cache_filename}", start_date=start_date, end_date=end_date)
    print(f"\nload_tickers stastistics")
    print(f"There are {len(validated_tickers)} tickers that were fetched.")
    print(f"There were {len(invalidated_tickers)} that were not retrieved.")
    print(f"The two caches share {len(set(validated_tickers) & set(invalidated_tickers))} tickers. These have been removed from the tickers you see.")

    validated_tickers = set(validated_tickers) - set(invalidated_tickers) # Another layer of validation

    # Clean current valid cache
    remove_invalid_tickers_from_cache(invalid_tickers=invalidated_tickers, cache_path=cached_path)

    return validated_tickers


def create_price_matrix(
    tickers: list[str],
    start_date: str = "2025-03-01",
    end_date: str = "2025-03-31",
    interval: str = "1d",
    sentiment_path: str | None = "./data/daily_sentiment.csv"
) -> ndarray:
    """
    Given a set of tickers, generate a price matrix that stores adjusted closing prices
    for each asset across time. Uses yfinance with auto_adjust=True.

    Args:
        tickers (list[str]): Tickers to treat as assets.
        start_date (str): Start of range.
        end_date (str): End of range.
        interval (str): Sampling interval (e.g., "1d", "1wk").
        sentiment_path (str | None): Sentiment data for a time period, used to align dates

    Returns:
        np.ndarray: Price matrix of shape (T x N), where T is timesteps and N is assets.
    """
    if sentiment_path:
      sentiment_series = load_sentiment_data(data_path=sentiment_path)

    df = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=True,  # Adjusted prices in 'Close'
        progress=False,
        threads=True,
    )

    if isinstance(df.columns, pd.MultiIndex):
        if "Close" not in df.columns.levels[0]:
            raise ValueError("'Close' not found in multi-index DataFrame.")
        df = df["Close"]
    else:
        if "Close" not in df.columns:
            raise ValueError("'Close' not found in single-index DataFrame.")
        df = df["Close"]

    # Drop rows with any missing values
    df = df.dropna()

    # Handle edge case: if only one ticker was passed, convert to 2D
    if isinstance(df, pd.Series):
        df = df.to_frame()

    price_matrix = df.to_numpy(dtype=np.float32)

    if sentiment_path:
      sentiment_series = load_sentiment_data(data_path=sentiment_path)
      sentiment_series = sentiment_series.loc[df.index]  # align by date
      sentiment_array = sentiment_series.to_numpy().reshape(-1, 1).astype(np.float32)

      assert len(sentiment_array) == len(price_matrix), "Sentiment and price data mismatch"

      return price_matrix, sentiment_array

    return price_matrix, None


def load_sentiment_data(data_path: str = "./data/daily_sentiment.csv") -> pd.DataFrame:
  df = pd.read_csv(data_path, parse_dates=["date"])
  df = df.set_index("date")
  return df["sentiment"]


def split_price_matrix(A: ndarray, sentiment_data: ndarray | None = None, train_ratio: float=.8) -> tuple[ndarray | None]:
  """
  Split price matrix into training and test set.
  """
  assert 0 < train_ratio < 1, "Train ratio is not between 0 and 1"
  T = A.shape[0]
  split_idx = int(T * train_ratio)
  train_matrix = A[:split_idx]
  test_matrix = A[split_idx:]

  if sentiment_data is not None:
    train_sentiment = sentiment_data[:split_idx]
    test_sentiment = sentiment_data[split_idx:]
    return train_matrix, test_matrix, train_sentiment, test_sentiment
  return train_matrix, test_matrix, None, None


def remove_invalid_tickers_from_cache(invalid_tickers: list[str], cache_path: str) -> None:
  """
  Given a set of `invalid_tickers` that have been found, that are currently in the valid cache, remove them from `cache_path`

  Args:
    invalid_tickers (list[str]): Tickers that have been invalidated
    cache_path (str): Where to read and write content.
  """
  current_tickers: set[str]
  with open(cache_path, "r") as cache:
    current_tickers = set(
      cache.read().splitlines()
    )
  updated_cache_content: set[str] = set(
    current_tickers - set(invalid_tickers)
  )
  with open(cache_path,"w") as cache:
    cache.write(updated_cache_content)


def sample_valid_tickers(
  tickers: list[str],
  num_stocks: int,
  start_date: str,
  end_date: str,
  cache_path: str,
  seed: int = 42,
  max_attempts:int=20,
  index: str ="nasdaq"
) -> list[str]:
  """
  Randomly sample `num_stocks` tickers and ensure they have valid price data.
  Retries until a valid set is found.
  """
  assert len(tickers) >= num_stocks, "Not enough tickers to sample from."

  random.seed(seed)

  valid_subset: list[str] = []
  attempt:int = 0

  while len(valid_subset) != num_stocks and attempt < max_attempts:
    portfolio_tickers = random.sample(tickers, k=num_stocks)

    print(f"Attempt {attempt + 1}\nTrying tickers: {portfolio_tickers}")

    valid_subset, invalid_subset = validate_tickers(
      tickers=portfolio_tickers, start_date=start_date,
      end_date=end_date,
      batch_size=num_stocks if num_stocks <= 20 else 20
    )

    remove_invalid_tickers_from_cache(invalid_tickers=invalid_subset, cache_path=cache_path)

    if len(valid_subset) == num_stocks:
      print(f"Found valid tickers: {valid_subset}")
      return valid_subset
    else:
      print(f"[!] Only {len(valid_subset)}/{num_stocks} were valid, retrying...")
      valid_subset = []

    tickers = load_tickers(
      start_date=start_date,
      end_date=end_date,
      index=index,
    )
    attempt += 1

  return valid_subset

