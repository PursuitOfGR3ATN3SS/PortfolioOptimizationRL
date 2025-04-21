import yfinance as yf
import random
import requests
import numpy as np
from numpy import ndarray
import os
import pandas as pd
import time

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
    batch_size: int = 25,
    interval: str = "1d",
    delay: int = 5,
    period: str = "1mo"
) -> list[str]:
    """
    Removes any tickers that are not available in the yfinance API.

    Args:
        tickers (list[str]): Tickers to be evaluated.
        batch_size (int): Number of tickers to request per call.
        interval (str): Sampling interval, e.g. '1d', '1m'.
        delay (int): Delay between batches (in seconds).
        period (str): How far back to retrieve data (only used with interval).

    Returns:
        list[str]: Tickers that return valid, non-empty price data.
    """
    n = len(tickers)
    assert batch_size <= n, f"The given batch size ({batch_size}) is larger than the number of tickers ({n})."

    valid_tickers = []

    for i in range(0, n, batch_size):
      batch = tickers[i:i + batch_size]
      try:
        data = yf.download(batch, interval=interval, period=period, progress=False, threads=True)
        if isinstance(data.columns, pd.MultiIndex):
          if "Close" in data.columns.levels[0]:
              clean_tickers = data["Close"].dropna(axis=1).columns.tolist()
              valid_tickers.extend(clean_tickers)
          else:
              print(f"No 'Close' data found in batch: {batch}")
        else:
            print(f"Unexpected format for batch: {batch}")
      except Exception as e:
          print(f"Batch failed: {batch} - {e}")

      time.sleep(delay)

    return list(set(valid_tickers))


def cache_tickers(tickers: list[str], out_path: str, validate: bool=True) -> None:
  """
  Stores the given tickers in a txt file within the output directory

  Args:
    tickers (list[str]): Tickers to be stored
    validate (bool): Whether to validate tickers before storage.
    out_path (bool): Where to write cache to.
  """
  if validate:
    tickers = validate_tickers(tickers=tickers)

  # Add tickers to cache file
  with open(out_path, "w") as cache_file:
    cache_file.write("\n".join(tickers))


def load_tickers(cache_dir: str | None = "./cache/", cache_filename:str | None = "tickers.cache", index: str = "all", batch_size:int=25) -> list[str]:
  """
  Performs pipeline to either load tickers from cache or get new set and update cache.

  Args:
    cache_dir (str | None): : Where the cache is located
      - Default is "./cache"
    cache_filename (str | None): Cache filename
      - Default is "tickers.cache"
    index (str): The index to request tickers from
      - Default is all
    batch_size (int): Number of tickers to check in single yfinance call.
      - Default is 25

  Returns:
    list[str] -> Validated set of tickers available in yfinance api.
  """
  cached_path: str = cache_dir + "/" + cache_filename if (cache_filename and cache_dir) else None
  inital_tickers: list[str] = get_all_tickers(cached_path=cached_path, index=index)

  if cached_path and os.path.exists(cached_path):
    return inital_tickers
  else:
    validated_tickers: list[str] = validate_tickers(tickers=inital_tickers)
    cache_tickers(tickers=validated_tickers, out_path=cached_path)
    return validated_tickers


def create_price_matrix(
    tickers: list[str],
    start_date: str = "2025-03-01",
    end_date: str = "2025-03-31",
    interval: str = "1d"
) -> ndarray:
    """
    Given a set of tickers, generate a price matrix that stores adjusted closing prices
    for each asset across time. Uses yfinance with auto_adjust=True.

    Args:
        tickers (list[str]): Tickers to treat as assets.
        start_date (str): Start of range.
        end_date (str): End of range.
        interval (str): Sampling interval (e.g., "1d", "1wk").

    Returns:
        np.ndarray: Price matrix of shape (T x N), where T is timesteps and N is assets.
    """
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

    return df.to_numpy(dtype=np.float32)


