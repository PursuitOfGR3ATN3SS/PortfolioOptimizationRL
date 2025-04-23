# === IMPORTS ===
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils import load_tickers, create_price_matrix, split_price_matrix, sample_valid_tickers, parse_cl_args, return_cl_args
from rl_utils import set_model, set_eval_callback, train_agent, evaluate_agent
from evaluation import plot_portfolio_growth

from env import PortfolioEnv
import numpy as np
from numpy import ndarray
import torch

# === RUN AGENT ===
def main() -> None:
  # Load command line args
  args = parse_cl_args()
  num_stocks, start_date, end_date, index, seed, cache_dir, use_sentiment, best_model_path, eval_dir = return_cl_args(args=args)

  # === Load portfolios tickers ===
  # Get all  U.S. or specified index Stock tickers
  tickers: list[str] = load_tickers(
    start_date=start_date,
    end_date=end_date,
    index=index,
    cache_dir=cache_dir
  )

  # Get all tickers for portfolio
  portfolio_tickers = sample_valid_tickers(
    tickers=tickers,
    num_stocks=num_stocks,
    seed=seed,
    start_date=start_date,
    end_date=end_date,
    cache_path=f"{cache_dir}/tickers.cache"
  )

  # Create data matrix for optimization
  price_data, sentiment_data = create_price_matrix(
    tickers=portfolio_tickers,
    start_date=start_date,
    end_date=end_date,
    interval="1d",
    sentiment_path="./data/daily_sentiment.csv" if use_sentiment else None
    )

  train_prices, test_prices, train_sentiment, test_sentiment = split_price_matrix(
    A=price_data, sentiment_data=sentiment_data, train_ratio=0.85,
  )

  # print(f"Shape of train: {train_prices.shape}")
  # print(f"Train data: {train_prices}")
  # print(f"Shape of test: {test_prices.shape}")
  # print(f"Test data: {test_prices}")

  # === SETUP ENV ===
  # Create portfolio
  train_env = DummyVecEnv([lambda: Monitor(PortfolioEnv(asset_closing_prices=train_prices.copy(), sentiment_data=train_sentiment))])
  test_env = DummyVecEnv([lambda: Monitor(PortfolioEnv(asset_closing_prices=test_prices.copy(), sentiment_data=test_sentiment))])

  check_env(test_env, warn=True)

  # === SETUP ===

  model: PPO = set_model(
    eval_dir=eval_dir,
    env=train_env,
  )
  eval_callback = set_eval_callback(
    env=train_env,
    eval_dir=eval_dir,
    eval_freq=10_000,
  )

  # === TRAINING ===
  model = train_agent(
    env=train_env,
    best_model_path=best_model_path,
    eval_callback=eval_callback,
  )

  # === EVALUATION ===
  portfolio_values, total_reward, cumulative_reward, average_daily_return, volatility = evaluate_agent(
    model=model,
    env=test_env,
  )

  print(f"\n📈 Evaluation Complete")
  print(f"Total Reward (Sum of Log Returns): {total_reward:.4f}")
  print(f"Cumulative Portfolio Return: {cumulative_reward:.4f}")
  print(f"Avg Daily Return: {average_daily_return:.4f}")
  print(f"Volatility: {volatility:.4f}")

  plot_portfolio_growth(portfolio_values=portfolio_values)

if __name__ == "__main__":
  main()