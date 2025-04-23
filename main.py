# === IMPORTS ===
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import argparse
import os, sys, random, time
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from utils import validate_tickers, load_tickers, create_price_matrix, split_price_matrix, plot_portfolio_growth, sample_valid_tickers
from env import PortfolioEnv
import numpy as np
from numpy import ndarray
import torch


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
  return parser.parse_args()

def return_cl_args(args) -> tuple[any]:
  return args.num_portfolio_stocks, args.start_date, args.end_date, args.stock_index, args.random_seed


# === RUN AGENT ===
def main() -> None:
  # Load command line args
  args = parse_cl_args()
  num_stocks, start_date, end_date, index, seed = return_cl_args(args=args)

  # === Load portfolios tickers ===
  # Get all  U.S. or specified index Stock tickers
  tickers: list[str] = load_tickers(index=index)

  # Get all tickers for portfolio
  portfolio_tickers = sample_valid_tickers(
    tickers=tickers,
    num_stocks=num_stocks,
    seed=seed,
    start_date=start_date,
    end_date=end_date,
  )

  # Create data matrix for optimization
  price_data, sentiment_data = create_price_matrix(
    tickers=portfolio_tickers,
    start_date=start_date,
    end_date=end_date,
    interval="1d",
    sentiment_path="./data/daily_sentiment.csv"
    )

  train_prices, test_prices, train_sentiment, test_sentiment = split_price_matrix(
    A=price_data, sentiment_data=sentiment_data, train_ratio=0.85,
  )

  # === SETUP ENV ===
  # Create portfolio
  train_env = DummyVecEnv([lambda: Monitor(PortfolioEnv(asset_closing_prices=train_prices.copy(), sentiment_data=train_sentiment))])
  test_env = DummyVecEnv([lambda: Monitor(PortfolioEnv(asset_closing_prices=test_prices.copy(), sentiment_data=test_sentiment))])

  check_env(test_env, warn=True)

  # === SETUP MODEL ===
  # Optimize portfolio
  previous_model_path: str = "./best_model/best_model.zip"
  eval_dir = "./best_model/"

  if os.path.exists(previous_model_path):
    model = PPO.load(previous_model_path, env=train_env)
  else:
    ppo_kwargs = {
    "n_steps": 128,
    "batch_size": 64,
    "learning_rate": 3e-4,
    "ent_coef": 0.01,
    "device": torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    "verbose": 1
    }
    model = PPO("MlpPolicy", train_env, **ppo_kwargs,)

  # === CALLBACK ===
  eval_callback = EvalCallback(
    train_env,
    best_model_save_path=eval_dir,
    # log_path=log_dir,
    eval_freq=1000,
    deterministic=True,
    render=False
  )

  # === TRAINING ===
  print(f"Starting training...")
  start_time = time.time()
  model.learn(total_timesteps=100_000, callback=eval_callback)
  end_time = time.time()
  print(f"Training complete in {end_time - start_time:.2f} seconds.")

  model = PPO.load(previous_model_path, env=train_env) # Load best model

  # === EVALUATION ===
  obs, _ = test_env.reset()
  done = False
  total_reward = 0
  portfolio_values = []

  while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = test_env.step(action)
    total_reward += reward
    portfolio_values.append(info["portfolio_return"])

  print(f"\n📈 Evaluation Complete")
  print(f"Total Reward (Sum of Log Returns): {total_reward:.4f}")
  print(f"Cumulative Portfolio Return: {np.exp(total_reward):.4f}")
  print(f"Avg Daily Return: {np.mean(portfolio_values):.4f}")

  plot_portfolio_growth(portfolio_values=portfolio_values)

if __name__ == "__main__":
  main()