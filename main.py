from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from utils import get_all_tickers, validate_tickers, load_tickers, create_price_matrix
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from env import PortfolioEnv
import random
import numpy as np
from numpy import ndarray
import torch
import matplotlib.pyplot as plt
import time

def plot_portfolio_growth(portfolio_values) -> None:
  portfolio_growth = np.cumprod(portfolio_values)
  plt.plot(portfolio_growth)
  plt.title("Portfolio Value Over Time")
  plt.xlabel("Time Step")
  plt.ylabel("Portfolio Value")
  plt.show()

def main(
  num_stocks: int = 20,
  seed: int = 42,
  ) -> None:

  # Get all  U.S. or specified index Stock tickers
  tickers: list[str] = load_tickers()
  assert len(tickers) > 0, "No tickers were fetched."
  max_num_tickers: int = len(tickers)

  # Get all tickers for portfolio
  assert max_num_tickers > num_stocks, "There are more stocks requested than valid tickers"
  random.seed(seed)
  portfolio_tickers: list[str] = random.sample(tickers, k=num_stocks) # All stock tickers that will be apart of the portfolio
  print(portfolio_tickers)

  # Create data matrix for optimization
  price_data: ndarray = create_price_matrix(tickers=portfolio_tickers, start_date="2025-03-01", end_date="2025-03-31", interval="1d")

  # Create portfolio
  env = PortfolioEnv(asset_closing_prices=price_data)
  check_env(env, warn=True)

  # Optimize portfolio
  model_path = "./best_model/best_model.zip"
  log_dir = "./logs/"
  eval_dir = "./best_model/"

  if os.path.exists(model_path): model = PPO.load(model_path, env=env)
  else:
    ppo_kwargs = {
    "n_steps": 128,
    "batch_size": 64,
    "learning_rate": 3e-4,
    "ent_coef": 0.01,
    "device": torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    "verbose": 1
}
    model = PPO("MlpPolicy", env, **ppo_kwargs,)
  env = DummyVecEnv([lambda: Monitor(PortfolioEnv(asset_closing_prices=price_data.copy()))])
  eval_callback = EvalCallback(env,
                                best_model_save_path=eval_dir,
                                # log_path=log_dir,
                                eval_freq=1000,
                                deterministic=True,
                                render=False
                                )
  start_time = time.time()
  print(f"Timing starting")
  model.learn(total_timesteps=100_000, callback=eval_callback)
  model = PPO.load(model_path, env=env)
  end_time = time.time()
  print(f"Training took {end_time - start_time}")

  # Evalaution
  env = env.envs[0]
  obs, _ = env.reset()[0]
  done = False
  total_reward = 0
  portfolio_values = []

  while not done:
    action, _ = model.predict(obs, deterministic=True)  # Use greedy policy for evaluation
    obs, reward, done, info = env.step(action)
    total_reward += reward
    portfolio_values.append(info["portfolio_return"])

  print(f"\n📈 Evaluation Complete")
  print(f"Total Reward (Sum of Log Returns): {total_reward:.4f}")
  print(f"Cumulative Portfolio Return: {np.exp(total_reward):.4f}")
  print(f"Avg Daily Return: {np.mean(portfolio_values):.4f}")

  plot_portfolio_growth(portfolio_values=portfolio_values)

if __name__ == "__main__":
  main()