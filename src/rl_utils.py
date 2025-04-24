from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env import PortfolioEnv
import os, time
import torch
import numpy as np
from numpy import ndarray

def set_model(
  env: DummyVecEnv,
  ppo_kwargs:dict[str, int|float|str] = {
    "n_steps": 128,
    "batch_size": 64,
    "learning_rate": 3e-4,
    "ent_coef": 0.01,
    "device": torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    "verbose": 1
    },
  best_model_path:str | None = None,
  policy: str = "MlpPolicy"
  ) -> any:

  model: PPO
  loaded: bool
  best_model_path = f"{best_model_path}.zip"
  if os.path.exists(best_model_path):
    model = PPO.load(best_model_path, env=env)
    loaded = True
  else:
    ppo_kwargs = ppo_kwargs
    model = PPO(policy, env, **ppo_kwargs,)
    loaded=False

  return model, loaded


def set_eval_callback(
  env: DummyVecEnv,
  eval_dir:str,
  eval_freq:int=1000,
  deterministic:bool = True,
  render: bool=False,
)-> EvalCallback:
  eval_callback = EvalCallback(
    env,
    best_model_save_path=eval_dir,
    eval_freq=eval_freq,
    deterministic=deterministic,
    render=render
  )
  return eval_callback


def set_env(train_prices: ndarray, test_prices: ndarray, train_sentiment: ndarray|None, test_sentiment: ndarray|None) -> tuple[PortfolioEnv, PortfolioEnv, DummyVecEnv, DummyVecEnv]:
  train_env: PortfolioEnv = PortfolioEnv(asset_closing_prices=train_prices.copy(), sentiment_data=train_sentiment)
  check_env(env=train_env, warn=True)
  test_env: PortfolioEnv = PortfolioEnv(asset_closing_prices=test_prices.copy(), sentiment_data=test_sentiment)
  vec_train_env = DummyVecEnv([lambda: Monitor(train_env)])
  vec_test_env = DummyVecEnv([lambda: Monitor(test_env)])
  return train_env, test_env, vec_train_env, vec_test_env


def train_agent(
  model: PPO,
  env: DummyVecEnv,
  best_model_dir:str,
  eval_callback: EvalCallback,
  timesteps:int=100_000,
  ) -> PPO:
  best_model_path = f"{best_model_dir}.zip"
  print(f"Starting training...")
  start_time = time.time()
  model.learn(total_timesteps=timesteps, callback=eval_callback, progress_bar=True)
  end_time = time.time()
  model.save(best_model_path)
  print(f"Training complete in {end_time - start_time:.2f} seconds.")

  time.sleep(5) # Delay to prevent access to recently written file

  model = PPO.load(best_model_path, env=env) # Load best model throughout training
  return model


def evaluate_agent(
  model: PPO,
  env: DummyVecEnv,
):
  obs = env.reset()
  done = False
  total_reward: int = 0
  rewards = []
  portfolio_values = []

  while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    reward = reward[0]
    done = done[0]
    info = info[0]
    total_reward += reward
    rewards.append(reward)
    portfolio_values.append(info["portfolio_return"])

  portfolio_values = np.array(portfolio_values) - 1
  cumulative_return: float = np.exp(total_reward)
  average_daily_return: float = np.mean(portfolio_values)
  volatility: float = np.std(portfolio_values)
  return portfolio_values, total_reward, cumulative_return, average_daily_return, volatility, rewards