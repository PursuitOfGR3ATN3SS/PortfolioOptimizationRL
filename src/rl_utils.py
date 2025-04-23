from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env import PortfolioEnv
import os, time
import torch
import numpy as np

def set_model(
  eval_dir:str,
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
  if os.path.exists(best_model_path):
    model = PPO.load(best_model_path, env=env)
  else:
    ppo_kwargs = ppo_kwargs
    model = PPO(policy, env, **ppo_kwargs,)

  return model


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


def train_agent(
  env: DummyVecEnv,
  best_model_path:str,
  eval_callback: EvalCallback,
  timesteps:int=100_000,
  ) -> PPO:
  print(f"Starting training...")
  start_time = time.time()
  model.learn(total_timesteps=timesteps, callback=eval_callback)
  end_time = time.time()
  print(f"Training complete in {end_time - start_time:.2f} seconds.")

  model = PPO.load(best_model_path, env=env) # Load best
  return model


def evaluate_agent(
  model: PPO,
  env: DummyVecEnv,
):
  obs, _ = env.reset()
  done = False
  total_reward: int = 0
  portfolio_values = []

  while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    portfolio_values.append(info["portfolio_return"])

  cumulative_return: float = np.exp(total_reward)
  average_daily_return: float = np.mean(portfolio_values)
  volatility: float = np.std(portfolio_values)
  return portfolio_values, total_reward, cumulative_return, average_daily_return, volatility