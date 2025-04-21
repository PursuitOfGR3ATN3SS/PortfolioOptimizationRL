from stable_baselines3.common.env_checker import check_env
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from env import PortfolioEnv  # or wherever your env is defined

# Create dummy price data for test
import numpy as np
dummy_prices = np.random.rand(100, 3).astype(np.float32)  # 100 days, 3 assets

# Initialize your environment
env = PortfolioEnv(asset_closing_prices=dummy_prices)

# Run the check
check_env(env, warn=True)  # warn=True = show warnings instead of errors
