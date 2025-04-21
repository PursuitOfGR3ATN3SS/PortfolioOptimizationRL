from gymnasium  import Env, spaces
import numpy as np
from numpy import ndarray

class PortfolioEnv(Env):
  """
  Portfolio Environment where agent learns to optimize allocations given the set of stocks.
  """

  def __init__(self, asset_closing_prices: ndarray, initial_cash: float = 1_000_000.0) -> None:
    """
    Args:
      asset_closing_prices (np.ndarray): The adjusted closing price of assets across time
        - Has shape (T, N)
          - T is the number of timesteps
          - N is the number of assets
      initial_cash (float): Funds provided to allocator before trades.
        - Default is 1_000_000.00
    """
    super().__init__()
    self.asset_closing_prices = asset_closing_prices
    self.initial_cash_amount = initial_cash
    self.num_timesteps, self.num_assets = self.asset_closing_prices.shape

    # RL parameters
    self.action_space = spaces.Box(low=0, high=1, shape=(self.num_assets, ),dtype=np.float32)
    self.observation_space = spaces.Box(
      low=0,
      high=1,
      shape=(self.num_assets * 2 + 1, ), # Normalized prices for each asset, Each assets weights within the portfolio, normalized cash balance
      dtype=np.float32
    )

    # State
    self.current_step: int = 0
    self.current_cash_amount: float = self.initial_cash_amount
    self.weights: ndarray = np.ones(self.num_assets, dtype=np.float32,) / self.num_assets

  def _get_obs(self) -> ndarray:
    """
    Construct the observation vector.

    Returns:
      observation (ndarray): Normalized prices + current weights + normalized cash
    """
    current_closing_prices = self.asset_closing_prices[self.current_step]
    normalized_prices = current_closing_prices / np.max(current_closing_prices)
    normalized_cash = self.current_cash_amount / self.initial_cash_amount

    observation = np.concatenate([normalized_prices, self.weights, [normalized_cash]]).astype(np.float32)
    return observation

  def reset(
    self,
    *,
    seed: int | None = None,
    return_info: bool = False,
    options: dict | None = None
) -> ndarray | tuple[ndarray, dict]:
    """
    Reset environment to initial state and return the first observation.

    Args:
      seed (int, optional): Random seed
      return_info (bool): Whether to return additional info
      options (dict, optional): Extra options (not used)

    Returns:
      ndarray | tuple[ndarray, dict]:
        - If `return_info` is False: returns the initial observation (state) as a NumPy array.
        - If `return_info` is True: returns a tuple (observation, info_dict), where info_dict contains auxiliary information.
    """
    super().reset(seed=seed)
    self.current_step = 0
    self.current_cash_amount = self.initial_cash_amount
    self.weights = np.ones(self.num_assets, dtype=np.float32) / self.num_assets
    return self._get_obs(), {}

  def step(self, action: ndarray) -> tuple[ndarray, float, bool, bool, dict]:
    """
    Execute one time step within the environment.

    Args:
      action (ndarray): The new portfolio allocation (weights), shape (num_assets,)

    Returns:
      observation (ndarray): Next state
      reward (float): Reward signal (log return)
      done (bool): Whether episode is complete
      info (dict): Additional info (e.g, portfolio return)
    """

    # Normalize action to sum weights to 1
    action = np.clip(action, 0, 1)
    if action.sum() == 0:
        action = np.ones_like(action) / len(action)  # fallback to uniform allocation
    else:
        action /= action.sum()

    # Get price movement from t -> t + 1
    previous_closing_prices = self.asset_closing_prices[self.current_step]
    self.current_step += 1
    next_closing_prices = self.asset_closing_prices[self.current_step]

    # Calculate price relative movement (next price / current price)
    price_relatives = next_closing_prices / previous_closing_prices

    # Calculate portfolio return (weighted by new action)
    portfolio_return = np.dot(action, price_relatives)

    # Use the log of the return for the return
    reward = float(np.log(portfolio_return))

    # Update internal state
    self.weights = action

    # Check if the episode is over
    done = self.current_step >= self.num_timesteps - 1

    # Create the next observation
    observation = self._get_obs()

    # Track return
    info = {"portfolio_return": portfolio_return}

    return observation, reward, done, False, info