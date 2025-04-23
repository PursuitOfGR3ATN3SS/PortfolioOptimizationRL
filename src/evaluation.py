import numpy as np
import csv
import matplotlib.pyplot as plt

def parse_eval_metrics_to_csv(data_location:str, output_dir:str) -> None:
  """
  Parses the .npz files to get easier format for reading for analysis
  """
  data = np.load(data_location)
  csv_content: str = ""

  with open(output_dir, "w", newline="") as file:
    writer = csv.writer(file)

    # Write header
    writer.writerow(list(data.files))

    max_arr_len: int = max(len(data[key] for key in data.files))

    for i in range(max_arr_len):
      row = [i]
      for key in data.files:
        arr = data[key]
        if i < len(arr):
          val = arr[i]
          if isinstance(val, np.ndarray):
            val = val.squeeze().tolist()
          row.append(val)
        else:
          row.append("")
      writer.writerow(row)


def read_eval_csv(data_location:str) -> tuple[list[int], list[float], list[int]]:
  csv_content: list[str]
  with open(data_location, "r") as data:
    csv_content = data.read().splitlines()
  timesteps, rewards, episodes = [], [], []
  for row in csv_content:
    t, reward, episode = row.split(",")
    timesteps.append(t)
    rewards.append(reward)
    episodes.append(episode)
  return timesteps, rewards, episodes


def plot_portfolio_growth(portfolio_values) -> None:
  portfolio_growth = np.cumprod(portfolio_values)
  plt.plot(portfolio_growth)
  plt.title("Portfolio Value Over Time")
  plt.xlabel("Time Step")
  plt.ylabel("Portfolio Value")
  plt.show()


def plot_rewards_across_timesteps(timesteps, rewards) -> None:
  pass