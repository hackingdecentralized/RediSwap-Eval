import numpy as np
import random
import pandas as pd


# Random seed
RANDOM_SEED = 0x06511
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# Constants
THE_MERGE_BLOCK_NUMBER = 15537394
THE_MERGE_SLOT = 4700013
THE_MERGE_BLOCK_TIMESTAMP = pd.to_datetime("2022-09-15 06:42:59")
SLOT_TIME = 12


# Address to coin mapping
ADDR_TO_COINS = {
    "0x0000000000000000000000000000000000000000": ("ETH", 18),
    "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599": ("WBTC", 8),
    "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2": ("WETH", 18),
    "0x6B175474E89094C44Da98b954EedeAC495271d0F": ("DAI", 18),
    "0xdAC17F958D2ee523a2206206994597C13D831ec7": ("USDT", 6),
    "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48": ("USDC", 6),
    "0x6982508145454Ce325dDbE47a25d4ec3d2311933": ("PEPE", 18),
    "0x7D1AfA7B718fb893dB30A3aBc0Cfc608AaCfeBB0": ("MATIC", 18),
    "0x514910771AF9Ca656af840dff83E8264EcF986CA": ("LINK", 18),   
}
# Add lowercase keys
COIN_ADDRESSES = list(ADDR_TO_COINS.keys())
for k in COIN_ADDRESSES:
    ADDR_TO_COINS[k.lower()] = ADDR_TO_COINS[k]

# COINS
STABLE_COINS = ["USDC", "USDT", "DAI"]
STABLE_COIN_ADDRESSES = [k for k in ADDR_TO_COINS if ADDR_TO_COINS[k][0] in STABLE_COINS]
TARGET_COINS = ["WBTC", "BTC", "ETH", "WETH", "USDC", "USDT", "DAI", "PEPE", "MATIC", "LINK"]
TARGET_COINS_ADDRESSES = [k for k in ADDR_TO_COINS if ADDR_TO_COINS[k][0] in TARGET_COINS]


# Distribution of searchers' beliefs
# Gaussian distribution with mean and standard deviation
def generate_gaussian(max_val, min_val, num_samples=10):
    # Calculate mean and standard deviation based on the provided range
    mean = (max_val + min_val) / 2
    # Assuming 99.7% of the data falls within [min_val, max_val] (3 standard deviations)
    std_dev = (max_val - min_val) / 6

    # Generate from a Gaussian distribution
    samples = np.random.normal(loc=mean, scale=std_dev, size=num_samples)

    # Ensure samples are within the range [min_val, max_val]
    samples = np.clip(samples, min_val, max_val)

    return sorted(samples, reverse=True)


# Pareto distribution
def generate_pareto(max_val, min_val, num_samples=10):
    # Calculate the shape parameter based on the provided range
    shape = 1.5
    scale = min_val

    # Generate from a Pareto distribution
    samples = np.random.pareto(a=shape, size=num_samples) + scale

    # Ensure samples are within the range [min_val, max_val]
    samples = np.clip(samples, min_val, max_val)

    return sorted(samples, reverse=True)


# time utils
# Find next block timestamp
def find_block_timestamp(timestamp):
    time_gap = timestamp - THE_MERGE_BLOCK_TIMESTAMP.timestamp()
    slot_gap = time_gap / SLOT_TIME
    slot = int(slot_gap)
    time_gap = pd.Timedelta(slot * SLOT_TIME, unit="s")
    return THE_MERGE_BLOCK_TIMESTAMP + time_gap
