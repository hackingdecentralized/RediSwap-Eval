import os
import pandas as pd


base_url = "https://data.binance.vision/data/spot/monthly/klines/{}/1s/{}-1s-{}.zip"

# Destination path
cur_dir = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(cur_dir, "data")
binance_path = os.path.join(data_path, "binance")
if not os.path.exists(binance_path):
    os.makedirs(binance_path)

# Token pairs
TOKEN_PAIRS = ["BTCDAI", "BTCUSDC", "BTCUSDT", "ETHDAI", "ETHUSDC", "ETHUSDT", "PEPEUSDC", "PEPEUSDT", "USDCUSDT", "USDTDAI", "LINKUSDT", "LINKUSDC", "MATICUSDT", "MATICUSDC", "DOGEUSDT"]

# Download data from 2023-09-01 to 2024-08-31
date_ranges = pd.date_range(start='2023-09-01', end='2024-08-31', freq='MS')
for date in date_ranges:
    month_str = date.strftime("%Y%m")
    month_path = date.strftime("%Y-%m")
    month_dir = os.path.join(binance_path, month_str)
    if not os.path.exists(month_dir):
        os.makedirs(month_dir)

    for token_pair in TOKEN_PAIRS:
        url = base_url.format(token_pair, token_pair, month_path)    
        print(f"{month_dir}/{token_pair}.zip")
        if os.path.exists(f"{month_dir}/{token_pair}.zip"):
            print(f"Skipping {url} as it already exists")
            continue

        print(f"Downloading {url} to {month_dir}/{token_pair}.zip")
        os.system(f"wget {url} -O {month_dir}/{token_pair}.zip")
