import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import time

# import utils
from utils import *


# Destination path
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CUR_DIR)
DATA_PATH = os.path.join(PARENT_DIR, "data")
FIGURES_PATH = os.path.join(PARENT_DIR, "figures")
BINANCE_DATA_PATH = os.path.join(DATA_PATH, "binance")

# Constants
SAMPLES = [3, 10, 20]


# "weth-usdc-cdf.pdf", "weth-usdt-cdf.pdf"
def plot(df, filename):
    plt.clf()
    plt.figure(figsize=(16, 10))
    sns.set_style('whitegrid')
    axes = []
    for sample, group in df.groupby('num_samples'):
        if sample == 3:
            line_width = 3
            line_style = '-'
        elif sample == 10:
            line_width = 4.5
            line_style = '--'
        else:
            line_width = 6
            line_style = '-.'
        
        ax = sns.ecdfplot(data=group, x='ratio', hue='distribution', palette="tab10", linewidth=line_width, label=f"{sample}", linestyle=line_style, legend=False)
        axes.append(ax)

    distribution_handles = []
    sample_handles = []
    distribution_handles.append(plt.Line2D([], [], color=sns.color_palette("tab10")[0],  linestyle='-', linewidth=4, label='Gaussian'))
    distribution_handles.append(plt.Line2D([], [], color=sns.color_palette("tab10")[1],  linestyle='-', linewidth=4, label='Pareto'))
    sample_handles.append(plt.Line2D([], [], color='black', linestyle='-', linewidth=3, label='3'))
    sample_handles.append(plt.Line2D([], [], color='black', linestyle='--', linewidth=4.5, label='10'))
    sample_handles.append(plt.Line2D([], [], color='black', linestyle='-.', linewidth=6, label='20'))


    legend1 = plt.legend(handles=sample_handles, loc='lower right', bbox_to_anchor=(1, 0), title='Arbitrageurs (#)', fontsize=34, title_fontsize=36)
    plt.gca().add_artist(legend1)

    legend2 = plt.legend(handles=distribution_handles, loc='lower right', bbox_to_anchor=(1, 0.4), title='Distribution', fontsize=34, title_fontsize=36)
    legend1_width = legend1.get_frame().get_width() 
    legend2.get_frame().set_width(legend1_width)

    plt.xlabel(None)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.ylabel('CDF', fontsize=40)
    plt.ylim(0, 1)
    plt.xlabel('LVR Reduction Ratio (%)', fontsize=40)
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')


# ETHUSDC, ETHUSDT
def load_binance_prices(token_pair):
    COLUMNS = "OpenTime,OpenPrice,HighPrice,LowPrice,ClosePrice,Volume,CloseTime,BaseAssetVolume,NumberOfTrades,TakerBuyVolume,TakerBuyBaseAssetVolume,Ignore".split(",")

    month_range = pd.date_range(start="2023-09-01", end="2024-08-31", freq="MS")
    binance_df = []
    for month in month_range:
        month_str = month.strftime("%Y%m")
        month_path = os.path.join(BINANCE_DATA_PATH, month_str)
        zip_file = os.path.join(month_path, f"{token_pair}.zip")
        month_df = pd.read_csv(zip_file, compression='zip', names=COLUMNS, header=None)
        binance_df.append(month_df)
    binance_df = pd.concat(binance_df)
    binance_df.set_index("OpenTime", inplace=True)
    return binance_df


def find_cex_price(start_time, end_time, binance_df):
    sub_df = binance_df.loc[int(start_time*1000):int(end_time*1000)]
    open_price = sub_df["OpenPrice"].iloc[0]
    close_price = sub_df["ClosePrice"].iloc[-1]
    high_price = sub_df["HighPrice"].max()
    low_price = sub_df["LowPrice"].min()
    return (open_price+close_price+high_price+low_price)/4, high_price, low_price 


def compute_lvr(prev_reserve0, prev_reserve1, reserve0, reserve1, price):
    return (price * prev_reserve0 + prev_reserve1) - (price * reserve0 + reserve1)


def simulate(token_df, binance_df):
    data = []
    for i, row in token_df.iterrows():
        if i == 0:
            continue
        prev_row = token_df.iloc[i-1]
        reverse0 = int(row['reserve0']) / 1e18
        reverse1 = int(row['reserve1']) / 1e6
        prev_reverse0 = int(prev_row['reserve0']) / 1e18
        prev_reverse1 = int(prev_row['reserve1']) / 1e6
        a = prev_reverse0 - reverse0
        prev_timestamp = prev_row['timestamp']
        timestamp = row['timestamp']

        avg_cex_price, high_price, low_price = find_cex_price(prev_timestamp, timestamp, binance_df)

        for num_samples in SAMPLES:
            # a > 0: sell ETH, buy USDC/USDT
            if a == 0:
                # Skip the case where a = 0
                continue
            elif a > 0:
                gaussian_samples = generate_gaussian(high_price, low_price, num_samples)
                pareto_samples = generate_pareto(high_price, low_price, num_samples)
            else:
                gaussian_samples = generate_gaussian(-low_price, -high_price, num_samples)
                gaussian_samples = [-i for i in gaussian_samples]
                pareto_samples = generate_pareto(-low_price, -high_price, num_samples)
                pareto_samples = [-i for i in pareto_samples]


            for distribution, name in zip([gaussian_samples, pareto_samples], ["Gaussian", "Pareto"]):
                lvrs = []
                for price in distribution:
                    k = prev_reverse0 * prev_reverse1
                    new_reverse0 = math.sqrt(k/price)
                    new_reverse1 = k / new_reverse0
                    new_a = prev_reverse0-new_reverse0
                    new_lvr = compute_lvr(prev_reverse0, prev_reverse1, new_reverse0, new_reverse1, price)
                    lvrs.append(new_lvr)

                
                lvrs = sorted(lvrs, reverse=True)
                lvrs = [max(lvr, 0) for lvr in lvrs]
                max_lvr = lvrs[0]
                second_lvr = lvrs[1]

                data.append((row['block'], num_samples, name, a, max_lvr, second_lvr, max_lvr-second_lvr, row["timestamp"]))
        
        if row["block"] % 50000 == 0:
            print(f"currently process block {row['block']}")


    df = pd.DataFrame(data, columns=["block", "num_samples", "distribution", "a", "max_lvr", "second_lvr", "actual_lvr", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df['date'] = df['timestamp'].dt.date
    df['ratio'] = 100 * df['actual_lvr'] / df['max_lvr']

    return df


if __name__ == "__main__":
    # Token pairs (USDT - ETH)
    print("Loading Uniswap data...")
    t = time.time()
    usdt_uniswapv2_file = os.path.join(DATA_PATH, "UniswapV2-USDT.csv.zip")
    usdt_uniswapv2_df = pd.read_csv(usdt_uniswapv2_file, compression='zip')

    # Token pairs (USDC - WETH)
    usdc_uniswapv2_file = os.path.join(DATA_PATH, "UniswapV2-USDC.csv.zip")
    usdc_uniswapv2_df = pd.read_csv(usdc_uniswapv2_file, compression='zip')
    columns = usdc_uniswapv2_df.columns.tolist()
    idx1, idx2 = columns.index('reserve0'), columns.index('reserve1')
    columns[idx1], columns[idx2] = columns[idx2], columns[idx1]
    usdc_uniswapv2_df.columns = columns

    print(f"✅ Uniswap data loaded. Time taken: {time.time() - t:.2f}s")

    # Load prices
    print("Loading Binance data...It may take a while...")
    t = time.time()
    usdt_binance_df = load_binance_prices("ETHUSDT")
    usdc_binance_df = load_binance_prices("ETHUSDC")
    print(f"✅ Binance data loaded. Time taken: {time.time() - t:.2f}s")
    
    # Simulate
    print("🚀🚀🚀 Simulating USDT...")
    t = time.time()
    usdt_df = simulate(usdt_uniswapv2_df, usdt_binance_df)
    print(f"✅ Simulation for USDT done. Time taken: {time.time() - t:.2f}s")
    usdt_output_file = os.path.join(FIGURES_PATH, "eth-usdt-cdf.pdf")
    plot(usdt_df, usdt_output_file)
    
    print("Simulating USDC...")
    t = time.time()
    usdc_df = simulate(usdc_uniswapv2_df, usdc_binance_df)
    print(f"✅ Simulation for USDC done. Time taken: {time.time() - t:.2f}s")
    usdc_output_file = os.path.join(FIGURES_PATH, "eth-usdc-cdf.pdf")
    plot(usdc_df, usdc_output_file)
    
    print("🎉🎉🎉 All done!")