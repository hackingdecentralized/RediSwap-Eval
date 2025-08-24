import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns


CUR_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CUR_DIR)
DATA_PATH = os.path.join(PARENT_DIR, "data")
BINANCE_DATA_PATH = os.path.join(DATA_PATH, "binance")
COLUMNS = "OpenTime,OpenPrice,HighPrice,LowPrice,ClosePrice,Volume,CloseTime,BaseAssetVolume,NumberOfTrades,TakerBuyVolume,TakerBuyBaseAssetVolume,Ignore".split(",")


# Load monthly prices from Binance
def load_monthly_prices(month_str):
    month_path = os.path.join(BINANCE_DATA_PATH, month_str)
    binance_prices = {}
    price_files = os.listdir(month_path)
    for price_file in price_files:
       price_file_path = os.path.join(month_path, price_file)
       print("Loading price data: ", price_file_path)
       if os.path.getsize(price_file_path) == 0:
            continue
       df = pd.read_csv(price_file_path, compression="zip", header=None, names=COLUMNS)
       df.set_index("OpenTime", inplace=True)
       binance_prices[price_file.replace(".zip", "")] = df.sort_values("OpenTime")

    return binance_prices


# Find the Binance price to the given timestamp
def find_binance_price(start_time, end_time, df):
    result = df.loc[int(start_time * 1000):int(end_time * 1000)]
    return result["HighPrice"].max(), result["LowPrice"].min()


# Compute the delta between the max and second max phi
def compute_delta(delta_x, delta_y, samples):
    phis = [i*delta_x+delta_y for i in samples]
    phis = sorted(phis, reverse=True)
    max_phi = phis[0]
    second_max_phi = phis[1]
    return max_phi - second_max_phi, second_max_phi


# Plot the results
def plot(df, filename):
    plt.clf()
    plt.figure(figsize=(16, 10), dpi=100)
    sns.set_theme(style="whitegrid")
    ax = sns.lineplot(
        data=df,
        x='Fee',
        y='Ratio',
        hue='Distribution',
        size='Arbitrageurs (#)',
        sizes=(3, 6),
        style='Distribution',
        markers=True,
        dashes=False,
        markersize=16,
        legend=False
    )
    plt.xlim(0, 0.55)
    plt.ylim(0, 100)
    plt.yticks(fontsize=40)
    plt.xticks(fontsize=40)
    plt.xlabel("Swap Fee (%)", fontsize=40)
    plt.ylabel("Better Execution Ratio (%)", fontsize=40)

    distribution_handles = []
    sample_handles = []
    distribution_handles.append(plt.Line2D([], [], color=sns.color_palette("tab10")[0],  linestyle='-', linewidth=4, marker='o', markersize=16, label='Gaussian'))
    distribution_handles.append(plt.Line2D([], [], color=sns.color_palette("tab10")[1],  linestyle='-', linewidth=4, marker='x', markersize=16, markeredgewidth=2, label='Pareto'))
    sample_handles.append(plt.Line2D([], [], color='black', linewidth=3, label='3'))
    sample_handles.append(plt.Line2D([], [], color='black', linewidth=4.5, label='10'))
    sample_handles.append(plt.Line2D([], [], color='black', linewidth=6, label='20'))


    legend2 = plt.legend(handles=distribution_handles, loc='upper right', bbox_to_anchor=(1, 1), title='Distribution', fontsize=34, title_fontsize=36)
    plt.gca().add_artist(legend2)
    legend1 = plt.legend(handles=sample_handles, loc='upper right', bbox_to_anchor=(1, 0.7), title='Arbitrageurs (#)', fontsize=34, title_fontsize=36)


    legend2_width = legend2.get_frame().get_width() 
    legend1.get_frame().set_width(legend2_width)

    plt.savefig(filename, bbox_inches='tight')

