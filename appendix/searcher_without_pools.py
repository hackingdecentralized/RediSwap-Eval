import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns


CUR_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CUR_DIR)
DATA_DIR = os.path.join(PARENT_DIR, "data")
FIGURES_DIR = os.path.join(PARENT_DIR, "figures")


def plot(df, filename, palette):
    plt.clf()
    plt.figure(figsize=(16, 9), dpi=100)
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x="month", y="ratio", data=df, color=palette)
    plt.xlabel(None)
    plt.ylabel("Percentage", fontsize=36)
    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)
    plt.ylim(0, 100)
    ax.set_xticks(ax.get_xticks()[::2])
    ax.set_xticklabels([label.get_text()[2:7] for label in ax.get_xticklabels()], fontsize=36)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', fontsize=30, padding=3)
    plt.savefig(filename, bbox_inches='tight')


if __name__ == "__main__":
    # Data from Allium
    # Date Range: 2023-09-01 00:00:00 - 2024-08-31 23:59:59
    uniswapx_df = pd.read_csv(os.path.join(DATA_DIR, "uniswapx_count.csv"))
    cowswap_df = pd.read_csv(os.path.join(DATA_DIR, "cowswap_count.csv"))

    uniswapx_df["ratio"] = uniswapx_df["unique_searcher_ratio"]*100
    cowswap_df["ratio"] = cowswap_df["unique_searcher_ratio"]*100

    uniswapx_output = os.path.join(FIGURES_DIR, "uniswapx_searcher_without_pools.pdf")
    cowswap_output = os.path.join(FIGURES_DIR, "cowswap_searcher_without_pools.pdf")

    palettes = sns.color_palette()
    plot(uniswapx_df, uniswapx_output, palettes[0])
    plot(cowswap_df, cowswap_output, palettes[1])
