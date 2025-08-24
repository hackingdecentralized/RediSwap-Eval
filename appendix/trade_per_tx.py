import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns


CUR_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CUR_DIR)
FIGURES_DIR = os.path.join(PARENT_DIR, "figures")


def plot(df, filename, palette):
    plt.clf()
    plt.figure(figsize=(16, 9), dpi=100)
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x="range", y="ratio", data=df, color=palette)
    plt.xlabel(None)
    plt.ylabel("Percentage", fontsize=36)
    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)
    plt.ylim(0, 100)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=30, padding=3)
    plt.savefig(filename, bbox_inches='tight')


if __name__ == "__main__":
    # Data from Allium
    # Date Range: 2023-09-01 00:00:00 - 2024-08-31 23:59:59
    columns = ["range", "number_of_trades"]
    cowswap_data = [
        ('1', 389858),
        ('2', 89277),
        ('3', 19792),
        ('4', 5261),
        ('>=5', 1651+840),
    ]

    uniswapx_data = [
        ('1', 723024),
        ('2', 1423),
        ('3', 188),
        ('4', 48),
        ('>=5', 19+10),
    ]

    uniswapx_df = pd.DataFrame(uniswapx_data, columns=columns)
    cowswap_df = pd.DataFrame(cowswap_data, columns=columns)

    uniswapx_df["ratio"] = uniswapx_df["number_of_trades"] / uniswapx_df["number_of_trades"].sum() * 100
    cowswap_df["ratio"] = cowswap_df["number_of_trades"] / cowswap_df["number_of_trades"].sum() * 100

    uniswapx_output = os.path.join(FIGURES_DIR, "uniswapx_order_per_tx.pdf")
    cowswap_output = os.path.join(FIGURES_DIR, "cowswap_order_per_tx.pdf")

    palettes = sns.color_palette()
    plot(uniswapx_df, uniswapx_output, palettes[0])
    plot(cowswap_df, cowswap_output, palettes[1])
